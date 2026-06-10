"""
Perovskite Defect Detection -- FastAPI Backend
==============================================
Architecture:
- Model loads in a dedicated subprocess (worker.py) at startup.
- Main uvicorn process communicates via multiprocessing.Queue.
- Endpoint is a plain `def` (thread pool) so it never blocks the event loop.
"""

import io
import sys
import os
import json
import base64
import logging
import traceback
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = Path(__file__).resolve().parent.parent / "server.log"

console_handler = logging.StreamHandler(sys.stdout)
file_handler    = logging.FileHandler(str(LOG_FILE), mode="w", encoding="utf-8")
fmtr = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s")
console_handler.setFormatter(fmtr)
file_handler.setFormatter(fmtr)
logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
log = logging.getLogger("perovskite")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
METRICS_DIR  = PROJECT_ROOT / "src" / "runs" / "metrics"
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
WORKER_PY      = Path(__file__).resolve().parent / "worker.py"
WORKER_ONNX_PY = Path(__file__).resolve().parent / "worker_onnx.py"

log.info("=" * 60)
log.info("Perovskite Defect Detector -- Backend Starting")
log.info("=" * 60)
log.info(f"PROJECT_ROOT : {PROJECT_ROOT}")

# Find YOLO11m weights — prefer .onnx (fast) over .pt (slow torch)
all_weights = sorted(PROJECT_ROOT.glob("src/runs/detect/*/weights/best.pt"))
BEST_WEIGHT = str(all_weights[1]) if len(all_weights) > 1 else str(all_weights[0]) if all_weights else ""
for w in all_weights:
    if "yolo11m" in str(w):
        BEST_WEIGHT = str(w)
        break

# Check if ONNX version exists next to the .pt file
BEST_ONNX = BEST_WEIGHT.replace("best.pt", "best.onnx")
if Path(BEST_ONNX).exists():
    USE_ONNX = True
    log.info(f"ONNX model found: {BEST_ONNX} -- using fast ONNX worker")
else:
    USE_ONNX = False
    log.info(f"No ONNX file found at {BEST_ONNX}")
    log.info(f"Selected model (.pt): {BEST_WEIGHT}")
    log.info("TIP: Run 'python export_onnx.py' once to create the ONNX file for instant startup!")

# ---------------------------------------------------------------------------
# Subprocess worker management
# ---------------------------------------------------------------------------
import subprocess
import threading
import queue
import pickle
import struct

worker_proc   = None
worker_ready  = False
worker_error  = None
req_queue     = queue.Queue()  # (request_bytes, response_queue)

def _worker_thread():
    """Manages the worker subprocess via stdin/stdout pipes."""
    global worker_proc, worker_ready, worker_error
    
    if USE_ONNX:
        script = str(WORKER_ONNX_PY)
        model_arg = BEST_ONNX
        log.info("Worker thread: starting ONNX worker (fast startup)...")
    else:
        script = str(WORKER_PY)
        model_arg = BEST_WEIGHT
        log.info("Worker thread: starting PyTorch worker (slow startup, may take minutes)...")

    try:
        worker_proc = subprocess.Popen(
            [sys.executable, "-u", script, model_arg],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        # Read stderr in background
        def drain_stderr():
            for line in iter(worker_proc.stderr.readline, b""):
                log.info(f"[worker] {line.decode(errors='replace').rstrip()}")
        threading.Thread(target=drain_stderr, daemon=True).start()

        # Wait for READY signal (first 5 bytes)
        ready_sig = worker_proc.stdout.read(5)
        if ready_sig == b"READY":
            worker_ready = True
            log.info("Worker subprocess READY -- YOLO model loaded and warmed up!")
        else:
            worker_error = f"Bad ready signal: {ready_sig}"
            log.error(f"Worker subprocess failed: {worker_error}")
            return

        # Serve requests
        while True:
            try:
                img_bytes, resp_q = req_queue.get()
                # Send length-prefixed request
                worker_proc.stdin.write(struct.pack(">I", len(img_bytes)))
                worker_proc.stdin.write(img_bytes)
                worker_proc.stdin.flush()
                # Read length-prefixed response
                raw_len = worker_proc.stdout.read(4)
                if not raw_len:
                    raise IOError("Worker closed stdout")
                resp_len = struct.unpack(">I", raw_len)[0]
                resp_bytes = worker_proc.stdout.read(resp_len)
                resp_q.put(pickle.loads(resp_bytes))
            except Exception as e:
                log.error(f"Worker comm error: {e}")
                resp_q.put({"error": str(e)})

    except Exception as e:
        worker_error = str(e)
        log.error(f"Worker thread crashed: {e}")
        log.error(traceback.format_exc())

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Perovskite Defect Detector", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def on_startup():
    log.info("FastAPI startup -- launching worker subprocess thread...")
    t = threading.Thread(target=_worker_thread, daemon=True)
    t.start()
    log.info("FastAPI ready. Model loading in worker subprocess (may take 1-3 min first time).")

# ---------------------------------------------------------------------------
# Class meta
# ---------------------------------------------------------------------------
CLASS_NAMES = {
    0: "PbI2 Excess",
    1: "3D Pinholes",
    2: "3D-2D Pinholes",
    3: "3D Background",
    4: "3D-2D Background",
}
CLASS_COLORS = {
    0: "#FF6B35", 1: "#00D4FF", 2: "#A855F7", 3: "#4ADE80", 4: "#FACC15",
}

# ---------------------------------------------------------------------------
# Inference endpoint  (sync def -- runs in thread pool)
# ---------------------------------------------------------------------------
@app.post("/api/detect")
def detect_defects(file: UploadFile = File(...), conf: float = 0.01):
    log.info(f"POST /api/detect -- file={file.filename} conf={conf}")

    if not worker_ready:
        if worker_error:
            raise HTTPException(503, f"Worker failed: {worker_error}")
        # Block here and wait — torch can take 3-10 min on Windows first run
        log.info("Request waiting for worker to become ready...")
        waited = 0
        while not worker_ready and not worker_error and waited < 600:
            import time as _time
            _time.sleep(2)
            waited += 2
        if worker_error:
            raise HTTPException(503, f"Worker failed: {worker_error}")
        if not worker_ready:
            raise HTTPException(503, "Model load timeout after 10 minutes. Please restart the server.")

    # Read & validate image
    contents = file.file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image file.")

    w, h = img.size
    # Size check removed to allow any dimensions

    # Send to worker subprocess (pickled payload with image bytes + conf threshold)
    resp_q = queue.Queue()
    payload = pickle.dumps({"image": contents, "conf": max(0.01, min(1.0, conf))})
    req_queue.put((payload, resp_q))
    try:
        result = resp_q.get(timeout=30)
    except queue.Empty:
        raise HTTPException(504, "Inference timed out after 30 seconds.")

    if "error" in result:
        raise HTTPException(500, f"Inference error: {result['error']}")

    # Draw bounding boxes
    annotated = img.copy()
    draw = ImageDraw.Draw(annotated)
    detections = []

    for box in result["boxes"]:
        cls_id = box["cls_id"]
        conf   = box["conf"]
        x1, y1, x2, y2 = box["xyxy"]
        label  = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
        color  = CLASS_COLORS.get(cls_id, "#FFFFFF")

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        text = f"{label} {conf:.0%}"
        bbox = draw.textbbox((x1, max(0, y1 - 14)), text)
        draw.rectangle([bbox[0]-1, bbox[1]-1, bbox[2]+1, bbox[3]+1], fill=color)
        draw.text((x1, max(0, y1 - 14)), text, fill="black")

        detections.append({
            "class_id": cls_id, "class_name": label,
            "confidence": round(conf, 4),
            "bbox": [round(x1,1), round(y1,1), round(x2,1), round(y2,1)],
            "color": color,
        })

    def img_to_b64(im):
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    log.info(f"  Returning {len(detections)} detections")

    response_data = {
        "success": len(detections) > 0,
        "num_detections": len(detections),
        "detections": detections,
        "original_b64": img_to_b64(img),
        "annotated_b64": img_to_b64(annotated),
    }

    # If zero defects, attach background classification from worker
    if len(detections) == 0 and "background_name" in result:
        response_data["is_background"]     = True
        response_data["background_class"]  = result["background_class"]
        response_data["background_name"]   = result["background_name"]
        response_data["bg_scores"]         = result.get("bg_scores", {})
        log.info(f"  Background image classified as: {result['background_name']}")

    return JSONResponse(response_data)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@app.get("/api/metrics")
def get_metrics():
    metrics = []
    if METRICS_DIR.exists():
        for jf in sorted(METRICS_DIR.glob("*.json")):
            with open(jf) as f:
                data = json.load(f)
                # Compute path to the run folder for frontend to construct image urls
                if "weights_path" in data:
                    try:
                        # Convert absolute to relative from 'runs' directory
                        wp = Path(data["weights_path"])
                        # 'src/runs/detect/yolo11m_.../weights/best.pt'
                        # parent is 'weights', parent.parent is 'yolo11m_...'
                        run_dir = wp.parent.parent.name
                        data["run_folder"] = f"detect/{run_dir}"
                    except Exception as e:
                        log.warning(f"Failed to infer run folder: {e}")
                metrics.append(data)
    return {"metrics": metrics}

# ---------------------------------------------------------------------------
# Manual Annotation Endpoints
# ---------------------------------------------------------------------------
from pydantic import BaseModel
from typing import List, Optional

@app.get("/api/dataset/folders")
def get_dataset_folders(root_dir: str):
    """List subfolders in the given root directory."""
    try:
        root_path = Path(root_dir)
        if not root_path.exists() or not root_path.is_dir():
            return {"error": "Invalid root directory path", "folders": []}
            
        folders = [f.name for f in root_path.iterdir() if f.is_dir()]
        return {"folders": sorted(folders)}
    except Exception as e:
        return {"error": str(e), "folders": []}

@app.get("/api/dataset/images")
def get_dataset_images(folder_path: str):
    """List all image files in a given folder absolute path."""
    try:
        p = Path(folder_path)
        if not p.exists() or not p.is_dir():
             return {"error": "Invalid folder path", "images": []}
             
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        images = [str(f.resolve()) for f in p.iterdir() if f.is_file() and f.suffix.lower() in valid_exts]
        return {"images": sorted(images)}
    except Exception as e:
         return {"error": str(e), "images": []}

@app.get("/api/dataset/image")
def get_dataset_image(image_path: str):
    """Serve a specific image from an absolute path."""
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
        
    if p.suffix.lower() in [".tif", ".tiff"]:
        try:
            img = Image.open(p).convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            buf.seek(0)
            return StreamingResponse(buf, media_type="image/jpeg")
        except Exception as e:
            log.error(f"Failed to convert TIFF: {e}")
            raise HTTPException(status_code=500, detail="Error converting TIFF to JPEG")
            
    return FileResponse(str(p))

@app.get("/api/dataset/labels")
def get_dataset_labels(image_path: str, class_folder: str, labels_root: str = ""):
    """Read YOLO formatted .txt labels for a specific image.
    labels_root: absolute path to the labels directory root.
                 If empty, falls back to PROJECT_ROOT/labels.
    """
    try:
        img_name = Path(image_path).stem
        root = Path(labels_root) if labels_root else PROJECT_ROOT / "labels"
        label_file = root / class_folder / f"{img_name}.txt"

        if not label_file.exists():
            return {"boxes": [], "label_file": str(label_file), "found": False}

        boxes = []
        with open(label_file, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append({
                        "class_id": int(parts[0]),
                        "x_center": float(parts[1]),
                        "y_center": float(parts[2]),
                        "width":    float(parts[3]),
                        "height":   float(parts[4])
                    })
        return {"boxes": boxes, "label_file": str(label_file), "found": True}
    except Exception as e:
        return {"error": str(e), "boxes": []}

class BoxDef(BaseModel):
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

class AnnotationSaveReq(BaseModel):
    image_path: str
    class_folder: str
    labels_root: str = ""
    boxes: List[BoxDef]

@app.post("/api/dataset/labels")
def save_dataset_labels(req: AnnotationSaveReq):
    """Save YOLO formatted .txt labels for an image."""
    try:
        img_name = Path(req.image_path).stem
        root = Path(req.labels_root) if req.labels_root else PROJECT_ROOT / "labels"
        labels_dir = root / req.class_folder
        labels_dir.mkdir(parents=True, exist_ok=True)
        label_file = labels_dir / f"{img_name}.txt"

        yolo_lines = [
            f"{b.class_id} {b.x_center:.6f} {b.y_center:.6f} {b.width:.6f} {b.height:.6f}"
            for b in req.boxes
        ]
        with open(label_file, "w") as f:
            f.write("\n".join(yolo_lines))

        log.info(f"Saved {len(req.boxes)} boxes → {label_file}")
        return {"success": True, "label_file": str(label_file)}
    except Exception as e:
        log.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

class OpenCVReq(BaseModel):
    image_path: str
    class_folder: str
    labels_root: str = ""
    class_id: int
    method: str
    threshold1: int
    threshold2: int
    brightness: int
    contrast: float
    min_area: int
    max_area: int
    use_clahe: bool
    overwrite: bool = True
    clahe_clip: float
    clahe_grid: int

@app.post("/api/opencv_detect")
def opencv_detect(req: OpenCVReq):
    """Run OpenCV auto-detection, save annotations, and return updated boxes."""
    try:
        sys.path.append(str(PROJECT_ROOT))
        from src.handlers.opencv_handler import auto_annotate_with_opencv

        root = Path(req.labels_root) if req.labels_root else PROJECT_ROOT / "labels"
        labels_dir = root / req.class_folder
        labels_dir.mkdir(parents=True, exist_ok=True)

        count = auto_annotate_with_opencv(
            image_list=[req.image_path],
            labels_dir=str(labels_dir),
            class_id=req.class_id,
            method=req.method,
            threshold1=req.threshold1,
            threshold2=req.threshold2,
            brightness=req.brightness,
            contrast=req.contrast,
            min_area=req.min_area,
            max_area=req.max_area,
            use_clahe=req.use_clahe,
            clahe_clip=req.clahe_clip,
            clahe_grid=req.clahe_grid,
            overwrite=req.overwrite
        )

        # Return the boxes from the labels we just saved
        return get_dataset_labels(req.image_path, req.class_folder, str(labels_dir.parent))

    except Exception as e:
        log.error(f"OpenCV Error: {traceback.format_exc()}")
        return {"error": str(e), "boxes": []}

# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": worker_ready,
        "model_loading": not worker_ready and worker_error is None,
        "error": worker_error,
    }

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

runs_dir = PROJECT_ROOT / "src" / "runs"
if runs_dir.exists():
    app.mount("/runs", StaticFiles(directory=str(runs_dir)), name="runs")

@app.get("/")
def serve_index():
    idx = FRONTEND_DIR / "index.html"
    return FileResponse(str(idx)) if idx.exists() else JSONResponse({"error": "index.html not found"}, status_code=404)
