"""
worker.py -- YOLO inference subprocess (PyTorch / ultralytics)
===============================================================
Launched by main.py as a child process.
Handles per-request confidence threshold and background classification.

Protocol (stdin/stdout binary, length-prefixed):
  - Sends "READY" (5 bytes) when model is loaded.
  - Reads: 4-byte big-endian uint32 length, then <length> bytes of
           pickle({image: bytes, conf: float}).
  - Writes: 4-byte big-endian uint32 length, then pickle of result dict.
"""

import sys
import io
import struct
import pickle
import traceback


DEFAULT_CONF = 0.05


def load_and_serve(weights_path: str):
    print(f"[worker] Loading YOLO from: {weights_path}", file=sys.stderr, flush=True)

    # ----------------------------------------------------------------
    # Load torch & ultralytics — this can take 3-10 min on first run
    # due to Windows Defender scanning DLLs. We just wait it out.
    # ----------------------------------------------------------------
    print("[worker] Importing torch (may take several minutes on first run)...", file=sys.stderr, flush=True)
    import torch
    print(f"[worker] torch {torch.__version__}, CUDA={torch.cuda.is_available()}", file=sys.stderr, flush=True)

    print("[worker] Importing ultralytics YOLO...", file=sys.stderr, flush=True)
    from ultralytics import YOLO
    import numpy as np
    from PIL import Image

    print("[worker] Loading model weights...", file=sys.stderr, flush=True)
    model = YOLO(weights_path)

    # Warm-up pass
    print("[worker] Warming up model (224x224 dummy)...", file=sys.stderr, flush=True)
    dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    model(dummy, imgsz=224, conf=0.25, verbose=False)

    print("[worker] Model ready!", file=sys.stderr, flush=True)

    # Signal ready to parent
    sys.stdout.buffer.write(b"READY")
    sys.stdout.buffer.flush()

    # Serve inference requests
    while True:
        try:
            # Read request length
            raw_len = sys.stdin.buffer.read(4)
            if not raw_len or len(raw_len) < 4:
                print("[worker] stdin closed, exiting.", file=sys.stderr, flush=True)
                break

            req_len = struct.unpack(">I", raw_len)[0]
            img_bytes = sys.stdin.buffer.read(req_len)

            # Unpack request (supports conf threshold)
            req_data    = pickle.loads(img_bytes)
            raw_image   = req_data["image"]
            conf_thresh = float(req_data.get("conf", DEFAULT_CONF))

            # Run inference — ultralytics handles letterboxing + NMS
            img = Image.open(io.BytesIO(raw_image)).convert("RGB")
            W, H = img.size
            print(f"[worker] Inference on {W}x{H} image (conf={conf_thresh})...",
                  file=sys.stderr, flush=True)

            results = model(img, imgsz=224, conf=conf_thresh, verbose=False)
            boxes_raw = results[0].boxes

            boxes = []
            for box in boxes_raw:
                boxes.append({
                    "cls_id": int(box.cls[0].item()),
                    "conf":   float(box.conf[0].item()),
                    "xyxy":   [float(c) for c in box.xyxy[0].tolist()],
                })

            result = {"boxes": boxes}

            # Background classification when zero defects detected
            if len(boxes) == 0:
                # Use raw class probabilities from the model output
                # Run a second pass at very low conf to get any signal
                results_bg = model(img, imgsz=224, conf=0.001, verbose=False)
                bg_boxes = results_bg[0].boxes
                if len(bg_boxes) > 0:
                    # Count class votes from low-conf predictions
                    class_votes = {}
                    for box in bg_boxes:
                        cid = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        class_votes[cid] = class_votes.get(cid, 0) + conf

                    # Check background classes (3=3D_bg, 4=3D-2D_bg)
                    bg3 = class_votes.get(3, 0)
                    bg4 = class_votes.get(4, 0)
                    if bg3 > 0 or bg4 > 0:
                        if bg3 >= bg4:
                            result["background_class"] = 3
                            result["background_name"]  = "3D_background"
                        else:
                            result["background_class"] = 4
                            result["background_name"]  = "3D-2D_background"
                        result["bg_scores"] = {
                            "3D_background":    round(bg3, 5),
                            "3D-2D_background": round(bg4, 5),
                        }

            print(f"[worker] Done: {len(boxes)} detections",
                  file=sys.stderr, flush=True)

        except Exception as e:
            print(f"[worker] Inference error: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            result = {"error": str(e)}

        # Write response
        resp_bytes = pickle.dumps(result)
        sys.stdout.buffer.write(struct.pack(">I", len(resp_bytes)))
        sys.stdout.buffer.write(resp_bytes)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[worker] ERROR: no weights path given", file=sys.stderr, flush=True)
        sys.exit(1)
    load_and_serve(sys.argv[1])
