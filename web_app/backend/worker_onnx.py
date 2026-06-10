"""
worker_onnx.py -- ONNX Runtime inference subprocess
=====================================================
Uses onnxruntime for fast startup (~1s vs 10+ min for torch CUDA).

Sliding-window tile inference at native 224×224 resolution.
Uses a very low internal threshold to capture weak signals (PbI2),
then lets the user's confidence slider filter results.

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
import numpy as np
from PIL import Image

TILE         = 224   # model input size
STRIDE       = 112   # 50% overlap between tiles
IOU_THRESH   = 0.45  # NMS IoU threshold
DEFAULT_CONF = 0.01  # very low internal threshold — UI slider filters above this


# ── Preprocessing ────────────────────────────────────────────────────
def preprocess_tile(tile: Image.Image):
    """Convert a PIL tile to a float32 NCHW batch tensor."""
    arr = np.array(tile.resize((TILE, TILE)), dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]   # [1, 3, 224, 224]


# ── Decode YOLO output ──────────────────────────────────────────────
def decode_tile(output, tile_x, tile_y, tile_w, tile_h, conf_thresh):
    """Decode YOLO ONNX output for one tile.
    output shape: [1, 9, 1029]  =>  pred [9, 1029]
    Rows 0-3 = cx, cy, w, h in 224-pixel space.
    Rows 4+  = class scores.
    """
    pred = output[0][0]              # [9, n_anchors]
    cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
    class_scores = pred[4:]          # [num_classes, n_anchors]
    confs = class_scores.max(axis=0)
    cls_ids = class_scores.argmax(axis=0)

    # Mean score per class (for background classification)
    mean_class_scores = class_scores.mean(axis=1).tolist()

    # Scale from 224-space to tile pixel space
    sx = tile_w / TILE
    sy = tile_h / TILE

    boxes = []
    for i in range(pred.shape[1]):
        conf = float(confs[i])
        if conf < conf_thresh:
            continue
        cls_id = int(cls_ids[i])

        # xyxy in tile pixel space
        x1 = float((cx[i] - w[i] / 2) * sx)
        y1 = float((cy[i] - h[i] / 2) * sy)
        x2 = float((cx[i] + w[i] / 2) * sx)
        y2 = float((cy[i] + h[i] / 2) * sy)

        # Map to original image coordinates
        x1 = float(max(0.0, x1 + tile_x))
        y1 = float(max(0.0, y1 + tile_y))
        x2 = float(x2 + tile_x)
        y2 = float(y2 + tile_y)

        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"cls_id": cls_id, "conf": conf, "xyxy": [x1, y1, x2, y2]})

    return boxes, mean_class_scores


# ── Class-aware NMS ──────────────────────────────────────────────────
def nms(boxes):
    """Class-aware NMS: only suppress boxes of the same class."""
    boxes = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best)
        def iou(a, b):
            if a["cls_id"] != b["cls_id"]:
                return 0.0
            ax1, ay1, ax2, ay2 = a["xyxy"]
            bx1, by1, bx2, by2 = b["xyxy"]
            ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
            return inter / union if union > 0 else 0.0
        boxes = [b for b in boxes if iou(best, b) < IOU_THRESH]
    return keep


# ── Main inference function ──────────────────────────────────────────
def run_tile_inference(sess, input_name, img, conf_thresh):
    """Slide 224×224 window across image, collect and NMS all detections."""
    W, H = img.size
    all_boxes = []
    all_tile_scores = []

    # If image fits in one tile, pad and run once
    if W <= TILE and H <= TILE:
        tile = Image.new("RGB", (TILE, TILE), (114, 114, 114))
        tile.paste(img, (0, 0))
        inp = preprocess_tile(tile)
        out = sess.run(None, {input_name: inp})
        boxes, scores = decode_tile(out, 0, 0, W, H, conf_thresh)
        return nms(boxes), scores

    # Build list of (x, y) tile origins
    ys = list(range(0, H - TILE + 1, STRIDE))
    if not ys or ys[-1] + TILE < H:
        ys.append(max(0, H - TILE))
    xs = list(range(0, W - TILE + 1, STRIDE))
    if not xs or xs[-1] + TILE < W:
        xs.append(max(0, W - TILE))

    for y in ys:
        for x in xs:
            tile_w = min(TILE, W - x)
            tile_h = min(TILE, H - y)
            crop = img.crop((x, y, x + tile_w, y + tile_h))

            if tile_w < TILE or tile_h < TILE:
                padded = Image.new("RGB", (TILE, TILE), (114, 114, 114))
                padded.paste(crop, (0, 0))
                tile_img = padded
            else:
                tile_img = crop

            inp = preprocess_tile(tile_img)
            out = sess.run(None, {input_name: inp})
            boxes, scores = decode_tile(out, x, y, tile_w, tile_h, conf_thresh)
            all_boxes.extend(boxes)
            all_tile_scores.append(scores)

    # Average class scores across all tiles
    if all_tile_scores:
        arr = np.array(all_tile_scores)
        mean_scores = arr.mean(axis=0).tolist()
    else:
        mean_scores = []

    return nms(all_boxes), mean_scores


# ── Subprocess entry point ───────────────────────────────────────────
def load_and_serve(onnx_path: str):
    print(f"[worker] Loading ONNX model: {onnx_path}", file=sys.stderr, flush=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    print(f"[worker] ONNX model loaded! Input: {input_name}", file=sys.stderr, flush=True)

    # Signal ready to parent
    sys.stdout.buffer.write(b"READY")
    sys.stdout.buffer.flush()

    while True:
        try:
            raw_len = sys.stdin.buffer.read(4)
            if not raw_len or len(raw_len) < 4:
                print("[worker] stdin closed, exiting.", file=sys.stderr, flush=True)
                break

            req_len = struct.unpack(">I", raw_len)[0]
            img_bytes = sys.stdin.buffer.read(req_len)

            req_data    = pickle.loads(img_bytes)
            raw_image   = req_data["image"]
            conf_thresh = float(req_data.get("conf", DEFAULT_CONF))

            img = Image.open(io.BytesIO(raw_image)).convert("RGB")
            W, H = img.size
            print(f"[worker] Tiled inference on {W}x{H} image "
                  f"(stride={STRIDE}, conf={conf_thresh})...",
                  file=sys.stderr, flush=True)

            boxes, mean_scores = run_tile_inference(
                sess, input_name, img, conf_thresh
            )
            result = {"boxes": boxes}

            # Background classification when zero defects detected
            if len(boxes) == 0 and len(mean_scores) >= 5:
                bg3_score = mean_scores[3]   # 3D_background
                bg4_score = mean_scores[4]   # 3D-2D_background
                if bg3_score >= bg4_score:
                    result["background_class"] = 3
                    result["background_name"]  = "3D_background"
                else:
                    result["background_class"] = 4
                    result["background_name"]  = "3D-2D_background"
                result["bg_scores"] = {
                    "3D_background":    round(bg3_score, 5),
                    "3D-2D_background": round(bg4_score, 5),
                }

            print(f"[worker] Done: {len(boxes)} detections after global NMS",
                  file=sys.stderr, flush=True)

        except Exception as e:
            print(f"[worker] Inference error: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            result = {"error": str(e)}

        resp_bytes = pickle.dumps(result)
        sys.stdout.buffer.write(struct.pack(">I", len(resp_bytes)))
        sys.stdout.buffer.write(resp_bytes)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("[worker] ERROR: no ONNX path given", file=sys.stderr, flush=True)
        sys.exit(1)
    load_and_serve(sys.argv[1])
