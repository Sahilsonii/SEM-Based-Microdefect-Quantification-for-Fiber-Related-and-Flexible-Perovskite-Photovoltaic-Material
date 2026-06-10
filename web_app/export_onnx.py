"""
Re-export YOLO11m to ONNX at imgsz=1024 for full-resolution inference.
Run once:  python export_onnx.py
"""
import sys, os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
all_weights = sorted(PROJECT_ROOT.glob("src/runs/detect/*/weights/best.pt"))
best_pt = None
for w in all_weights:
    if "yolo11m" in str(w):
        best_pt = str(w)
        break
if not best_pt and all_weights:
    best_pt = str(all_weights[0])

if not best_pt:
    print("ERROR: No best.pt found!")
    sys.exit(1)

print(f"Exporting: {best_pt}")

from ultralytics import YOLO
model = YOLO(best_pt)

# Export at imgsz=1024 so full-res SEM images (1024x768) are processed
# without downsampling that destroys small PbI2 features.
# dynamic=True allows any input size.
onnx_path = model.export(
    format="onnx",
    imgsz=1024,
    dynamic=True,
    simplify=True,
    opset=17,
)

print(f"\nONNX exported to: {onnx_path}")
print("Done! Restart the web app server to use the new model.")
