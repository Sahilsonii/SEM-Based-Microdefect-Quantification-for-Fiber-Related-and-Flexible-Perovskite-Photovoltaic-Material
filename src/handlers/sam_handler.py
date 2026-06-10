import os
import sys
import numpy as np
import cv2
import torch

# Add SAM path
sam_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sam_pinhole_annotation")
if sam_path not in sys.path:
    sys.path.append(sam_path)

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


def _load_image_rgb(img_path):
    """Load image as RGB numpy array. Falls back to PIL for .tif files."""
    img = cv2.imread(img_path)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image as _PIL
        return np.array(_PIL.open(img_path).convert("RGB"))
    except Exception:
        return None


def _build_mask_generator(sam, conf_threshold, points_per_side):
    """Build SamAutomaticMaskGenerator, gracefully handling older SAM versions."""
    gen_kwargs = dict(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=conf_threshold,
        stability_score_thresh=conf_threshold,
    )
    try:
        return SamAutomaticMaskGenerator(min_mask_region_area=0, **gen_kwargs)
    except TypeError:
        return SamAutomaticMaskGenerator(**gen_kwargs)


def _load_sam_model(conf_threshold, points_per_side):
    """Load SAM model and return (mask_generator, error_string)."""
    if not SAM_AVAILABLE:
        return None, (
            "segment-anything is not installed.\n"
            "Run:  pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    sam_checkpoint = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "sam_pinhole_annotation", "sam", "sam_vit_b_01ec64.pth"
    )
    if not os.path.exists(sam_checkpoint):
        return None, f"SAM checkpoint not found: {sam_checkpoint}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    return _build_mask_generator(sam, conf_threshold, points_per_side), None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def generate_sam_preview(img_path, conf_threshold=0.35, points_per_side=64,
                         min_area=10, max_area=None, alpha=0.45):
    """
    Run SAM on a single image and return a colourful mask-overlay as an
    RGB numpy array (same size as the input image).

    Returns:
        overlay (np.ndarray, uint8, H×W×3)  – image with coloured masks
        n_masks (int)                        – total masks before area filter
        n_kept  (int)                        – masks kept after area filter
    """
    mask_generator, err = _load_sam_model(conf_threshold, points_per_side)
    if err:
        raise RuntimeError(err)

    img_rgb = _load_image_rgb(img_path)
    if img_rgb is None:
        raise IOError(f"Cannot read image: {img_path}")

    h, w = img_rgb.shape[:2]
    masks = mask_generator.generate(img_rgb)

    overlay = img_rgb.copy().astype(np.float32)
    rng = np.random.default_rng(42)
    kept = 0

    for mask_data in sorted(masks, key=lambda m: m["area"], reverse=True):
        seg  = mask_data["segmentation"]
        area = int(seg.sum())

        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue
        kept += 1

        colour = rng.integers(80, 256, size=3).astype(np.float32)   # bright random colour
        for c in range(3):
            overlay[:, :, c] = np.where(
                seg,
                overlay[:, :, c] * (1 - alpha) + colour[c] * alpha,
                overlay[:, :, c]
            )

        # Draw a thin white contour around each mask for clarity
        mask_u8  = seg.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    return overlay.clip(0, 255).astype(np.uint8), len(masks), kept


def auto_annotate_with_sam(images, output_dir, conf_threshold=0.35,
                           points_per_side=64, min_area=10, max_area=None,
                           overwrite=True):
    """
    Auto-annotate images using SAM with automatic mask generation.

    Returns:
        labeled_count (int) – number of images that got ≥1 detection saved
    """
    mask_generator, err = _load_sam_model(conf_threshold, points_per_side)
    if err:
        raise RuntimeError(err)

    labeled_count = 0

    for img_path in images:
        base_name  = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(output_dir, base_name + ".txt")

        # Skip only if overwrite is disabled AND a non-empty label file exists
        if not overwrite and os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            continue

        img_rgb = _load_image_rgb(img_path)
        if img_rgb is None:
            continue

        h, w = img_rgb.shape[:2]
        masks = mask_generator.generate(img_rgb)

        yolo_lines = []
        for mask_data in masks:
            seg       = mask_data["segmentation"]
            mask_area = int(seg.sum())

            if mask_area < min_area:
                continue
            if max_area is not None and mask_area > max_area:
                continue

            coords = np.argwhere(seg)
            if len(coords) == 0:
                continue

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            x_center = ((x_min + x_max) / 2) / w
            y_center = ((y_min + y_max) / 2) / h
            box_w    = (x_max - x_min) / w
            box_h    = (y_max - y_min) / h

            yolo_lines.append(
                f"1 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            )

        os.makedirs(output_dir, exist_ok=True)
        if yolo_lines:
            with open(label_path, "w") as f:
                f.write("\n".join(yolo_lines))
            labeled_count += 1
        else:
            # Write empty file so this image is marked as processed
            open(label_path, "w").close()

    return labeled_count
