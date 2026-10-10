"""Preprocess raw petri dish images and root masks into 256x256 patches.

Bridges Ahmed's raw data assets (cv4-raw-train, cv4-raw-val) to the patched
four-folder structure that train.py consumes. Refactored from Task_4.ipynb
with the year-specific pairing logic stripped out since the data asset
already merges years and applies the train/val split.

For each image, the pipeline runs three stages:
    1. Pair the image with its mask by filename stem.
    2. Detect the petri dish bounding box and crop the image and mask to the
       inner plate, producing a square crop where the dish fills the frame.
    3. Pad the cropped square to a multiple of the patch size and extract
       non-overlapping tiles, writing them as paired .png files.

Designed to be invoked either directly from the command line or by an Azure
ML pipeline component, which calls the same argparse interface.
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PATCH_SIZE = 256
DEFAULT_MASK_SUFFIX = "_root_mask.tif"


TRAIN_IMAGES_SUBDIR = ("train_images", "train")
TRAIN_MASKS_SUBDIR = ("train_masks", "train")
VAL_IMAGES_SUBDIR = ("val_images", "val")
VAL_MASKS_SUBDIR = ("val_masks", "val")


def detect_petri_bbox(gray: np.ndarray) -> tuple[int, int, int]:
    """Locate the petri dish as a square bbox via adaptive thresholding.

    Returns (x, y, side) where side is the square's edge length. Falls back to
    a centered square covering the smaller image dimension if no contour is
    found.
    """
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -10)
    if np.mean(thr) < 127:
        thr = 255 - thr

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape

    if not cnts:
        side = min(h, w)
        return ((w - side) // 2, (h - side) // 2, side)

    cx, cy = w / 2.0, h / 2.0

    def score(c: np.ndarray) -> float:
        # Largest contour wins, with a small penalty for distance from image
        # center so background blobs don't beat the actual dish.
        x, y, ww, hh = cv2.boundingRect(c)
        area = ww * hh
        mx, my = x + ww / 2.0, y + hh / 2.0
        dist = ((mx - cx) ** 2 + (my - cy) ** 2) ** 0.5
        return area - 0.1 * dist

    best = max(cnts, key=score)
    x, y, ww, hh = cv2.boundingRect(best)

    side = max(ww, hh)
    side = min(side, h, w)
    mx, my = x + ww // 2, y + hh // 2
    xs = max(0, min(int(mx - side / 2), w - side))
    ys = max(0, min(int(my - side / 2), h - side))

    return (xs, ys, side)


def refine_to_inner_plate(gray: np.ndarray, bbox: tuple[int, int, int], pad: int = 20) -> tuple[int, int, int]:
    """Shrink the bbox to the inner plate by trimming dark borders.

    The initial detection finds the dish silhouette including its rim; the
    inner plate (where roots actually grow) is brighter and surrounded by a
    darker ring. Project pixel intensity onto each axis, Otsu-threshold the
    projections, and take the longest contiguous bright run as the inner
    bounds.
    """
    x, y, s = bbox
    crop = gray[y : y + s, x : x + s]

    rmean = crop.mean(axis=1)
    cmean = crop.mean(axis=0)

    _, rbin = cv2.threshold(rmean.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, cbin = cv2.threshold(cmean.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    rbin = rbin > 0
    cbin = cbin > 0

    def longest_run(b: np.ndarray) -> tuple[int, int]:
        best = (0, -1)
        i = 0
        while i < len(b):
            if b[i]:
                j = i
                while j < len(b) and b[j]:
                    j += 1
                if (j - 1) - i > best[1] - best[0]:
                    best = (i, j - 1)
                i = j
            else:
                i += 1
        return best

    r0, r1 = longest_run(rbin)
    c0, c1 = longest_run(cbin)

    r0 = max(0, r0 - pad)
    r1 = min(s - 1, r1 + pad)
    c0 = max(0, c0 - pad)
    c1 = min(s - 1, c1 + pad)

    hh = r1 - r0 + 1
    ww = c1 - c0 + 1

    side = min(hh, ww)
    cy = (r0 + r1) // 2
    cx = (c0 + c1) // 2

    r0 = max(0, cy - side // 2)
    c0 = max(0, cx - side // 2)

    return (x + c0, y + r0, side)


def crop_square(arr: np.ndarray, bbox: tuple[int, int, int]) -> np.ndarray:
    """Slice a square region from an image given (x, y, side)."""
    x, y, s = bbox
    return arr[y : y + s, x : x + s]


def normalize_mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    """Convert a mask of any dtype/encoding to uint8 with roots at 255.

    Block B masks vary in both dtype (uint8, uint16, float) and value
    encoding: some store roots as 255, others as 1 (binary label maps). Both
    are valid mask conventions, but the training generator binarizes mask
    PNGs against a mid-range threshold, so roots must be written at 255, not
    1, or every root pixel reads as background and the model trains on blank
    labels.

    Rather than scaling by the max (which is fragile and was previously
    short-circuited for uint8 inputs, leaving 0/1 masks unscaled), this
    treats any nonzero pixel as foreground and maps it to 255. This is
    correct for binary segmentation masks regardless of their original
    encoding.
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def pad_to_multiple(arr: np.ndarray, patch_size: int) -> np.ndarray:
    """Zero-pad to the next multiple of patch_size in each dimension."""
    h, w = arr.shape
    new_h = ((h - 1) // patch_size + 1) * patch_size
    new_w = ((w - 1) // patch_size + 1) * patch_size
    padded = np.zeros((new_h, new_w), dtype=arr.dtype)
    padded[:h, :w] = arr
    return padded


def patch_pair(
    img: np.ndarray,
    mask: np.ndarray,
    stem: str,
    out_img_dir: Path,
    out_mask_dir: Path,
    patch_size: int,
) -> int:
    """Extract non-overlapping patches from a paired image and mask.

    Both arrays are padded to a multiple of patch_size, then tiled with
    stride=patch_size. Each output patch is named ``{stem}_y{y:04d}_x{x:04d}.png``
    so paired image and mask patches share a filename across the two output
    directories.
    """
    assert img.shape == mask.shape, f"shape mismatch for {stem}: {img.shape} vs {mask.shape}"

    img_p = pad_to_multiple(img, patch_size)
    mask_p = pad_to_multiple(mask, patch_size)

    H, W = img_p.shape
    count = 0

    for y in range(0, H, patch_size):
        for x in range(0, W, patch_size):
            img_patch = img_p[y : y + patch_size, x : x + patch_size]
            mask_patch = mask_p[y : y + patch_size, x : x + patch_size]
            if img_patch.shape != (patch_size, patch_size):
                continue
            if mask_patch.shape != (patch_size, patch_size):
                continue
            patch_name = f"{stem}_y{y:04d}_x{x:04d}.png"
            cv2.imwrite(str(out_img_dir / patch_name), img_patch)
            cv2.imwrite(str(out_mask_dir / patch_name), mask_patch)
            count += 1

    return count


def find_image_mask_pairs(
    images_dir: Path,
    masks_dir: Path,
    mask_suffix: str,
) -> tuple[list[tuple[Path, Path]], int]:
    """Pair each image with its mask by filename stem, across two directories.

    For an image ``foo.png`` in images_dir, looks for ``foo{mask_suffix}`` in
    masks_dir. Ahmed's data asset keeps images and masks in separate
    subfolders (``train_images/train/`` and ``train_masks/train/``) to mirror
    the structure produced by patching, which matches what
    ``flow_from_directory`` consumes downstream. Returns
    (pairs, missing_count) so callers can log how many images had no matching
    mask, the way Task_4.ipynb's per-year processing did.
    """
    image_paths = sorted(images_dir.glob("*.png"))
    pairs: list[tuple[Path, Path]] = []
    missing = 0

    for img_path in image_paths:
        stem = img_path.stem
        mask_path = masks_dir / f"{stem}{mask_suffix}"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            missing += 1

    return pairs, missing


def process_split(
    images_dir: Path,
    masks_dir: Path,
    out_img_dir: Path,
    out_mask_dir: Path,
    patch_size: int,
    mask_suffix: str,
    label: str,
) -> tuple[int, int]:
    """Crop, patch, and save every paired image+mask in one split.

    Returns (image_count, patch_count) for logging by the caller.
    """
    pairs, missing = find_image_mask_pairs(images_dir, masks_dir, mask_suffix)
    logger.info(
        "%s split: %d image+mask pairs (%d images had no matching mask)",
        label,
        len(pairs),
        missing,
    )

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_mask_dir.mkdir(parents=True, exist_ok=True)

    total_patches = 0
    for img_path, mask_path in pairs:
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if img is None or mask_raw is None:
            logger.warning("Skipping unreadable pair: %s / %s", img_path, mask_path)
            continue

        bbox = detect_petri_bbox(img)
        bbox = refine_to_inner_plate(img, bbox, pad=20)
        img_crop = crop_square(img, bbox)
        mask_crop = crop_square(normalize_mask_to_uint8(mask_raw), bbox)

        n = patch_pair(
            img_crop,
            mask_crop,
            img_path.stem,
            out_img_dir,
            out_mask_dir,
            patch_size,
        )
        total_patches += n

    logger.info("%s split: wrote %d patches", label, total_patches)
    return len(pairs), total_patches


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Argparse rather than Typer for the same reason train.py uses argparse:
    Azure ML pipeline components launch via a plain shell command, and
    argparse maps to that contract most directly.
    """
    parser = argparse.ArgumentParser(description="Crop petri dishes and patch raw images+masks for cv4 training.")
    parser.add_argument(
        "--train-data-path",
        required=True,
        help=("Path to the raw train data asset. Must contain train_images/train/ and train_masks/train/ subdirectories."),
    )
    parser.add_argument(
        "--val-data-path",
        required=True,
        help=("Path to the raw val data asset. Must contain val_images/val/ and val_masks/val/ subdirectories."),
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Destination folder for the patched four-folder output.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=DEFAULT_PATCH_SIZE,
        help="Edge length of square patches in pixels (default: 256).",
    )
    parser.add_argument(
        "--mask-suffix",
        type=str,
        default=DEFAULT_MASK_SUFFIX,
        help=("Filename suffix appended to an image's stem to find its mask (default: '_root_mask.tif')."),
    )
    return parser.parse_args()


def _log_directory_contents(label: str, root: Path) -> None:
    """Log every file under root for diagnostic purposes.

    Useful when running inside an Azure ML job and a mounted data asset
    appears empty: this surfaces the actual structure so we know whether the
    asset is empty, has a different layout than expected, or is mounted at a
    different path.
    """
    logger.info("Listing contents of %s root: %s", label, root)
    if not root.exists():
        logger.info("  (root does not exist)")
        return
    entries = list(root.rglob("*"))
    if not entries:
        logger.info("  (root is empty)")
        return
    for p in entries[:50]:
        kind = "DIR " if p.is_dir() else "FILE"
        logger.info("  %s %s", kind, p.relative_to(root))
    if len(entries) > 50:
        logger.info("  ... and %d more entries", len(entries) - 50)


def main() -> None:
    """Run preprocessing on both train and val splits."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    args = parse_args()

    train_root = Path(args.train_data_path)
    val_root = Path(args.val_data_path)
    out_root = Path(args.output_path)

    if not train_root.exists():
        raise FileNotFoundError(f"--train-data-path does not exist: {train_root}")
    if not val_root.exists():
        raise FileNotFoundError(f"--val-data-path does not exist: {val_root}")

    train_images_dir = train_root / "train_images" / "train"
    train_masks_dir = train_root / "train_masks" / "train"
    val_images_dir = val_root / "val_images" / "val"
    val_masks_dir = val_root / "val_masks" / "val"

    for d, name in [
        (train_images_dir, "train_images/train"),
        (train_masks_dir, "train_masks/train"),
        (val_images_dir, "val_images/val"),
        (val_masks_dir, "val_masks/val"),
    ]:
        if not d.exists():
            raise FileNotFoundError(f"Expected subfolder '{name}' not found inside data asset at {d}")

    train_out_img = out_root / TRAIN_IMAGES_SUBDIR[0] / TRAIN_IMAGES_SUBDIR[1]
    train_out_mask = out_root / TRAIN_MASKS_SUBDIR[0] / TRAIN_MASKS_SUBDIR[1]
    val_out_img = out_root / VAL_IMAGES_SUBDIR[0] / VAL_IMAGES_SUBDIR[1]
    val_out_mask = out_root / VAL_MASKS_SUBDIR[0] / VAL_MASKS_SUBDIR[1]

    logger.info("Patch size: %dx%d", args.patch_size, args.patch_size)
    logger.info("Mask suffix: %s", args.mask_suffix)
    logger.info("Output root: %s", out_root)

    _log_directory_contents("TRAIN", train_root)
    _log_directory_contents("VAL", val_root)

    train_pairs, train_patches = process_split(
        train_images_dir,
        train_masks_dir,
        train_out_img,
        train_out_mask,
        args.patch_size,
        args.mask_suffix,
        label="TRAIN",
    )
    val_pairs, val_patches = process_split(
        val_images_dir,
        val_masks_dir,
        val_out_img,
        val_out_mask,
        args.patch_size,
        args.mask_suffix,
        label="VAL",
    )

    logger.info(
        "Done. Train: %d pairs -> %d patches; Val: %d pairs -> %d patches.",
        train_pairs,
        train_patches,
        val_pairs,
        val_patches,
    )


if __name__ == "__main__":
    main()
