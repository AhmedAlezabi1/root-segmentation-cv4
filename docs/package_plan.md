# Package plan

Module structure, interfaces, and dependencies for the cv4 package. Based on pipeline 245769 (see docs/pipeline_selection.md for the selection decision). The original notebook is in reference/Task_8.ipynb.

## Task 1.2.3: Module Mapping

The pipeline has 15 notebook cells. Below is how each piece of code maps to a Python module. Visualization-only cells (2, 5, 6, 9, 12) are not included in the core package — they become optional utility functions or frontend logic.

### Module: config.py
**Purpose:** Central place for all default parameters. No logic, just values.

```
Source: Scattered across cells 0, 3, 7, 10, 11, 14

Parameters to extract:
- PATCH_SIZE = 256
- STRIDE = 128
- DEFAULT_MODEL_PATH = "models/unet_256px.h5"
- DEFAULT_LANE_EDGES_PATH = "models/lane_edges.npy"
- HYSTERESIS_LOW = 0.05
- HYSTERESIS_HIGH = 0.25
- PLATE_SIZE_MM = 150.0
- EXPECTED_ROOTS = 5
- MICRO_CLOSE_KERNEL_SIZE = 4
- MICRO_CLOSE_ITERATIONS = 1
- BRIDGE_MAX_DIST = 60
- BRIDGE_MAX_DX = 4
- BRIDGE_MIN_DY = 8
- BRIDGE_THICKNESS = 3
- SPECK_AREA = 3
- EXCLUDE_TOP_FRAC = 0.15
- TOP_BAND_FRAC = 0.30
- MIN_AREA_CANDIDATE = 10
- MIN_HEIGHT_PX = 20
- MIN_LANE_SCORE = 120.0
- MERGE_DIST_PX = 22
- (All the scoring weights and pathfinding parameters from cells 10 and 11)
```

---

### Module: preprocessing.py
**Purpose:** Everything that happens to the raw image before the model sees it.

```
Source: Cell 0 (partial), Cell 1

Functions:
─────────────────────────────────────────────────────────
load_image(path: str) → np.ndarray
  - Loads a grayscale image from disk
  - Source: cv2.imread(p, cv2.IMREAD_GRAYSCALE) scattered throughout cells
  - Add: validation (file exists, is valid image, is grayscale)

detect_petri_bbox(gray: np.ndarray) → tuple[int, int, int]
  - Finds the petri dish bounding box (x, y, side)
  - Source: Cell 1, lines 2–52
  - Uses: adaptive thresholding, morphology, contour finding
  - Returns: (x, y, side) tuple

refine_to_inner_plate(gray: np.ndarray, bbox: tuple, pad: int = 20) → tuple[int, int, int]
  - Refines outer dish bbox to inner plate region
  - Source: Cell 1, lines 54–110
  - Uses: row/column intensity profiles, Otsu threshold, longest run detection

crop_square(img: np.ndarray, bbox: tuple[int, int, int]) → np.ndarray
  - Crops a square region from the image
  - Source: Cell 1, lines 112–115

pad_to_patch_size(img: np.ndarray, patch_size: int = 256) → tuple[np.ndarray, tuple, tuple]
  - Pads image so dimensions are divisible by patch_size
  - Source: Cell 1, lines 118–138
  - Returns: (padded_image, (top, bottom), (left, right))

preprocess_image(path: str, patch_size: int = 256) → dict
  - Convenience function that chains all the above steps
  - Returns dict with: original_image, cropped_dish, padded_dish, bbox, pad_h, pad_w, original_shape

Libraries needed: cv2, numpy
```

---

### Module: inference.py
**Purpose:** Model loading and running predictions on patches.

```
Source: Cell 0 (model loading + f1 metric), Cell 3

Functions:
─────────────────────────────────────────────────────────
f1(y_true, y_pred) → tensor
  - Custom F1 metric needed to load the model
  - Source: Cell 0, lines 28–42
  - Must be passed as custom_objects={"f1": f1} when loading

load_model(model_path: str = None) → keras.Model
  - Loads the Attention U-Net with custom metric
  - Source: Cell 0, lines 50–55
  - If model_path is None, uses config.DEFAULT_MODEL_PATH
  - Add: error handling for file not found, corrupt model

get_spline_window(patch_size: int, power: int = 2) → np.ndarray
  - Creates 2D Hann window for smooth patch blending
  - Source: Cell 3, lines 3–7

predict_smooth_windowing(img_padded: np.ndarray, model, patch_size: int = 256, stride: int = 128, batch_size: int = 32) → np.ndarray
  - Runs inference on all overlapping patches with TTA (horizontal flip)
  - Blends predictions using spline windows
  - Source: Cell 3, lines 9–62
  - Returns: full-size probability map (float32, values 0.0–1.0)

Libraries needed: tensorflow, keras, numpy, scipy.signal
```

---

### Module: postprocessing.py
**Purpose:** Converting the probability map into a clean binary mask.

**Scope (important):** All functions in this module operate on numpy arrays in memory. They do NOT read from or write to disk. The pipeline.py orchestrator passes numpy arrays between modules — file I/O is out of scope for this module.

For batch processing of mask files on disk (used by training data prep, dataset generation, etc.), thin wrapper functions can exist alongside the in-memory versions. These wrappers should loop over files and call the in-memory functions internally — never duplicate the logic. Naming convention for batch wrappers: prefix with `batch_` (e.g., `batch_micro_close_and_bridge`).

```
Source: Cell 3 (threshold function), Cell 4 (unpadding), Cell 7 (morphology + bridging)

Functions (in-memory, used by pipeline.py):
─────────────────────────────────────────────────────────
apply_double_threshold(prob_map: np.ndarray, low: float = 0.05, high: float = 0.25) → np.ndarray
  - Hysteresis thresholding: pixels above 'high' are definite roots,
    pixels between 'low' and 'high' are roots only if connected to definite ones
  - Source: Cell 3, lines 64–67
  - Returns: binary mask (uint8, 0 or 1)

unpad(pred_padded: np.ndarray, pad_h_tuple: tuple, pad_w_tuple: tuple) → np.ndarray
  - Removes the symmetric padding added in preprocessing
  - Source: Cell 4, lines 5–10

micro_close(mask01: np.ndarray, kernel_size: int = 4, iterations: int = 1) → np.ndarray
  - Morphological closing with elliptical kernel to fill tiny gaps
  - Source: Cell 7, batch processing loop (uses cv2.morphologyEx with MORPH_CLOSE)

skeleton_endpoints(skel01: np.ndarray) → np.ndarray
  - Finds endpoints of a skeleton (pixels with only 1 neighbor)
  - Source: Cell 7, lines 38–44
  - Returns: (N, 2) array of (y, x) coordinates

bridge_mask_gaps(mask01: np.ndarray, max_dist: int = 60, max_dx: int = 4, min_dy: int = 8, thickness: int = 3) → np.ndarray
  - Connects nearby skeleton endpoints across disconnected components
  - Only bridges within the same lane, with vertical bias
  - Source: Cell 7, lines 52–108
  - Note: uses a lane_index helper internally (np.searchsorted based lane assignment)
    to enforce same-lane bridging. Not a public function — lives inside this module only.

postprocess_mask(prob_map: np.ndarray, pad_h: tuple, pad_w: tuple) → np.ndarray
  - Convenience function that chains: threshold → unpad → micro_close → bridge
  - This is the function pipeline.py calls
  - Returns: clean binary mask at original crop size

Optional batch wrappers (for training data prep / dataset generation):
─────────────────────────────────────────────────────────
batch_micro_close_and_bridge(in_dir: str, out_dir: str, ...) → None
  - Reads PNG masks from in_dir, applies micro_close + bridge_mask_gaps to each,
    writes results to out_dir
  - Internally loops over files and calls the in-memory functions above
  - NOT used by pipeline.py — used by data preparation scripts

Libraries needed: cv2, numpy, skimage (measure, morphology, filters)
```

**Note on lane edges generation:** Generating `lane_edges.npy` is NOT a postprocessing step — it's a one-time setup utility. See `scripts/generate_lane_edges.py` (Task 2.1.8) for where this lives.

---

### Module: segmentation.py
**Purpose:** Splitting the full root mask into 5 individual plant instances.

```
Source: Cell 8 (lane edge computation), Cell 10 (lane-based instance extraction)

Functions:
─────────────────────────────────────────────────────────
x_crown_from_region(region) → float
  - Finds the median x-coordinate at a component's topmost row
  - Used to assign components to lanes
  - Source: Cell 8, lines 18–22 (also duplicated in Cell 10)

compute_lane_edges(masks: list[np.ndarray], expected_roots: int = 5) → np.ndarray
  - Computes global lane boundaries from multiple masks (in-memory)
  - Uses top-K components by area, ordered left-to-right
  - Source: Cell 8 (entire cell)
  - Returns: array of 6 edge values [0, edge1, edge2, edge3, edge4, W]
  - NOTE: This is a setup-time function, not a runtime pipeline step. It's called
    once by scripts/generate_lane_edges.py (Task 2.1.8) — that script produces
    masks in memory by running preprocessing → inference → postprocessing on a
    directory of original images, then passes the masks here. pipeline.py loads
    the precomputed lane_edges.npy at runtime — it does NOT call this function.

lane_from_x(x: float, lane_edges: np.ndarray, expected_roots: int = 5) → int
  - Maps an x-coordinate to a lane index (0–4)
  - Source: Cell 10, lines 42–44

lane_clean_and_instances(mask01: np.ndarray, lane_edges: np.ndarray, cfg: dict = None) → tuple
  - Main instance segmentation function
  - Removes specks, scores candidates per lane, picks best per lane,
    merges nearby fragments
  - Source: Cell 10, lines 50–180 (the large function)
  - Returns: (clean_union_mask, list_of_5_lane_masks, debug_info)

crop_to_root(mask01: np.ndarray, margin: int = 25) → np.ndarray | None
  - Tight crop around nonzero pixels (used for visualization of individual plants)
  - Source: Cell 10, lines 36–42

Libraries needed: cv2, numpy, skimage (measure, morphology)
```

---

### Module: localization.py
**Purpose:** Finding the root tip for each plant using skeletonization and graph pathfinding.

```
Source: Cell 11 (skeletonization + pathfinding), Cell 14 (coordinate conversion)

Functions:
─────────────────────────────────────────────────────────
analyze_components(mask01: np.ndarray) → tuple[np.ndarray | None, np.ndarray | None]
  - Finds the best skeleton component (tallest, thickest)
  - Source: Cell 11, lines 44–74
  - Returns: (winner_skeleton_mask, distance_transform) or (None, None)

score_start_candidate(G, start_node, end_node, dt, plant_center_x) → float
  - Scores a path start candidate by straightness and centrality
  - Only used for huge plants (>35k pixels)
  - Source: Cell 11, lines 77–110

get_path_from_component(winner_mask: np.ndarray, dt: np.ndarray, plant_area: int) → list[tuple] | None
  - Builds a weighted graph from the skeleton and finds the main root path
    using Dijkstra's algorithm
  - Has three strategies based on plant size: small, standard, huge
  - Source: Cell 11, lines 113–250
  - Returns: list of (y, x) tuples representing the main root path, or None

find_main_root(mask01: np.ndarray) → list[tuple] | None
  - Convenience function combining analyze_components + get_path_from_component
  - Source: Cell 11, find_main_root_component_method function
  - Returns: list of (y, x) coordinates or None

get_root_tip(path: list[tuple]) → tuple[int, int] | None
  - Extracts the root tip (last/lowest point in the path)
  - Source: Cell 12, line: tip_y, tip_x = path[-1]
  - Returns: (y, x) pixel coordinates

pixel_to_mm(pixel_coords: tuple, plate_size_px: int, plate_size_mm: float = 150.0) → tuple[float, float]
  - Converts pixel coordinates to millimeters
  - Source: Cell 14, lines: tip_mm_x = plate_px_x * SCALE
  - Scale = plate_size_mm / plate_size_px

get_confidence_score(prob_map_unpadded: np.ndarray, tip_y: int, tip_x: int, radius: int = 5) → float
  - Extracts confidence at the root tip location from the probability map
  - Average probability in a small region around the tip
  - IMPORTANT: prob_map must be unpadded to match the coordinate space of the root tip.
    The pipeline unpads the prob_map in the postprocessing step, so use that unpadded version here,
    not the raw padded output from inference.
  - Source: NEW (derived from the probability map, not in original notebook as a function)
  - Returns: float between 0.0 and 1.0

Libraries needed: cv2, numpy, networkx, skimage (morphology, measure)
```

---

### Module: pipeline.py
**Purpose:** Chains all modules together into one call.

```
Source: NEW (the notebook runs cells sequentially; this orchestrator doesn't exist yet)

Functions:
─────────────────────────────────────────────────────────
run_pipeline(image_path: str, model_path: str = None, lane_edges: np.ndarray = None) → dict
  - Runs the full pipeline from raw image to root tip results
  - Lane edges handling:
    - If lane_edges is provided, use them directly
    - If lane_edges is None, load from config.DEFAULT_LANE_EDGES_PATH
    - If the file doesn't exist either, raise a clear error:
      "Lane edges not found. Run compute_lane_edges first or provide a path."
  - Steps:
    1. preprocessing.preprocess_image(image_path) → padded dish + metadata (includes bbox)
    2. inference.predict_smooth_windowing(padded_dish, model) → probability map (padded space)
    3. postprocessing.postprocess_mask(prob_map, pad_h, pad_w) → clean binary mask (unpadded)
       Also unpad the prob_map itself for use in confidence scoring later.
    4. segmentation.lane_clean_and_instances(mask, lane_edges) → 5 plant masks + debug_info
       Log debug_info (chosen labels, scores per lane) for diagnostics.
    5. For each plant mask:
       localization.find_main_root(plant_mask) → path
       localization.get_root_tip(path) → (y, x) in pixels
       localization.pixel_to_mm((y, x), plate_size_px) → (y, x) in mm
         NOTE: plate_size_px = bbox[2] (the side length from preprocessing step 1)
       localization.get_confidence_score(prob_map_unpadded, y, x) → confidence
  - Returns dict:
    {
      "image_path": str,
      "binary_mask": np.ndarray,          # full clean mask
      "plant_masks": list[np.ndarray],    # 5 individual masks
      "plants": [
        {
          "lane": int,                     # 0–4
          "root_tip_px": {"x": int, "y": int},
          "root_tip_mm": {"x": float, "y": float},
          "confidence": float,
          "has_root": bool,
          "main_root_path": list[tuple]    # optional, for visualization
        },
        ... (5 entries, one per lane)
      ],
      "processing_time_seconds": float,
      "plate_bbox": tuple                  # (x, y, side)
    }

Libraries needed: all of the above (imports from other modules)
```

---

### Module: cli.py
**Purpose:** Command-line interface (Sprint 2, Issue 2.2)

```
Will call pipeline.run_pipeline() and format output.
Entry point registered in pyproject.toml.
Not mapped here — will be built from scratch in Sprint 2.
```

---

## Task 1.2.4: Interface Definitions

Data flow between modules, showing exactly what each module passes to the next.

### preprocessing → inference

```
preprocessing.preprocess_image(image_path) returns:
{
  "original_image": np.ndarray,        # grayscale, shape (H, W), dtype uint8
  "cropped_dish": np.ndarray,          # grayscale, shape (~2734, ~2734), dtype uint8
  "padded_dish": np.ndarray,           # grayscale, shape divisible by 256, dtype uint8
  "bbox": tuple[int, int, int],        # (x, y, side)
  "pad_h": tuple[int, int],            # (top, bottom)
  "pad_w": tuple[int, int],            # (left, right)
  "original_shape": tuple[int, int]    # (H, W) of original image
}

inference needs: padded_dish (np.ndarray)
```

### inference → postprocessing

```
inference.predict_smooth_windowing(padded_dish, model) returns:
  np.ndarray  # shape matches padded_dish, dtype float32, values 0.0–1.0
              # this is the probability map (how confident the model is at each pixel)

postprocessing needs: prob_map (np.ndarray), pad_h (tuple), pad_w (tuple)
```

### postprocessing → segmentation

```
postprocessing.postprocess_mask(prob_map, pad_h, pad_w) returns:
  np.ndarray  # shape matches cropped_dish (unpadded), dtype uint8, values 0 or 1
              # clean binary mask after thresholding + morphology + bridging

segmentation needs: mask01 (np.ndarray), lane_edges (np.ndarray)
```

### segmentation → localization

```
segmentation.lane_clean_and_instances(mask01, lane_edges) returns:
  tuple of:
    clean_union: np.ndarray        # combined clean mask, dtype uint8, 0 or 1
    lane_masks: list[np.ndarray]   # 5 masks, one per plant lane, each dtype uint8, 0 or 1
                                   # empty lanes have all-zero masks
    debug_info: dict               # scoring details — pipeline.py should log this for diagnostics
                                   # contains: chosen_labels, chosen_scores, top_band_px, y_cut_px

localization needs: each lane_mask individually (np.ndarray)
```

### localization → output

```
For each lane_mask, localization produces:
  find_main_root(mask01) returns:
    list[tuple[int, int]] or None  # list of (y, x) pixel coordinates of main root path
                                   # None if no root found

  get_root_tip(path) returns:
    tuple[int, int] or None        # (y, x) pixel coordinates of root tip

  pixel_to_mm(coords, plate_size_px) returns:
    tuple[float, float]            # (y_mm, x_mm)
    NOTE: plate_size_px = bbox[2] (the side length returned by preprocessing step 1)

  get_confidence_score(prob_map_unpadded, y, x) returns:
    float                          # 0.0–1.0
    NOTE: prob_map must be unpadded to match root tip coordinate space.
    Pipeline step 3 should unpad both the binary mask AND the probability map.
```

---

## Summary: Notebook Cell → Module Mapping

| Cell | Description | Target module |
|------|-------------|---------------|
| 0 | Imports, config, model loading, f1 metric | config.py, inference.py |
| 1 | Dish detection, refinement, crop, pad | preprocessing.py |
| 2 | Visualization of crops (display only) | Not in core package |
| 3 | Spline window, smooth inference, thresholding | inference.py, postprocessing.py |
| 4 | Unpadding, save full-size masks | postprocessing.py |
| 5 | Confidence heatmaps (display only) | Not in core package |
| 6 | Overlay visualization (display only) | Not in core package |
| 7 | Micro-close, gap bridging | postprocessing.py |
| 8 | Lane edge computation | segmentation.py |
| 9 | Lane boundary visualization (display only) | Not in core package |
| 10 | Lane-based instance extraction | segmentation.py |
| 11 | Skeletonization, pathfinding, main root | localization.py |
| 12 | Root tip visualization (display only) | Not in core package |
| 13 | Root length measurement | localization.py (optional) |
| 14 | Coordinate conversion (px to mm) | localization.py |

---

## Team Module Assignments

| Module | Assigned to | Complexity | Notes |
|--------|-------------|------------|-------|
| config.py | Samara | Low | Every module imports from it, so it's a blocker. Should be done first. |
| preprocessing.py | Floris | Medium | Clean, self-contained functions |
| inference.py | Ahmed | High | Knows the model best, TF custom metric |
| postprocessing.py | Stijn | Medium | Multiple operations, bridging logic is complex |
| segmentation.py | Nazeer | High | Largest single function (lane_clean_and_instances), many parameters |
| localization.py | Samara | High | Graph construction + Dijkstra, three plant-size strategies |
| pipeline.py | Samara | Low | Wiring, depends on all other modules having defined interfaces |
| cli.py | Ahmed | — | Built after modules work |
| scripts/generate_lane_edges.py | Samara | Low | One-time setup script, runs preprocessing → inference → postprocessing on a directory of images, then calls compute_lane_edges to produce lane_edges.npy |
