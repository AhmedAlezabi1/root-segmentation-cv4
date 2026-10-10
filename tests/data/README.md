# Test fixtures

Test images and expected outputs used by integration tests
(`tests/test_pipeline_integration.py`).

## Files

| File | Purpose |
|---|---|
| `valid_dish.png` | Real petri dish image from the Block B test set. The full pipeline should process it successfully and return 5 plant entries with at least 4 root detections. |
| `corrupt.png` | Intentionally invalid file (random bytes saved with a `.png` extension). Used to verify the pipeline raises a clear error on unreadable input. |

## Notes

- Fixtures used here are for **integration testing**, full pipeline runs on real images.
- Unit tests (e.g., `test_localization.py`, `test_segmentation.py`) use synthetic
  numpy arrays built in code, not files. Don't add fixtures here for those.
- If you regenerate the model or `lane_edges.npy`, the pipeline's exact output
  values may change. Re-run the integration test and update assertions if needed.