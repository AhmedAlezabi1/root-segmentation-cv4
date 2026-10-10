"""Training-specific configuration constants.

Kept separate from cv4.config (which is inference-side) so that training
dependencies do not leak into the inference deployment image. Values here are
defaults: train.py exposes them as CLI arguments, and the Azure ML sweep job
overrides them per trial.
"""

# Patch size the U-Net operates on. Matches the inference pipeline's patch
# size in cv4.config so that the trained model is directly usable downstream
# without resizing.
PATCH_SIZE = 256

# Default training hyperparameters. Overridable from the command line.
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 50

# Focal Tversky loss constants. Chosen during Block B research to penalize
# missed roots (false negatives) more heavily than spurious detections,
# reflecting the cost asymmetry in plant phenotyping: a missed root is
# worse than a false positive a user can visually reject.
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
TVERSKY_GAMMA = 4 / 3

# Early stopping and learning rate schedule defaults.
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.2
MIN_LEARNING_RATE = 1e-7

# Validation F1 threshold below which a trained model will not be registered
# in the Azure ML Model Registry. Set well below the Block B baseline of 0.88
# so it functions as a regression guard, not an aspirational target.
REGISTRATION_F1_THRESHOLD = 0.80
