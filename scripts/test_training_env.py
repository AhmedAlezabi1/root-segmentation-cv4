"""
Test script submitted as an Azure ML job to verify the cv4-training
environment can import TensorFlow and see GPU devices.

This script runs INSIDE the Azure ML environment, not locally.
Submit it with:

    uv run --group azure python scripts/submit_test_training_env.py

Do not run this file directly — it has no Azure ML dependencies and
is intended to execute inside the registered cv4-training container.
"""

import sys

import tensorflow as tf

print(f"Python     : {sys.version}")
print(f"TensorFlow : {tf.__version__}")

gpus = tf.config.list_physical_devices("GPU")
print(f"GPUs found : {len(gpus)}")
for gpu in gpus:
    print(f"  {gpu}")

if not gpus:
    print("WARNING: No GPU devices found — running on CPU only.")
else:
    print("OK: TensorFlow can see the GPU.")
