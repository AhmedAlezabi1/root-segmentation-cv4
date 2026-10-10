"""Unit tests for the inference module.

Tests the 4 core functions: f1, load_model, get_spline_window,
and predict_smooth_windowing. Uses both fake and real models.

Run with: uv run pytest tests/test_inference.py -v
Coverage: uv run pytest tests/test_inference.py --cov=cv4.inference --cov-report=term-missing
"""

import os

import numpy as np
import pytest
import tensorflow as tf

from cv4.config import DEFAULT_MODEL_PATH
from cv4.inference import f1, get_spline_window, load_model, predict_smooth_windowing


class TestF1:
    """Tests for the f1 custom metric function.

    The f1 function is a custom Keras metric needed to load the
    trained UNet model. These tests check that it returns correct
    scores for known inputs.
    """

    def test_perfect_predictions_returns_one(self):
        """F1 should be 1.0 when predictions match labels exactly."""
        y_true = tf.constant([1.0, 1.0, 0.0, 0.0])
        y_pred = tf.constant([1.0, 1.0, 0.0, 0.0])
        result = f1(y_true, y_pred)
        assert result.numpy() == pytest.approx(1.0, abs=0.01)

    def test_wrong_predictions_returns_zero(self):
        """F1 should be 0.0 when predictions are completely wrong."""
        y_true = tf.constant([1.0, 1.0, 1.0, 1.0])
        y_pred = tf.constant([0.0, 0.0, 0.0, 0.0])
        result = f1(y_true, y_pred)
        assert result.numpy() == pytest.approx(0.0, abs=0.01)

    def test_partial_predictions(self):
        """F1 should be between 0 and 1 for partial matches."""
        y_true = tf.constant([1.0, 1.0, 0.0, 0.0])
        y_pred = tf.constant([1.0, 0.0, 0.0, 0.0])
        result = f1(y_true, y_pred)
        score = result.numpy()
        assert 0.0 < score < 1.0

    def test_returns_tensor(self):
        """F1 should return a TensorFlow tensor."""
        y_true = tf.constant([1.0, 0.0])
        y_pred = tf.constant([1.0, 0.0])
        result = f1(y_true, y_pred)
        assert isinstance(result, tf.Tensor)


class TestLoadModel:
    """Tests for the load_model function.

    Checks error handling for bad paths and corrupt files,
    and also tests loading the real UNet model if available.
    """

    def test_file_not_found_raises_error(self):
        """Should raise FileNotFoundError for a path that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_model("this/path/does/not/exist.h5")

    def test_invalid_file_raises_value_error(self, tmp_path):
        """Should raise ValueError for a file that isn't a valid model."""
        fake_model = tmp_path / "fake_model.h5"
        fake_model.write_text("this is not a real model")
        with pytest.raises(ValueError):
            load_model(str(fake_model))

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_MODEL_PATH),
        reason="Real model file not available",
    )
    def test_load_real_model(self):
        """Should load the real UNet model from the weights folder."""
        model = load_model(DEFAULT_MODEL_PATH)
        assert model is not None
        assert isinstance(model, tf.keras.Model)

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_MODEL_PATH),
        reason="Real model file not available",
    )
    def test_real_model_input_shape(self):
        """Real model should expect 256x256 input patches."""
        model = load_model(DEFAULT_MODEL_PATH)
        input_shape = model.input_shape
        assert input_shape[1] == 256
        assert input_shape[2] == 256


class TestGetSplineWindow:
    """Tests for the get_spline_window function.

    The spline window is used to blend overlapping patches smoothly.
    These tests check its shape, values, and symmetry.
    """

    def test_output_shape(self):
        """Window shape should match (patch_size, patch_size)."""
        window = get_spline_window(256)
        assert window.shape == (256, 256)

    def test_output_shape_different_size(self):
        """Should work with different patch sizes."""
        window = get_spline_window(128)
        assert window.shape == (128, 128)

    def test_returns_numpy_array(self):
        """Should return a numpy array."""
        window = get_spline_window(256)
        assert isinstance(window, np.ndarray)

    def test_values_between_zero_and_one(self):
        """All values should be between 0 and 1."""
        window = get_spline_window(256)
        assert window.min() >= 0.0
        assert window.max() <= 1.0

    def test_center_is_highest(self):
        """Center of the window should have the highest value."""
        window = get_spline_window(256)
        center = window[128, 128]
        corner = window[0, 0]
        assert center > corner

    def test_window_is_symmetric(self):
        """Window should be the same flipped horizontally and vertically."""
        window = get_spline_window(256)
        assert np.allclose(window, window[::-1, :])
        assert np.allclose(window, window[:, ::-1])

    def test_higher_power_sharper_falloff(self):
        """Higher power should make corner values even smaller."""
        window_low = get_spline_window(256, power=1)
        window_high = get_spline_window(256, power=4)
        assert window_high[0, 0] <= window_low[0, 0]


class TestPredictSmoothWindowing:
    """Tests for the predict_smooth_windowing function.

    Uses a tiny fake model for most tests to keep them fast.
    Also includes a real model test if the weights file exists.
    """

    @pytest.fixture
    def dummy_model(self):
        """Create a tiny fake model for testing.

        This avoids needing the real UNet for basic tests.
        It takes (batch, 256, 256, 1) and returns same shape.
        """
        inputs = tf.keras.Input(shape=(256, 256, 1))
        x = tf.keras.layers.Conv2D(1, 3, padding="same", activation="sigmoid")(inputs)
        return tf.keras.Model(inputs, x)

    def test_output_shape(self, dummy_model):
        """Output should have same shape as input image."""
        fake_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        result = predict_smooth_windowing(fake_image, dummy_model, patch_size=256, stride=128)
        assert result.shape == (512, 512)

    def test_output_is_numpy_array(self, dummy_model):
        """Should return a numpy array."""
        fake_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        result = predict_smooth_windowing(fake_image, dummy_model, patch_size=256, stride=128)
        assert isinstance(result, np.ndarray)

    def test_output_values_between_zero_and_one(self, dummy_model):
        """Probability map values should be between 0 and 1."""
        fake_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        result = predict_smooth_windowing(fake_image, dummy_model, patch_size=256, stride=128)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_output_dtype_is_float(self, dummy_model):
        """Output should be float, not integer."""
        fake_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        result = predict_smooth_windowing(fake_image, dummy_model, patch_size=256, stride=128)
        assert np.issubdtype(result.dtype, np.floating)

    def test_black_image_low_predictions(self, dummy_model):
        """A completely black image should produce low probability values.

        If the model sees nothing (all zeros), it shouldn't predict
        roots everywhere. The average prediction should be below 0.5.
        """
        black_image = np.zeros((512, 512), dtype=np.uint8)
        result = predict_smooth_windowing(black_image, dummy_model, patch_size=256, stride=128)
        assert result.mean() < 0.5

    def test_white_image(self, dummy_model):
        """A completely white image should produce valid output.

        Even with an all-white image, the function should still
        return a proper probability map with correct shape and
        values between 0 and 1.
        """
        white_image = np.full((512, 512), 255, dtype=np.uint8)
        result = predict_smooth_windowing(white_image, dummy_model, patch_size=256, stride=128)
        assert result.shape == (512, 512)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    @pytest.mark.skipif(
        not os.path.exists(DEFAULT_MODEL_PATH),
        reason="Real model file not available",
    )
    def test_with_real_model(self):
        """Should produce valid output with the real UNet model.

        Loads the actual trained model and runs prediction on a
        random image to make sure the full pipeline works end to end.
        """
        model = load_model(DEFAULT_MODEL_PATH)
        fake_image = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        result = predict_smooth_windowing(fake_image, model, patch_size=256, stride=256)
        assert result.shape == (512, 512)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
