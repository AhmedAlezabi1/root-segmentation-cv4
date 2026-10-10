"""Residual Attention U-Net architecture.

Combines two extensions to the standard U-Net: residual connections inside
each conv block (to keep gradients flowing through the deep encoder/decoder)
and attention gates on each skip connection (to suppress irrelevant encoder
features before they reach the decoder). Both extensions matter more for
thin-structure segmentation than for the typical organ-segmentation
benchmarks U-Nets were designed against.
"""

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Input,
    MaxPooling2D,
    UpSampling2D,
    add,
    concatenate,
    multiply,
)
from tensorflow.keras.models import Model


def residual_block(x, filters, stride=1):
    """Apply a two-conv residual block with an identity (or projection) shortcut.

    When the input channel count or spatial stride differs from the block
    output, a 1x1 projection conv reshapes the shortcut so the addition is
    well-defined. Otherwise the shortcut is the unchanged input tensor.

    Args:
        x: Input tensor.
        filters: Output channel count for both conv layers.
        stride: Stride for the first conv. Use 1 for same-resolution blocks.

    Returns:
        Output tensor after the residual addition and final ReLU.
    """
    shortcut = x

    x = Conv2D(filters, (3, 3), strides=stride, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def attention_block(x, gating, inter_shape):
    """Apply an attention gate over a skip connection.

    The gating signal (a lower-resolution decoder feature map) is broadcast
    against the higher-resolution encoder feature map x to produce a soft
    spatial mask in [0, 1]. The mask is then upsampled to the resolution of
    x and multiplied element-wise, suppressing encoder activations that the
    decoder considers irrelevant at this stage of reconstruction.

    Args:
        x: Encoder skip-connection tensor (higher spatial resolution).
        gating: Decoder gating tensor (lower spatial resolution).
        inter_shape: Channel count of the intermediate projection space.

    Returns:
        The attention-gated feature map, projected to inter_shape channels.
    """
    shape_x = K.int_shape(x)

    theta_x = Conv2D(inter_shape, (2, 2), strides=(1, 1), padding="same")(x)
    phi_g = Conv2D(inter_shape, (1, 1), padding="same")(gating)

    concat_xg = add([theta_x, phi_g])
    act_xg = Activation("relu")(concat_xg)

    psi = Conv2D(1, (1, 1), padding="same")(act_xg)
    sigmoid_xg = Activation("sigmoid")(psi)

    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_sigmoid_xg = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)

    y = multiply([x, upsample_sigmoid_xg])
    return Conv2D(inter_shape, (1, 1), padding="same")(y)


def build_res_attention_unet(img_height, img_width, img_channels):
    """Construct the full Residual Attention U-Net.

    Four-level encoder/decoder with filter counts 32, 64, 128, 256 and a
    bottleneck at 512. Each skip connection passes through an attention gate
    before being concatenated with the corresponding decoder upsample.

    Args:
        img_height: Patch height in pixels (typically PATCH_SIZE).
        img_width: Patch width in pixels (typically PATCH_SIZE).
        img_channels: Number of input channels (1 for grayscale roots).

    Returns:
        An uncompiled Keras Model with a single sigmoid output channel.
    """
    inputs = Input((img_height, img_width, img_channels))

    x = Conv2D(32, (3, 3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    c1 = residual_block(x, 32)
    p1 = MaxPooling2D((2, 2))(c1)
    c2 = residual_block(p1, 64)
    p2 = MaxPooling2D((2, 2))(c2)
    c3 = residual_block(p2, 128)
    p3 = MaxPooling2D((2, 2))(c3)
    c4 = residual_block(p3, 256)
    p4 = MaxPooling2D((2, 2))(c4)

    c5 = residual_block(p4, 512)

    u6 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c5)
    att6 = attention_block(x=c4, gating=u6, inter_shape=256)
    u6 = concatenate([u6, att6])
    c6 = residual_block(u6, 256)

    u7 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    att7 = attention_block(x=c3, gating=u7, inter_shape=128)
    u7 = concatenate([u7, att7])
    c7 = residual_block(u7, 128)

    u8 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    att8 = attention_block(x=c2, gating=u8, inter_shape=64)
    u8 = concatenate([u8, att8])
    c8 = residual_block(u8, 64)

    u9 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c8)
    att9 = attention_block(x=c1, gating=u9, inter_shape=32)
    u9 = concatenate([u9, att9])
    c9 = residual_block(u9, 32)

    outputs = Conv2D(1, (1, 1), activation="sigmoid")(c9)
    return Model(inputs=[inputs], outputs=[outputs])
