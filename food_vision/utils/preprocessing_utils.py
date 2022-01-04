import tensorflow as tf


def preprocess_image(img, label, img_shape=224, scale=True):
    """
    Converts image datatype from 'uint8' -> 'float32' and reshapes image to
    [img_shape, img_shape, color_channels]

    Parameters
    ----------
    img (tensor): image tensor
    label (tensor): the label of the input image
    img_shape (int): size to resize target image to, default 224
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    img = tf.image.resize(img, [img_shape, img_shape])
    img = tf.cast(img, tf.float32)
    if scale:
        # Rescale the image (get all values between 0 and 1)
        return img / 255.0, label
    else:
        return img, label
