import tensorflow as tf
import numpy as np
def weak_aug(x):
    x = tf.image.resize_with_pad(
       x, 32 + 6, 32 + 6,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.image.random_crop(x, [tf.shape(x)[0], 32, 32, 3])
    #x=tf.keras.preprocessing.image.random_shift(np.array(x),0.1,0.1,row_axis=1,col_axis=2,channel_axis=3)
    x = tf.image.random_flip_left_right(x)
    return x

def strong_aug(x):
    x = tf.image.resize_with_pad(
       x, 32 + 6, 32 + 6,method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    x = tf.image.random_crop(x, [tf.shape(x)[0], 32, 32, 3])
    #x=tf.keras.preprocessing.image.random_shift(np.array(x),0.1,0.1,row_axis=1,col_axis=2,channel_axis=3)
    x = tf.image.random_flip_left_right(x)
    x=tf.map_fn(cutout_numpy,x)
    # for i in range(len(x)):
    #     x[i]=cutout_numpy(x[i])
    return x

def create_cutout_mask(img_height, img_width, num_channels, size):
    """Creates a zero mask used for cutout of shape `img_height` x `img_width`.

    Args:
      img_height: Height of image cutout mask will be applied to.
      img_width: Width of image cutout mask will be applied to.
      num_channels: Number of channels in the image.
      size: Size of the zeros mask.

    Returns:
      A mask of shape `img_height` x `img_width` with all ones except for a
      square of zeros of shape `size` x `size`. This mask is meant to be
      elementwise multiplied with the original image. Additionally returns
      the `upper_coord` and `lower_coord` which specify where the cutout mask
      will be applied.
    """
    assert img_height == img_width

    # Sample center where cutout mask will be applied
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    # Determine upper right and lower left corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2),
                   min(img_width, width_loc + size // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]
    assert mask_height > 0
    assert mask_width > 0

    mask = np.ones((img_height, img_width, num_channels))
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = 0
    return mask, upper_coord, lower_coord


def cutout_numpy(img, size=16):
    """Apply cutout with mask of shape `size` x `size` to `img`.

    The cutout operation is from the paper https://arxiv.org/abs/1708.04552.
    This operation applies a `size`x`size` mask of zeros to a random location
    within `img`.

    Args:
      img: Numpy image that cutout will be applied to.
      size: Height/width of the cutout mask that will be

    Returns:
      A numpy tensor that is the result of applying the cutout mask to `img`.
    """
    if size <= 0:
        return img
    assert len(img.shape) == 3
    img_height, img_width, num_channels = img.shape
    mask = create_cutout_mask(img_height, img_width, num_channels, size)[0]
    return img * mask