from __future__ import division

import tensorflow as tf
import numpy as np

masks = np.ones([64, 32, 32, 3])
masks[:, 16:48, 16:48, :] = 0.0
print(1 - masks)

with tf.Session() as sess:
    a = tf.constant([
        [[[1], [2], [3], [4]], [[5], [6], [7], [8]], [[9], [10], [11], [12]], [[13], [14], [15], [16]]],
        [[[0], [1], [2], [3]], [[4], [5], [6], [7]], [[8], [9], [10], [11]], [[12], [13], [14], [15]]]], dtype=tf.float32)
    print(a.eval())
    b = tf.map_fn(lambda img: tf.image.crop_to_bounding_box(img, 1,1,2,2), a)
    print(b.eval())
