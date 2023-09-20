"""
This code was created by Ertugrul Bayraktar and Cihat Bora Yigit for their paper titled "Conditional-Pooling for Improved Data Transmission," 
which has been accepted as a journal paper in Pattern Recognition. The copyright of the code belongs to Ertugrul Bayraktar and Cihat Bora Yigit, 
and it is licensed under the MIT License.
"""


import tensorflow as tf

def custom_pooling(x , size, stride):
    """
    The custom_pooling() function takes three inputs:
    x: The input tensor, which is expected to be a 4D tensor with the shape (batch_size, height, width, channels).
    size: The size of the pooling kernel.
    stride: The stride of the pooling kernel.
    
    For each channel, it extracts patches of size size with stride stride.
    It returns the pooled tensor following the concatenation of pooled channels.
    """

    rgb = tf.split(x, x.shape[3], 3)

    part = []
    for i in range(len(rgb)):
        x_kernels = tf.image.extract_patches(rgb[i], (1, size, size, 1), (1, stride, stride, 1), (1, 1, 1, 1), 'VALID')

        mean = tf.reduce_mean(x_kernels, axis=3, keepdims=True)

        count_greater = tf.reduce_sum(tf.cast(tf.greater(x_kernels, mean), dtype=tf.float32))
        count_less = tf.reduce_sum(tf.cast(tf.less(x_kernels, mean), dtype=tf.float32))

        mean_greater = tf.reduce_mean(tf.where(tf.greater(x_kernels, mean), x_kernels, tf.zeros_like(x_kernels)), axis=3, keepdims=True)
        mean_less = tf.reduce_mean(tf.where(tf.less(x_kernels, mean), x_kernels, tf.zeros_like(x_kernels)), axis=3, keepdims=True)

        greater_than_less = tf.greater(count_greater, count_less)
        final_mean = tf.where(greater_than_less, mean_greater, mean_less)

        equal_counts = tf.equal(count_greater, count_less)
        final_mean = tf.where(equal_counts, tf.reduce_mean(x_kernels, axis=3, keepdims=True), final_mean)

        x_pooled = tf.reduce_mean(final_mean, axis=3, keepdims=True)

        part.append(x_pooled)

    res = tf.concat(part, axis=3)
    return res
