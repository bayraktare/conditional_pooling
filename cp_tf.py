"""
This code was created by Ertugrul Bayraktar and Cihat Bora Yigit for their paper titled "Conditional-Pooling for Improved Data Transmission," 
which has been accepted as a journal paper in Pattern Recognition. The copyright of the code belongs to Ertugrul Bayraktar and Cihat Bora Yigit, 
and it is licensed under the MIT License.
"""

import tensorflow as tf

def ConditionalPoolingLayer(x , size, stride):
    """
    The ConditionalPoolingLayer() function takes three inputs:
    x: The input tensor, which is expected to be a 4D tensor with the shape (batch_size, height, width, channels).
    size: The size of the pooling kernel.
    stride: The stride of the pooling kernel.
    
    For each channel, it extracts patches of size size with stride stride.
    It returns the pooled tensor following the concatenation of pooled channels.
    """
    
    # Split the input into RGB channels
    rgb = tf.split(x, x.shape[3], 3) 

    part = []
    for i in range(len(rgb)):
        # Extract patches from each RGB channel
        x_kernels = tf.image.extract_patches(rgb[i], (1, size, size, 1), (1, stride, stride, 1), (1, 1, 1, 1), 'VALID')

        # Calculate the mean of each patch
        mean = tf.reduce_mean(x_kernels, axis=3, keepdims=True)

        # Count the number of values greater and less than the mean    
        count_greater = tf.reduce_sum(tf.cast(tf.greater(x_kernels, mean), dtype=tf.float32))
        count_less = tf.reduce_sum(tf.cast(tf.less(x_kernels, mean), dtype=tf.float32))

        # Calculate the mean of values greater and less than the mean
        mean_greater = tf.reduce_mean(tf.where(tf.greater(x_kernels, mean), x_kernels, tf.zeros_like(x_kernels)), axis=3, keepdims=True)
        mean_less = tf.reduce_mean(tf.where(tf.less(x_kernels, mean), x_kernels, tf.zeros_like(x_kernels)), axis=3, keepdims=True)

        # Determine if there are more values greater than less
        greater_than_less = tf.greater(count_greater, count_less)

        # Combine the means based on the count comparison
        final_mean = tf.where(greater_than_less, mean_greater, mean_less)

        # Check if the counts are equal and use a uniform mean
        equal_counts = tf.equal(count_greater, count_less)
        final_mean = tf.where(equal_counts, tf.reduce_mean(x_kernels, axis=3, keepdims=True), final_mean)

        # Calculate the final conditional mean for this channel
        x_pooled = tf.reduce_mean(final_mean, axis=3, keepdims=True)

        # Store the channel result
        part.append(x_pooled)

    # Concatenate the results from all channels
    res = tf.concat(part, axis=3)
    return res
