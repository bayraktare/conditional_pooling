# Conditional-Pooling
The repository for [Conditional-Pooling for Improved Data Transmission](https://www.sciencedirect.com/science/article/pii/S0031320323006763), which is accepted as journal paper at Pattern Recognition. 

__Brief description of Conditional-Pooling:__ Conditional-Pooling is an innovative pooling technique designed to enhance feature extraction and representation within deep learning models, particularly in the field of computer vision. It differs from traditional pooling methods like average and max pooling by dynamically considering the distribution of pixel values relative to their mean. By assigning greater weight to pixels exceeding the mean, Conditional-Pooling adaptively captures salient features, making it highly effective in scenarios involving complex visual data with inherent noise and variability. This technique has shown significant promise in improving the performance of deep neural networks across various image-related tasks, making it a valuable addition to the toolkit of machine learning practitioners and researchers.

![fig2(1)](https://github.com/bayraktare/conditional_pooling/assets/17570285/b3659949-a262-4e04-9000-5d27ab4a8005)
Comparison of avg. and max-pooling with Conditional-Pooling, which generates outputs with the aim of making input patterns more salient with a sensible algorithm by controlling the neighbouring pixels. However, avg. and max-pooling are unchangeable techniques, which suffer from invariances. Conditional-Pooling highlights the influence of the majority of neighbouring pixels inside the kernel.


__Why Conditional-Pooling is important in the context of deep learning and pattern recognition?__ Conditional-Pooling is vital in deep learning and pattern recognition due to its unique ability to adaptively enhance feature extraction. This innovation is crucial as it significantly improves the representation of complex visual data, making it more effective in tasks involving intricate patterns, noise, and variability. By addressing the limitations of traditional pooling methods, Conditional-Pooling empowers deep learning models to achieve higher accuracy and robustness in tasks like image classification, object detection, and more. Its importance lies in its capacity to elevate the performance of neural networks in real-world scenarios where data can be challenging and diverse.

__The Algorithm of Conditional-Pooling is as follows:__

![Screenshot 2023-09-14 at 12-19-14 Pattern_Recognition](https://github.com/bayraktare/conditional_pooling/assets/17570285/4c21ff79-9a78-4868-a7c8-35e3265ce40b)


## Getting Started
__Installation Instructions for Conditional-Pooling__
Conditional-Pooling can be easily set up and used in your own projects with the following steps:

__Prerequisites:__

    Python 3.x
    PyTorch 1.x or TensorFlow 2.x (either framework is sufficient)

__Installation Steps:__

    Clone the Conditional-Pooling repository: git clone https://github.com/yourusername/conditional-pooling.git

    Navigate to the repository directory: cd conditional-pooling

Conditional-Pooling is available for both PyTorch and TensorFlow. Choose the framework you prefer.

__For PyTorch:__

    Use the cp_torch.py file for PyTorch-based projects.

__For TensorFlow:__

    Use the cp_tf.py file for TensorFlow-based projects.

__Usage:__

Here's how to use Conditional-Pooling in your own project:

    Import the Conditional-Pooling module: 

    For PyTorch: from cp_torch import CustomPoolingLayer

    For TensorFlow: from cp_tf import custom_pooling_layer

- Make an instance of the Conditional-Pooling layer within your neural network architecture.

- Use the Conditional-Pooling layer in your forward pass as demonstrated in the provided script.

- Customize the parameters (e.g., kernel size, stride, padding) as needed for your specific task.

- With these installation instructions, you can easily set up and leverage Conditional-Pooling in your PyTorch or TensorFlow projects to enhance feature representation and discrimination in your deep learning models.

__Code Examples:__

We have included code examples of Conditional-Pooling for different tasks, such as image classification and object detection, in the repository. You can find these examples in the examples directory. These examples demonstrate how to integrate Conditional-Pooling into various deep learning tasks to enhance feature representation and discrimination.

If you find this code or paper useful, please cite as:

    @article{bayraktar-yigit-conditional-pooling,
      author    = {Ertugrul Bayraktar and Cihat Bora Yigit},
      title     = {Conditional-Pooling for Improved Data Transmission},
      journal   = {Pattern Recognition},
      year      = {2023},
      pages     = {109978},
      url       = {https://www.sciencedirect.com/science/article/pii/S0031320323006763},
      doi       = {https://doi.org/10.1016/j.patcog.2023.109978},
      issn      = {0031-3203}
    }

  
  


    
