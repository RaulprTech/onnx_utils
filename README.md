# ONNX Utils
This repository hosts resources aimed at simplifying the use and understanding of ONNX (Open Neural Network Exchange), an open format for representing machine learning models.

## Contents
* ``onnx_utils`` File: Contains a set of useful functions for querying information within ONNX files.
* ``onnx_classes File``: Includes a collection of classes designed to facilitate information retrieval within ONNX files.
* ``examples Folder``: Houses notebooks detailing the functionality of each function and class in the onnx_utils and onnx_classes files. It also provides practical usage examples.
* ``resnet18 File``: Offers an example demonstrating how to export a model in .pth format to .onnx. This process is also explained in the notebook within the examples folder.
* ``models Folder``: Contains example models that you can use to test the repository's functions.

## Usage
To get started, explore the notebook in the examples folder to grasp the functionality of each function and class. The necessary dependencies for running the code are listed in the "requirements.txt" file.

## Classes and Methods
Here's a brief overview of the classes and methods included in the repository:

### Layer Class
Base class for ONNX layer operations.

* __init__(onnx_model_path, layer_name): Initializes the layer object.
* ``get_inputs()``: Retrieves the input of the layer.
* ``get_outputs()``: Retrieves the output of the layer.
* ``get_attribute(attr_name)``: Retrieves the attribute of the layer.

### ConvLayer Class
Derived from Layer, tailored for Convolutional layers.

* ``get_weights()``: Retrieves the weights of the Conv layer.
* ``get_bias()``: Retrieves the bias of the Conv layer.
* ``get_kernel_size()``: Retrieves the kernel size of the Conv layer.
* ``get_stride()``: Retrieves the stride of the Conv layer.
* ``get_padding()``: Retrieves the padding of the Conv layer.
* ``get_groups()``: Retrieves the groups of the Conv layer.

### ReluLayer Class
Derived from Layer, tailored for ReLU activation layers.

### MaxPoolLayer Class
Derived from Layer, tailored for MaxPool layers.

* ``get_kernel_size()``: Retrieves the kernel size of the MaxPool layer.
* ``get_stride()``: Retrieves the stride of the MaxPool layer.
* ``get_padding()``: Retrieves the padding of the MaxPool layer.

### AddLayer Class
Derived from Layer, tailored for Add layers.

### GlobalAveragePoolLayer Class
Derived from Layer, tailored for GlobalAveragePool layers.

### FlattenLayer Class
Derived from Layer, tailored for Flatten layers.

### GemmLayer Class
Derived from Layer, tailored for Gemm (fully connected) layers.

* ``get_weights()``: Retrieves the weights of the Gemm layer.
* ``get_bias()``: Retrieves the bias of the Gemm layer.

Feel free to explore, contribute, and provide feedback to improve this resource.