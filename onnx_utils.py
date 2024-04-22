import torch
import onnx
import os
from torchvision import models
# import numpy as np

def get_layers(onnx_model_path):
    """
    Retrieve the layers and their frequency from an ONNX model.

    Parameters:
    onnx_model_path (str): The path to the ONNX model file.

    Returns:
    dict: A dictionary containing the layer names as keys and their frequency as values.
    """
    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Get the model's graph
    graph = model.graph

    # Initialize a dictionary to store layers and their frequency
    layers = {}

    # Iterate over the graph's nodes
    for node in graph.node:
        layer_name = node.op_type
        layers[layer_name] = layers.get(layer_name, 0) + 1

    return layers


def get_layer_names(onnx_model_path):
    """
    Retrieve the names of the layers from an ONNX model.

    Parameters:
    onnx_model_path (str): The path to the ONNX model file.

    Returns:
    list: A list of layer names.
    """
    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Get the model's graph
    graph = model.graph

    # Initialize a list to store layer names
    layer_names = []

    # Iterate over the graph's nodes
    for node in graph.node:
        layer_names.append(node.name)

    return layer_names


def input_to_value(onnx_model_path, input_name):
    """
    Retrieve the value of a specific input in an ONNX model.

    Parameters:
    onnx_model_path (str): The path to the ONNX model file.
    input_name (str): The name of the input tensor.

    Returns:
    numpy.ndarray: The value of the input tensor.
    """
    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Get the model's graph
    graph = model.graph

    # Iterate over the initializers in the graph
    for init_tensor in graph.initializer:
        if input_name in init_tensor.name:
            # Convert the initializer tensor to a numpy array
            value = onnx.numpy_helper.to_array(init_tensor)
            return value
    print("Value not found")


def get_inputs(onnx_model_path, layer_name):
    """
    Retrieve the input names of a specific layer in an ONNX model.

    Parameters:
    onnx_model_path (str): The path to the ONNX model file.
    layer_name (str): The name of the layer.

    Returns:
    list: A list of input names for the specified layer.
    """
    # Load the ONNX model
    model = onnx.load(onnx_model_path)

    # Get the model's graph
    graph = model.graph

    # Iterate over the graph's nodes
    for node in graph.node:
        if node.name == layer_name:
            return node.input



# def get_weights(onnx_model_path, layer, kind):
#     inputs = get_inputs(onnx_model_path, layer)
#     print(inputs)

#     if kind == "x" or kind == "X":
#         print(inputs[0])
#         x = input_to_value(onnx_model_path, inputs[0])
#         return x

#     elif (kind == "w" or kind == "W") and len(inputs)>1:
#         w = input_to_value(onnx_model_path, inputs[1])
#         return w

#     elif (kind == "b" or kind == "B") and len(inputs)>2:
#         b = input_to_value(onnx_model_path, inputs[2])
#         return b

#     elif kind == "a" or kind == "A":
#         all_vals = []
#         # Iterar sobre las entradas de la capa
#         for input in inputs:
#             all_vals.append(input_to_value(onnx_model_path, input))

#         return all_vals
#     else:
#       print(f"The param {kind} is invalid")
    
#     # Agregar un mensaje de error si kind no es válido
#     raise ValueError(f"El parámetro 'kind' debe ser 'x', 'w', 'b' o 'a', pero se recibió: {kind}")
    


# if __name__ == "__main__":
#     get_layers()
#     get_layer_names()
#     input_to_value()
#     get_inputs()

