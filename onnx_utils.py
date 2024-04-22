import torch
import onnx
import os
from torchvision import models

def get_layers(onnx_model_path):
    # Cargar el modelo ONNX
    model = onnx.load(onnx_model_path)

    # Obtener el grafo del modelo
    graph = model.graph

    # Inicializar un diccionario para almacenar las capas y su frecuencia
    layers = {}

    # Iterar sobre las operaciones del grafo
    for node in graph.node:
        # Obtener el nombre de la capa y agregarlo al diccionario
        layer_name = node.op_type
        if layer_name in layers:
            layers[layer_name] += 1
        else:
            layers[layer_name] = 1

    return layers


def get_layer_names(onnx_model_path):
    # Cargar el modelo ONNX
    model = onnx.load(onnx_model_path)

    # Obtener el grafo del modelo
    graph = model.graph

    # Inicializar una lista para almacenar los nombres de las capas
    layer_names = []

    # Iterar sobre las operaciones del grafo
    for node in graph.node:
        # Agregar el nombre de la operación a la lista de nombres de las capas
        layer_names.append(node.name)

    return layer_names


# Busca el valor correspondiente al nombre del input dado
def input_to_value(onnx_model_path, input_name):
    # Cargar el modelo ONNX
    model = onnx.load(onnx_model_path)

    # Obtener el grafo del modelo
    graph = model.graph

    # Inicializar una variable para almacenar los valores de los pesos
    value = None

    # Iterar sobre los inicializadores del grafo
    for init_tensor in graph.initializer:
        if input_name in init_tensor.name:
            # Convertir el tensor de inicialización a un array de numpy
            value = onnx.numpy_helper.to_array(init_tensor)
            return value
    print("Not found value")



def get_inputs(onnx_model_path, layer_name):
    # Cargar el modelo ONNX
    model = onnx.load(onnx_model_path)

    # Obtener el grafo del modelo
    graph = model.graph

    # Iterar sobre las operaciones del grafo
    for node in graph.node:
        # Verificar si el nombre de la operación corresponde a la capa
        if node.name == layer_name:
            # print(node.input)
            inputs = node.input
            return inputs
        


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
    


if __name__ == "__main__":
    get_layers()
    get_layer_names()
    input_to_value()
    get_inputs()

