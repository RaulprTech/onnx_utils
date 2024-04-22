import torch
import onnx
import os
from torchvision import models

def convert_pth_to_onnx(model_name):
    # Verificar si estamos en Google Colab
    try:
        import google.colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    # Ruta al modelo en la carpeta "model"
    model_path = f"../models/{model_name}.pth"

    if not os.path.isfile(f"../{model_name}.onnx"):
        if IN_COLAB:
            # Subir el modelo a Google Colab
            uploaded = files.upload()

            # Cargar el modelo pre-entrenado de PyTorch desde la carpeta "model" en Colab
            model_path = list(uploaded.keys())[0]  # Obtiene el nombre del archivo subido
            model = torch.load(model_path, map_location=torch.device('cpu'))  # Cargar el modelo en CPU
        else:
            # Verificar si el modelo especificado existe en la carpeta "model"
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"No se encontró el archivo {model_name}.pth en la carpeta 'models'")

            model = torch.load(model_path, map_location=torch.device('cpu'))  # Cargar el modelo en CPU

            # Verificar si el modelo es un diccionario (OrderedDict)
            if isinstance(model, dict) and 'state_dict' in model:
                # Reconstruir el modelo desde el diccionario
                model = models.resnet18(pretrained=False)  # Reemplaza YourModelClass con la clase de tu modelo
                model.load_state_dict(model['state_dict'])
            elif not isinstance(model, torch.nn.Module):
                print(f"Tipo de modelo cargado: {type(model)}")
                raise TypeError("El archivo cargado no parece ser un modelo de PyTorch válido.")

        model.eval()  # Establecer el modelo en modo de evaluación

        # Definir una entrada de ejemplo (tensor)
        dummy_input = torch.randn(1, 3, 224, 224)  # Ejemplo de tensor de entrada

        # Exportar el modelo a ONNX
        onnx_path = f"{os.path.splitext(model_name)[0]}.onnx"
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11)

        print(f"Modelo exportado correctamente como '{onnx_path}' usando el archivo {model_path}")


# Llamar a la función y especificar el nombre del archivo .pth
convert_pth_to_onnx("resnet18")