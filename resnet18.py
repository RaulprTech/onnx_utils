import torch
import onnx
import os
from torchvision import models


def convert_pth_to_onnx(model_name):
    """
    Convert a PyTorch model (.pth) to ONNX format (.onnx).

    Parameters:
    model_name (str): The name of the PyTorch model file (without extension).

    Raises:
    FileNotFoundError: If the specified .pth file is not found.
    TypeError: If the loaded file is not a valid PyTorch model.

    Returns:
    None
    """
    
    # Check if running in Google Colab
    try:
        import google.colab # Import only if running in Google Colab
        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    # Path to the model in the "models" folder
    model_path = f"models/{model_name}.pth"

    # Check if .onnx file already exists
    if not os.path.isfile(f"models/{model_name}.onnx"):
        if IN_COLAB:
            # Upload the model to Google Colab
            uploaded = files.upload()

            # Load the pretrained PyTorch model from the "models" folder in Colab
            model_path = list(uploaded.keys())[0]  # Get the uploaded file name
            model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model on CPU
        else:
            # Check if the specified model exists in the "models" folder
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"No such file: {model_name}.pth in the 'models' folder")

            model = torch.load(model_path, map_location=torch.device('cpu'))  # Load the model on CPU

            # Check if the model is a dictionary (OrderedDict)
            if isinstance(model, dict) and 'state_dict' in model:
                # Reconstruct the model from the dictionary
                model = models.resnet18(pretrained=False)  # Replace YourModelClass with your model's class
                model.load_state_dict(model['state_dict'])
            elif not isinstance(model, torch.nn.Module):
                print(f"Loaded model type: {type(model)}")
                raise TypeError("The loaded file does not appear to be a valid PyTorch model.")

        model.eval()  # Set the model to evaluation mode

        # Define a sample input (tensor)
        dummy_input = torch.randn(1, 3, 224, 224)  # Example input tensor

        # Export the model to ONNX
        onnx_path = f"{os.path.splitext(model_name)[0]}.onnx"
        torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=11)

        print(f"Model successfully exported as '{onnx_path}' using the file {model_path}")

# Call the function and specify the .pth file name
convert_pth_to_onnx("resnet18")
