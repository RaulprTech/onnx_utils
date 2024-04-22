import onnx

class Layer:
    """
    Base class for handling ONNX layer information.
    """

    def __init__(self, onnx_model_path, layer_name):
        """
        Initialize the Layer object.

        Parameters:
        onnx_model_path (str): The path to the ONNX model file.
        layer_name (str): The name of the layer in the ONNX model.
        """
        self.model = onnx.load(onnx_model_path)
        self.graph = self.model.graph
        self.layer_name = layer_name
        self.node = self._find_node_by_name(layer_name)

    def _find_node_by_name(self, name):
        """
        Find a node in the ONNX graph by its name.

        Parameters:
        name (str): The name of the node.

        Returns:
        onnx.NodeProto: The node corresponding to the given name.
        """
        for node in self.graph.node:
            if node.name == name:
                return node
        raise ValueError(f"No layer found with the name {name}")

    def get_inputs(self):
        """
        Retrieve the input names of the layer.

        Returns:
        str: The name of the input tensor.
        """
        return self.node.input[0]

    def get_outputs(self):
        """
        Retrieve the output names of the layer.

        Returns:
        list: A list of output tensor names.
        """
        return self.node.output

    def get_attribute(self, attr_name):
        """
        Retrieve an attribute by its name from the layer node.

        Parameters:
        attr_name (str): The name of the attribute.

        Returns:
        onnx.AttributeProto or None: The attribute value or None if not found.
        """
        for attr in self.node.attribute:
            if attr.name == attr_name:
                return attr
        return None


##################    CONVOLUTIONAL LAYER         ####################################################################


class ConvLayer(Layer):
    """
    Class for handling convolutional layers in ONNX models.
    """

    def __init__(self, onnx_model_path, layer_name):
        super().__init__(onnx_model_path, layer_name)

    def get_weights(self):
        """
        Retrieve the weights of the convolutional layer.

        Returns:
        str: The name of the weight tensor.
        """
        for initializer in self.graph.initializer:
            if initializer.name == self.node.input[1]:
                return initializer.name

    def get_bias(self):
        """
        Retrieve the bias of the convolutional layer.

        Returns:
        str: The name of the bias tensor.
        """
        for initializer in self.graph.initializer:
            if initializer.name == self.node.input[2]:
                return initializer.name

    def get_kernel_size(self):
        """
        Retrieve the kernel size of the convolutional layer.

        Returns:
        list: A list of integers representing the kernel size.
        """
        return self.get_attribute('kernel_shape').ints

    def get_stride(self):
        """
        Retrieve the stride of the convolutional layer.

        Returns:
        list: A list of integers representing the stride.
        """
        return self.get_attribute('strides').ints

    def get_padding(self):
        """
        Retrieve the padding of the convolutional layer.

        Returns:
        list: A list of integers representing the padding.
        """
        return self.get_attribute('pads').ints

    def get_groups(self):
        """
        Retrieve the groups of the convolutional layer.

        Returns:
        int: The number of groups.
        """
        return self.get_attribute('group').i


##################    RELU LAYER         ####################################################################

class ReluLayer(Layer):
    """
    Class for handling ReLU activation layers in ONNX models.
    """

    def __init__(self, onnx_model_path, layer_name):
        """
        Initialize the ReluLayer object.

        Parameters:
        onnx_model_path (str): The path to the ONNX model file.
        layer_name (str): The name of the ReLU layer in the ONNX model.
        """
        super().__init__(onnx_model_path, layer_name)


##################    MAXPOOL LAYER        #################################################################
class MaxPoolLayer(Layer):
    """
    Class for handling MaxPool layers in ONNX models.
    """

    def __init__(self, onnx_model_path, layer_name):
        """
        Initialize the MaxPoolLayer object.

        Parameters:
        onnx_model_path (str): The path to the ONNX model file.
        layer_name (str): The name of the MaxPool layer in the ONNX model.
        """
        super().__init__(onnx_model_path, layer_name)

    def get_kernel_size(self):
        """
        Retrieve the kernel size of the MaxPool layer.

        Returns:
        list: A list of integers representing the kernel size.
        """
        return self.get_attribute('kernel_shape').ints

    def get_stride(self):
        """
        Retrieve the stride of the MaxPool layer.

        Returns:
        list: A list of integers representing the stride.
        """
        return self.get_attribute('strides').ints

    def get_padding(self):
        """
        Retrieve the padding of the MaxPool layer.

        Returns:
        list: A list of integers representing the padding.
        """
        return self.get_attribute('pads').ints


##################    ADD LAYER        #####################################################################
class AddLayer(Layer):
    """
    Class for handling Add layers in ONNX models.
    """

    def __init__(self, onnx_path, layer_name):
        """
        Initialize the AddLayer object.

        Parameters:
        onnx_model_path (str): The path to the ONNX model file.
        layer_name (str): The name of the Add layer in the ONNX model.
        """
        super().__init__(onnx_path, layer_name)


##################    GLOBALAVERAGEPOOL LAYER        #######################################################
class GlobalAveragePoolLayer(Layer):
    """
    Class for handling GlobalAveragePool layers in ONNX models.
    """

    def __init__(self, onnx_path, layer_name):
        """
        Initialize the GlobalAveragePoolLayer object.

        Parameters:
        onnx_model_path (str): The path to the ONNX model file.
        layer_name (str): The name of the GlobalAveragePool layer in the ONNX model.
        """
        super().__init__(onnx_path, layer_name)


##################    FLATTEN LAYER        ###############################################################
class FlattenLayer(Layer):
    """
    Class for handling Flatten layers in ONNX models.
    """

    def __init__(self, onnx_path, layer_name):
        """
        Initialize the FlattenLayer object.

        Parameters:
        onnx_model_path (str): The path to the ONNX model file.
        layer_name (str): The name of the Flatten layer in the ONNX model.
        """
        super().__init__(onnx_path, layer_name)


##################    GEMM LAYER        ################################################################
class GemmLayer(Layer):
    """
    Class for handling Gemm (fully connected) layers in ONNX models.
    """

    def __init__(self, onnx_path, layer_name):
        """
        Initialize the GemmLayer object.

        Parameters:
        onnx_model_path (str): The path to the ONNX model file.
        layer_name (str): The name of the Gemm layer in the ONNX model.
        """
        super().__init__(onnx_path, layer_name)

    def get_weights(self):
        """
        Retrieve the weights of the Gemm layer.

        Returns:
        str: The name of the weight tensor.
        """
        for initializer in self.graph.initializer:
            if initializer.name == self.node.input[1]:
                return initializer.name

    def get_bias(self):
        """
        Retrieve the bias of the Gemm layer.

        Returns:
        str: The name of the bias tensor.
        """
        for initializer in self.graph.initializer:
            if initializer.name == self.node.input[2]:
                return initializer.name
