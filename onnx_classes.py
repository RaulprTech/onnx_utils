import onnx

class Layer:
    def __init__(self, onnx_model_path, layer_name):
        self.model = onnx.load(onnx_model_path)
        self.graph = self.model.graph
        self.layer_name = layer_name
        self.node = self._find_node_by_name(layer_name)

    def _find_node_by_name(self, name):
        for node in self.graph.node:
            if node.name == name:
                return node
        raise ValueError(f"No se encontró una capa con el nombre {name}")

    def get_inputs(self):
        return self.node.input[0]

    def get_outputs(self):
        return self.node.output

    def get_attribute(self, attr_name):
        for attr in self.node.attribute:
            if attr.name == attr_name:
                return attr
        return None
    



##################    CONVOLUTIONAL LAYER         ######################################################

class ConvLayer(Layer):
    def __init__(self, onnx_model_path, layer_name):
        super().__init__(onnx_model_path, layer_name)

    def get_weights(self):
        for initializer in self.graph.initializer:
            if initializer.name == self.node.input[1]:  # Suponiendo que el segundo input es el peso
                return initializer.name

    def get_bias(self):
        for initializer in self.graph.initializer:
            if initializer.name == self.node.input[2]:  # Suponiendo que el tercer input es el sesgo
                b = "" + initializer.name
                return b

    def get_kernel_size(self):
        return self.get_attribute('kernel_shape').ints

    def get_stride(self):
        return self.get_attribute('strides').ints

    def get_padding(self):
        return self.get_attribute('pads').ints

    def get_groups(self):
        return self.get_attribute('group').i
    



##################    RELU LAYER         #################################################################

class ReluLayer(Layer):
    def __init__(self, onnx_model_path, layer_name):
        super().__init__(onnx_model_path, layer_name)




##################    MAXPOOL LAYER        #############################################################


class MaxPoolLayer(Layer):
    def __init__(self, onnx_model_path, layer_name):
        super().__init__(onnx_model_path, layer_name)

    def get_kernel_size(self):
        return self.get_attribute('kernel_shape').ints

    def get_stride(self):
        return self.get_attribute('strides').ints

    def get_padding(self):
        return self.get_attribute('pads').ints
    


##################    ADD LAYER        ################################################################


class AddLayer(Layer):
    def __init__(self, onnx_path, layer_name):
        super().__init__(onnx_path, layer_name)



##################    GLOBALAVERALPOOL LAYER        ###################################################



class GlobalAveragePoolLayer(Layer):
    def __init__(self, onnx_path, layer_name):
        super().__init__(onnx_path, layer_name)



##################    FLATTEN LAYER        ###############################################################


class FlattenLayer(Layer):
    def __init__(self, onnx_path, layer_name):
        super().__init__(onnx_path, layer_name)



##################    GEMM LAYER        ###############################################################

class GemmLayer(Layer):
    def __init__(self, onnx_path, layer_name):
        super().__init__(onnx_path, layer_name)

    def get_weights(self):
        for initializer in self.graph.initializer:
            if initializer.name == self.node.input[1]:  # Suponiendo que el segundo input es el peso
                return initializer.name

    def get_bias(self):
        for initializer in self.graph.initializer:
            if initializer.name == self.node.input[2]:  # Suponiendo que el tercer input es el sesgo
                return initializer.name