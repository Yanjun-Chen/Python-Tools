import torch
import torch.nn as nn
import numpy as np
import importlib
import argparse
from mobilenet_v1 import MobilenetV1

class onnx_generator(object):

    def __init__(self, model, filename='mbn_v1_block.onnx', input_shape=[1, 32, 224, 224]):
        self.model = model
        self.filename = filename
        self.input = torch.randn(*input_shape, requires_grad=True)
    
    def save_onnx(self):
        print('exporting to onnx')
        self.model.eval()
        self.model.cpu()

        with torch.no_grad():
            torch.onnx.export(self.model, self.input, self.filename,
                                    keep_initializers_as_inputs=True)

    def chech_onnx(self):
        import onnx
        onnx_model = onnx.load(self.filename)
        onnx.checker.check_model(onnx_model)

    def set_input_shape(self, *input_shape):
        self.input = torch.randn(*input_shape, requires_grad=True)

    def torch(self):
        self.model.eval()
        return self.to_numpy(self.model(self.input))

    def onnx(self):
        import onnxruntime
        ort_session = onnxruntime.InferenceSession(self.filename)
        ort_inputs = {ort_session.get_inputs()[0].name: self.to_numpy(self.input)}
        ort_outs = ort_session.run(None, ort_inputs)
        return ort_outs[0]
    
    def test_onnx(self):
        np.testing.assert_allclose(self.torch(), self.onnx(), rtol=1e-03, atol=1e-05)
        print('Successfully saved to' + self.filename)

    @staticmethod
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='M-Pytorch Onnx Generator')
    parser.add_argument('--model_filename', type=str, default='MobilenetV1', help='Model filename')
    parser.add_argument('--class_name', type=str, default='Model', help='class or function to return Model')
    
    args = parser.parse_args()

    """get model"""

    model = MobilenetV1()
    print(model)

    """generate onnx"""
    onnx_filename = args.model_filename + ".onnx"
    oc = onnx_generator(model, filename=onnx_filename, input_shape=( 1, 3, 1024, 1024))
    oc.save_onnx()
    oc.chech_onnx()
    oc.test_onnx()