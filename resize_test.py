import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import onnx
import onnx.optimizer
import onnx.helper
import onnxruntime
import onnx.shape_inference

def add_initializers_into_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    for x in model.graph.initializer:
        input_names = [x.name for x in model.graph.input]
        if x.name not in input_names:
            shape = onnx.TensorShapeProto()
            for dim in x.dims:
                shape.dim.extend([onnx.TensorShapeProto.Dimension(dim_value=dim)])
            model.graph.input.extend(
                [onnx.ValueInfoProto(name=x.name,
                                     type=onnx.TypeProto(tensor_type=onnx.TypeProto.Tensor(elem_type=x.data_type,
                                                                                           shape=shape)))])
    return model

def optimize(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    :param model: The onnx model.
    :return: The optimized onnx model.
    Before simplifying, use this method to generate value_info, which is used in `forward_all`
    After simplifying, use this method to fold constants generated in previous step into initializer,
    and eliminate unused constants.
    """

    # Due to a onnx bug, https://github.com/onnx/onnx/issues/2417, we need to add missing initializers into inputs
    input_num = len(model.graph.input)
    model = add_initializers_into_inputs(model)
    onnx.helper.strip_doc_string(model)
    model = onnx.optimizer.optimize(model, ['eliminate_deadend', 'eliminate_identity', 'eliminate_nop_dropout',
                                            'eliminate_nop_monotone_argmax', 'eliminate_nop_pad',
                                            'extract_constant_to_initializer', 'eliminate_unused_initializer',
                                            'eliminate_nop_transpose'],
                                    fixed_point=True)
    del model.graph.input[input_num:]
    onnx.checker.check_model(model)
    return model

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

class Resize_Test(nn.Module):
    def __init__(self):
        super(Resize_Test, self).__init__()


    def forward(self, x):
        #x = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=True)
        x = F.interpolate(x, size=(128, 128), mode='nearest')
        return x

def main():
    model = Resize_Test()
    batch_size=1
    dummy_input = torch.randn(batch_size, 3, 32, 32, device="cuda")
    model(dummy_input)
    print("exporting model")
    # Export the model
    torch.onnx.export(model,           # model being run
                      dummy_input,                         # model input (or a tuple for multiple inputs)
                      "model.onnx",   # where to save the model (can be a file or file-like object)
                      export_params=True,        # store the trained parameter weights inside the model file
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names = ['input'],   # the model's input names
                      output_names = ['output'], # the model's output names
                      #dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}},    # variable lenght axes
                      verbose=True,
                      opset_version=11,
                      keep_initializers_as_inputs=True)

    #####################################################################
    print("checking onnx model")
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    onnx.checker.check_graph(onnx_model.graph)
    #inferred_model = onnx.shape_inference.infer_shapes(onnx_model)
    #onnx.checker.check_model(inferred_model)
    print("onnx model is checked")

    ####################################################################
    print("optimizing onnx model")
    optimized_onnx_model = optimize(onnx_model)
    onnx.checker.check_model(optimized_onnx_model)
    onnx.checker.check_graph(optimized_onnx_model.graph)
    #inferred_model = onnx.shape_inference.infer_shapes(optimized_onnx_model)
    #onnx.checker.check_model(inferred_model)
    print("optimization done")
    onnx.save(optimized_onnx_model, 'optimized_model.onnx')

    #############################################

    ort_session = onnxruntime.InferenceSession("model.onnx")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)


    print("??????")

    ort_session = onnxruntime.InferenceSession('optimized_model.onnx')
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
    ort_outs = ort_session.run(None, ort_inputs)


    print("??????")




if __name__ == '__main__':
    main()