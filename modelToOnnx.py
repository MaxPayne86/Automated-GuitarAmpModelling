# Deps
# pip3 install torch==1.11.0+cu115 torchvision==0.12.0+cu115 torchaudio==0.11.0+cu115 torchtext==0.12.0 -f https://download.pytorch.org/whl/torch_stable.html
# pip3 install onnx==1.12.0
# pip3 install onnxruntime==1.12.0
import argparse
import json
import numpy as np

import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.networks as networks
import CoreAudioML.dataset as dataset

import torch

import onnx
import onnxruntime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config', '-l', help="Json config file describing the nn and the dataset", default='RNN-aidadsp-1')
    parser.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
    args = parser.parse_args()

    # Open config file
    config = args.config_location + "/" + args.load_config + ".json"
    with open(config) as json_file:
        config_data = json.load(json_file)
        device = config_data['device']

    results_path = "Results/" + device + "-" + args.load_config

    # Decide which model to use based on ESR results from
    # training
    stats = results_path + "/training_stats.json"
    with open(stats) as json_file:
        data = json.load(json_file)
        test_lossESR_final = data['test_lossESR_final']
        test_lossESR_best = data['test_lossESR_best']
        tmp = min(test_lossESR_final, test_lossESR_best)
        if tmp == test_lossESR_final:
            model_file = results_path + "/model.json"
        else:
            model_file = results_path + "/model_best.json"

    print("Using %s file" % model_file)

    with open(model_file) as json_file:
        model_data = json.load(json_file)

    # Load PyTorch model
    torch_model = networks.load_model(model_data)
    print(torch_model)

    # Generate input 3D tensor
    batch_size = 2048
    #x = np.random.uniform(-1.0, 1.0, (batch_size, input_size, 1))
    #x = np.float32(x) # See https://github.com/pytorch/pytorch/issues/2138
    # From wav file
    data = dataset.DataSet(data_dir='', extensions='')
    data.create_subset('data')
    data.load_file('in-gtr-2.wav', set_names='data')
    x = data.subsets['data'].data['data'][0]
    x = x[32000:(32000+batch_size), :, :].cpu().numpy()
    #x = np.float64(x) # Onnx: LSTM operator does not support double yet
    print("Input scalar type = %s" % (str(np.common_type(x))))
    x = torch.from_numpy(x)

    # PyTorch model prediction
    torch_model.skip = 0 # We need to assure there is no skip param involved for this test
    torch_model.reset_hidden()
    torch_model.eval() # Set the model to inference mode
    #torch_model.double() # Onnx: LSTM operator does not support double yet
    with torch.no_grad():
        torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,                # model being run
                    x,                            # model input (or a tuple for multiple inputs)
                    "output.onnx",                # where to save the model (can be a file or file-like object)
                    export_params=True,           # store the trained parameter weights inside the model file
                    opset_version=10,             # the ONNX version to export the model to
                    do_constant_folding=True,     # whether to execute constant folding for optimization
                    input_names = ['input'],      # the model's input names
                    output_names = ['output'],    # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'}, # variable length axes
                                'output' : {0 : 'batch_size'}})

    # Before verifying the model’s output with ONNX Runtime, we will check the ONNX model with ONNX’s API
    onnx_model = onnx.load("output.onnx")
    onnx.checker.check_model(onnx_model)

    # Compute onnx model output
    ort_session = onnxruntime.InferenceSession("output.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare ONNX Runtime and PyTorch results
    threshold = 1.0e-5;
    nErrs = 0;
    max_error = 0.0;
    max_error_index = 0
    for n in range(0, len(x)):
        err = abs(to_numpy(torch_out)[n] - ort_outs[0][n])
        if(err > threshold):
            #print("%f %f" % (to_numpy(torch_out)[n], ort_outs[0][n]))
            if err > max_error:
                max_error = err
                max_error_index = n
            #max_error = max(err, max_error)
            nErrs = nErrs + 1

    if(nErrs > 0):
        print("Onnx conversion FAILED, number of errors: %d" % nErrs)
        print("Maximum error: %.3f" % max_error)
        print("Maximum error index: %d" % max_error_index)
        print("Data: %.8f %.8f" % (to_numpy(torch_out)[max_error_index], ort_outs[0][max_error_index]))

    #np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
