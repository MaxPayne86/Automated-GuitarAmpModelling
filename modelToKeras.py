import argparse
import json
import numpy as np
from tensorflow import keras
from tensorflow import convert_to_tensor
from model_utils import save_model

import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.networks as networks
import CoreAudioML.dataset as dataset

import torch

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

    # Open model file
    with open(model_file) as json_file:
        model_data = json.load(json_file)
        try:
            unit_type = model_data['model_data']['unit_type']
            input_size = model_data['model_data']['input_size']
            skip = int(model_data['model_data']['skip']) # How many input elements are skipped
            hidden_size = model_data['model_data']['hidden_size']
            output_size = model_data['model_data']['output_size']
            bias_fl = bool(model_data['model_data']['bias_fl'])
            WVals = np.array(model_data['state_dict']['rec.weight_ih_l0'])
            UVals = np.array(model_data['state_dict']['rec.weight_hh_l0'])
            bias_ih_l0 =  np.array(model_data['state_dict']['rec.bias_ih_l0'])
            bias_hh_l0 = np.array(model_data['state_dict']['rec.bias_hh_l0'])
            lin_weight = np.array(model_data['state_dict']['lin.weight'])
            lin_bias = np.array(model_data['state_dict']['lin.bias'])
        except KeyError:
            print("Model file %s is corrupted" % (save_path + "/model.json"))

    # Load PyTorch model
    pytorch_m = networks.load_model(model_data)
    print(pytorch_m)

    # Construct TensorFlow model
    keras.backend.set_floatx('float32')
    model = keras.Sequential()
    input_layer = keras.layers.InputLayer(input_shape=(None, None, input_size))
    #model.add(input_layer)

    # Optionally inject error in weights to validate conversion
    #WVals = np.ones_like(WVals)

    if unit_type == "LSTM":
        lstm_weights = []

        WVals = np.transpose(WVals)
        print("Imported: p.shape(WVals)=%s" % (str(np.shape(WVals))))
        #WVals = WVals.reshape(hidden_size, 4, input_size)
        #for subm in WVals:
        #    subm[[0, 1]] = subm[[1, 0]]
        #WVals = WVals.reshape(input_size, 4*hidden_size)
        lstm_weights.append(WVals)

        UVals = np.transpose(UVals)
        print("Imported: p.shape(UVals)=%s" % (str(np.shape(UVals))))
        #UVals = UVals.reshape(hidden_size, 4, hidden_size)
        #for subm in UVals:
        #    subm[[0, 1]] = subm[[1, 0]]
        #UVals = UVals.reshape(hidden_size, 4*hidden_size)
        lstm_weights.append(UVals)

        BVals = (bias_ih_l0 + bias_hh_l0)
        print("Imported: p.shape(BVals)=%s" % (str(np.shape(BVals))))
        lstm_weights.append(BVals) # BVals is (hidden_size*4, )
        lstm_layer = keras.layers.LSTM(units=hidden_size, activation=None, return_sequences=True, use_bias=bias_fl, unit_forget_bias=False)
        lstm_layer.build(input_shape=(None, None, input_size))
        weights = lstm_layer.get_weights()
        for elem in weights:
            print("Desired: p.shape(elem)=%s" % (str(np.shape(elem))))
        lstm_layer.set_weights(lstm_weights)
        model.add(lstm_layer)
    elif unit_type == "GRU":
        gru_weights = []

        WVals = np.transpose(WVals)
        print("Imported: p.shape(WVals)=%s" % (str(np.shape(WVals))))
        WVals = WVals.reshape(hidden_size, 3, input_size)
        for subm in WVals:
            subm[[0, 1]] = subm[[1, 0]]
        WVals = WVals.reshape(input_size, 3*hidden_size)
        gru_weights.append(WVals)

        UVals = np.transpose(UVals)
        print("Imported: p.shape(UVals)=%s" % (str(np.shape(UVals))))
        UVals = UVals.reshape(hidden_size, 3, hidden_size)
        for subm in UVals:
            subm[[0, 1]] = subm[[1, 0]]
        UVals = UVals.reshape(hidden_size, 3*hidden_size)
        gru_weights.append(UVals)

        tmp = np.zeros((2, hidden_size*3))
        tmp[0] = bias_ih_l0
        tmp[1] = bias_hh_l0
        tmp = tmp.reshape(2, 3, -1)
        BVals = tmp[:, [1, 0, 2], :].reshape((2, -1))
        print("Imported: p.shape(BVals)=%s" % (str(np.shape(BVals))))
        gru_weights.append(BVals) # BVals is (2, hidden_size*3)
        gru_layer = keras.layers.GRU(units=hidden_size, return_sequences=True, use_bias=bias_fl)
        gru_layer.build(input_shape=(None, None, input_size))
        weights = gru_layer.get_weights()
        for elem in weights:
            print("Desired: p.shape(elem)=%s" % (str(np.shape(elem))))
        gru_layer.set_weights(gru_weights)
        model.add(gru_layer)
    else:
        print("Cannot parse unit_type = %s" % unit_type)
        exit(1)

    dense_weights = []
    dense_weights.append(lin_weight.reshape(hidden_size, 1)) # lin_weight is (1, hidden_size)
    dense_weights.append(lin_bias) # lin_bias is (1,)
    dense_layer = keras.layers.Dense(units=1, use_bias=True)
    dense_layer.build(input_shape=[None, None, hidden_size])
    dense_layer.set_weights(dense_weights)
    model.add(dense_layer)
    model.build(input_shape=(None, None, input_size))
    config = model.get_config()
    print(config)
    model.summary()

    # Compare PyTorch and Tensorflow models
    # Generate input 3D tensor
    n_samples = 2048
    #in_r1 = np.random.uniform(-1.0, 1.0, (n_samples, input_size, input_size))
    #in_r1 = np.float32(in_r1) # See https://github.com/pytorch/pytorch/issues/2138
    # From wav file
    data = dataset.DataSet(data_dir='', extensions='')
    data.create_subset('data')
    data.load_file('in-gtr-2.wav', set_names='data')
    in_r1 = data.subsets['data'].data['data'][0]
    in_r1 = in_r1[32000:(32000+n_samples), :, :].cpu().numpy()

    # PyTorch model prediction
    pytorch_m.skip = 0 # We need to assure there is no skip param involved for this test
    pytorch_m.reset_hidden()
    #pytorch_m.double() # See https://github.com/pytorch/pytorch/issues/2138
    with torch.no_grad():
        pred_r1_pytorch_m = pytorch_m(torch.from_numpy(in_r1))

    # Tensorflow model prediction
    tensorflow_m = model
    tensorflow_m.reset_states()
    pred_r1_tensorflow_m = tensorflow_m(convert_to_tensor(in_r1))

    offset = 0 # If desired, discard first results in loss calculation
    y_1 = pred_r1_pytorch_m.cpu().numpy()[offset:, 0, 0]
    y_2 = pred_r1_tensorflow_m.numpy()[offset:, 0, 0]

    loss = keras.losses.mean_squared_error(y_1, y_2)
    print("loss = \n%.8f\n" % np.sum(loss.numpy()))

    threshold = 1.0e-8;
    nErrs = 0;
    max_error = 0.0;
    for n in range(0, len(in_r1)):
        err = abs(y_1[n] - y_2[n])
        if(err > threshold):
            #print("%f %f" % (y_1[n], y_2[n]))
            max_error = max(err, max_error)
            nErrs = nErrs + 1

    if(nErrs > 0):
        print("Test FAILED, number of errors: %d" % nErrs)
        print("Maximum error: %.3f" % max_error)

    save_model(model, results_path + "/model_keras.json", keras.layers.InputLayer, skip=skip)

    # Save test data to be used with tests in RTNeural library
    np.savetxt("pytorch_x.csv", in_r1[:, 0], delimiter='', newline='\n')
    np.savetxt("pytorch_y.csv", y_1, delimiter='', newline='\n')
    np.savetxt("pytorch_tf_y.csv", y_2, delimiter='', newline='\n')
