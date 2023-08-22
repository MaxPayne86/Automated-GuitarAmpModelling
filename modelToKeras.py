import argparse
import json
import numpy as np
from tensorflow import keras
from model_utils import save_model
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', '-lm', help="Json model as exported from Pytorch", default='')
    parser.add_argument('--load_config', '-l', help="Json config file describing the nn and the dataset", default='LSTM-12')
    parser.add_argument('--results_path', '-rp', help="Directory of the resulting model", default='None')
    parser.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
    parser.add_argument('--aidax', '-ax', default='', help='The output file extension will be .aidax')
    args = parser.parse_args()

    if not args.load_model:
        # Open config file
        config = args.config_location + "/" + args.load_config + ".json"
        with open(config) as json_file:
            config_data = json.load(json_file)
            device = config_data['device']
            unit_type = config_data['unit_type']
            hidden_size = config_data['hidden_size']
            skip = config_data['skip_con']
            metadata = config_data['metadata']

        if args.results_path == "None":
            results_path = "Results/" + device + '_' + unit_type + '-' + str(hidden_size) + '-' + str(skip)
        else:
            results_path = args.results_path

        # Decide which model to use based on ESR results from
        # training, extract input/output batch consequently
        stats = results_path + "/training_stats.json"
        with open(stats) as json_file:
            data = json.load(json_file)
            test_lossESR_final = data['test_lossESR_final']
            test_lossESR_best = data['test_lossESR_best']
            esr = min(test_lossESR_final, test_lossESR_best)
            input_batch = data['input_batch']
            if esr == test_lossESR_final:
                model = results_path + "/model.json"
                output_batch = data['output_batch_final']
            else:
                model = results_path + "/model_best.json"
                output_batch = data['output_batch_best']
    else:
        model = args.load_model
        results_path = os.path.dirname(args.load_model)

    #print("Using %s file" % model)

    # Open model file
    with open(model) as json_file:
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
            print("Model file %s is corrupted" % (model))

    # Construct TensorFlow model
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(None, input_size)))

    if unit_type == "LSTM":
        lstm_weights = []
        #print("-- Adding %s..." % "rec.weight_ih_l0 / WVals")
        lstm_weights.append(np.transpose(WVals))
        #print("-- Adding %s..." % "rec.weight_hh_l0 / UVals")
        lstm_weights.append(np.transpose(UVals))
        #print("-- Adding %s..." % "rec.bias_ih_l0, rec.bias_hh_l0 / BVals")
        BVals = (bias_ih_l0 + bias_hh_l0)
        lstm_weights.append(BVals) # BVals is (hidden_size*4, )
        lstm_layer = keras.layers.LSTM(hidden_size, activation=None, weights=lstm_weights, return_sequences=True, recurrent_activation=None, use_bias=bias_fl, unit_forget_bias=False)
        model.add(lstm_layer)
    elif unit_type == "GRU":
        gru_weights = []
        #print("-- Adding %s..." % "rec.weight_ih_l0 / WVals")
        WVals = np.transpose(WVals)
        i = 0
        for row in WVals:
            row = np.concatenate((row[hidden_size:hidden_size*2], row[0:hidden_size], row[hidden_size*2:]))
            WVals[i] = row
            i = i + 1
        gru_weights.append(WVals)
        #print("-- Adding %s..." % "rec.weight_hh_l0 / UVals")
        UVals = np.transpose(UVals)
        i = 0
        for row in UVals:
            row = np.concatenate((row[hidden_size:hidden_size*2], row[0:hidden_size], row[hidden_size*2:]))
            UVals[i] = row
            i = i + 1
        gru_weights.append(UVals)
        #print("-- Adding %s..." % "rec.bias_ih_l0, rec.bias_hh_l0 / BVals")
        tmp = np.zeros((2, hidden_size*3))
        tmp[0] = np.concatenate((bias_ih_l0[hidden_size:hidden_size*2], bias_ih_l0[0:hidden_size], bias_ih_l0[hidden_size*2:]))
        tmp[1] = np.concatenate((bias_hh_l0[hidden_size:hidden_size*2], bias_hh_l0[0:hidden_size], bias_hh_l0[hidden_size*2:]))
        BVals = tmp
        gru_weights.append(BVals) # BVals is (2, hidden_size*3)
        gru_layer = keras.layers.GRU(hidden_size, activation=None, weights=gru_weights, return_sequences=True, recurrent_activation=None, use_bias=bias_fl)
        model.add(gru_layer)
    else:
        print("Cannot parse unit_type = %s" % unit_type)
        exit(1)

    dense_weights = []
    dense_weights.append(lin_weight.reshape(hidden_size, 1)) # lin_weight is (1, hidden_size)
    dense_weights.append(lin_bias) # lin_bias is (1,)
    dense_layer = keras.layers.Dense(1, weights=dense_weights, kernel_initializer="orthogonal", bias_initializer='random_normal')
    model.add(dense_layer)

    if not args.load_model:
        metadata['esr'] = esr
    else:
        input_batch = None
        output_batch = None
        metadata = None

    if args.aidax:
        output_model_path = results_path + "/model_keras.aidax"
    else:
        output_model_path = results_path + "/model_keras.json"

    # Using save_model method from model_utils module from RTNeural project
    save_model(model, output_model_path, keras.layers.InputLayer, skip=skip, input_batch=input_batch, output_batch=output_batch, metadata=metadata, verbose=False)
