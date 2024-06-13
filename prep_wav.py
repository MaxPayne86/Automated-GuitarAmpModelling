# Creating a valid dataset for the trainining script
# using wav files provided by user.
# Example of usage:
# python3 prep_wav.py -f input.wav target.wav -l "RNN-aidadsp-1"
# the files will be splitted 70% 15% 15%
# and used to populate train test val.
# This is done to have different data for training, testing and validation phase
# according with the paper.
# If the user provide multiple wav files pairs e.g. guitar_in.wav guitar_tg.wav bass_in.wav bass_tg.wav
# then 70% of guitar_in.wav is concatenated to 70% of bass_in.wav and so on.
# If the user provide guitar and bass files of the same length, then the same amount
# of guitar and bass recorded material will be used for network training.

import CoreAudioML.miscfuncs as miscfuncs
from CoreAudioML.dataset import audio_splitter
from scipy.io import wavfile
import numpy as np
import argparse
import os
import csv
from colab_functions import save_wav, parse_csv, peak, align_target
import librosa


def nonConditionedWavParse(args):
    print("")
    print("Using config file %s" % args.load_config)
    file_name = ""
    configs = miscfuncs.json_load(args.load_config, args.config_location)
    try:
        file_name = configs['file_name']
    except KeyError:
        print("Error: config file doesn't have file_name defined")
        exit(1)
    try:
        blip_offset = configs['blip_offset']
    except KeyError:
        print("Warning: config file doesn't have blip_offset defined")
        blip_offset = 0
    try:
        blip_window = configs['blip_window']
    except KeyError:
        print("Warning: config file doesn't have blip_window defined")
        blip_window = None
    try:
        blip_locations = configs['blip_locations']
    except KeyError:
        print("Warning: config file doesn't have blip_locations defined")
        blip_locations = None
    try:
        blip_window = configs['blip_window']
    except KeyError:
        print("Warning: config file doesn't have blip_window defined")
        blip_window = None
    if args.denoise:
        from colab_functions import denoise
    try:
        samplerate = int(configs['samplerate'])
        print("Using samplerate = %.2f" % samplerate)
    except KeyError:
        print("Error: config file doesn't have samplerate defined")
        samplerate = None
        exit(1)

    train_in = np.ndarray([0], dtype=np.float32)
    train_tg = np.ndarray([0], dtype=np.float32)
    test_in = np.ndarray([0], dtype=np.float32)
    test_tg = np.ndarray([0], dtype=np.float32)
    val_in = np.ndarray([0], dtype=np.float32)
    val_tg = np.ndarray([0], dtype=np.float32)

    for in_file, tg_file in zip(args.files[::2], args.files[1::2]):
        #print("Input file name: %s" % in_file)
        x_all, in_rate = librosa.load(in_file, sr=None, mono=True)
        #print("Target file name: %s" % tg_file)
        y_all, tg_rate = librosa.load(tg_file, sr=None, mono=True)

        if in_rate != samplerate or tg_rate != samplerate:
            print("Input samplerate = %.2f, desired samplerate = %.2f" % (in_rate, samplerate))
            print("Target samplerate = %.2f, desired samplerate = %.2f" % (tg_rate, samplerate))
            print("Resampling files the desired samplerate")
            x_all = librosa.resample(x_all, orig_sr=in_rate, target_sr=samplerate)
            y_all = librosa.resample(y_all, orig_sr=tg_rate, target_sr=samplerate)

        # Auto-align
        if blip_locations and blip_window:
            y_all_aligned = align_target(tg_data=y_all, blip_offset=blip_offset, blip_locations=tuple(blip_locations), blip_window=blip_window)
            if y_all_aligned is not None:
                y_all = y_all_aligned
            else:
                print("Error! Was not able to calculate alignment delay!")
                exit(1)
        else:
            print("Warning! Auto-alignment disabled...")

        if(x_all.size != y_all.size):
            min_size = min(x_all.size, y_all.size)
            #print("Warning! Length for audio files\n\r  %s\n\r  %s\n\rdoes not match, setting both to %d [samples]" % (in_file, tg_file, min_size))
            x_all = np.resize(x_all, min_size)
            y_all = np.resize(y_all, min_size)

        # Noise reduction, using CPU
        if args.denoise:
            y_all = denoise(waveform=y_all)

        # Normalization
        if args.norm:
            in_lvl = peak(x_all)
            y_all = peak(y_all, in_lvl)

        # Default to 70% 15% 15% split
        if not args.csv_file:
            splitted_x = audio_splitter(x_all, [0.70, 0.15, 0.15])
            splitted_y = audio_splitter(y_all, [0.70, 0.15, 0.15])
        else:
            # Csv file to be placed in same dir of input file
            [train_bounds, test_bounds, val_bounds] = parse_csv(os.path.dirname(in_file) + '/' + args.csv_file)
            splitted_x = [np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32)]
            splitted_y = [np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32)]
            for bounds in train_bounds:
                splitted_x[0] = np.append(splitted_x[0], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[0] = np.append(splitted_y[0], audio_splitter(y_all, bounds, unit='s'))
            for bounds in test_bounds:
                splitted_x[1] = np.append(splitted_x[1], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[1] = np.append(splitted_y[1], audio_splitter(y_all, bounds, unit='s'))
            for bounds in val_bounds:
                splitted_x[2] = np.append(splitted_x[2], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[2] = np.append(splitted_y[2], audio_splitter(y_all, bounds, unit='s'))

        train_in = np.append(train_in, splitted_x[0])
        train_tg = np.append(train_tg, splitted_y[0])
        test_in = np.append(test_in, splitted_x[1])
        test_tg = np.append(test_tg, splitted_y[1])
        val_in = np.append(val_in, splitted_x[2])
        val_tg = np.append(val_tg, splitted_y[2])

    print("Saving processed wav files into dataset")

    save_wav("Data/train/" + file_name + "-input.wav", samplerate, train_in)
    save_wav("Data/train/" + file_name + "-target.wav", samplerate, train_tg)

    save_wav("Data/test/" + file_name + "-input.wav", samplerate, test_in)
    save_wav("Data/test/" + file_name + "-target.wav", samplerate, test_tg)

    save_wav("Data/val/" + file_name + "-input.wav", samplerate, val_in)
    save_wav("Data/val/" + file_name + "-target.wav", samplerate, val_tg)

def conditionedWavParse(args):
    print("Using config file %s" % args.load_config)
    file_name = ""
    configs = miscfuncs.json_load(args.load_config, args.config_location)
    try:
        file_name = configs['file_name']
    except KeyError:
        print("Error: config file doesn't have file_name defined")
        exit(1)
    try:
        blip_offset = configs['blip_offset']
    except KeyError:
        print("Warning: config file doesn't have blip_offset defined")
        blip_offset = 0
    try:
        blip_locations = configs['blip_locations']
    except KeyError:
        print("Warning: config file doesn't have blip_locations defined")
        blip_locations = None
    try:
        blip_window = configs['blip_window']
    except KeyError:
        print("Warning: config file doesn't have blip_window defined")
        blip_window = None
    if args.denoise:
        from colab_functions import denoise
    try:
        samplerate = int(configs['samplerate'])
        print("Using samplerate = %.2f" % samplerate)
    except KeyError:
        print("Error: config file doesn't have samplerate defined")
        samplerate = None
        exit(1)

    params = configs['params']

    counter = 0
    main_rate = 0
    all_train_in = np.array([[]]*(1 + params['n']), dtype=np.float32) # 1 channel for in audio, n channels per parameters
    all_train_tg = np.array([[]], dtype=np.float32) # 1 channels of all (out audio)
    all_test_in = np.array([[]]*(1 + params['n']), dtype=np.float32) # 1 channel for in audio, n channels per parameters
    all_test_tg = np.array([[]], dtype=np.float32) # 1 channels of all (out audio)
    all_val_in = np.array([[]]*(1 + params['n']), dtype=np.float32) # 1 channel for in audio, n channels per parameters
    all_val_tg = np.array([[]], dtype=np.float32) # 1 channels of all (out audio)

    for entry in params['datasets']:
        #print("Input file name: %s" % entry['input'])
        x_all, in_rate = librosa.load(entry['input'], sr=None, mono=True)
        #print("Target file name: %s" % entry['target'])
        y_all, tg_rate = librosa.load(entry['target'], sr=None, mono=True)

        if in_rate != samplerate or tg_rate != samplerate:
            print("Input samplerate = %.2f, desired samplerate = %.2f" % (in_rate, samplerate))
            print("Target samplerate = %.2f, desired samplerate = %.2f" % (tg_rate, samplerate))
            print("Resampling files the desired samplerate")
            x_all = librosa.resample(x_all, orig_sr=in_rate, target_sr=samplerate)
            y_all = librosa.resample(y_all, orig_sr=tg_rate, target_sr=samplerate)

        # Auto-align
        if blip_locations and blip_window:
            y_all_aligned = align_target(tg_data=y_all, blip_offset=blip_offset, blip_locations=tuple(blip_locations), blip_window=blip_window)
            if y_all_aligned is not None:
                y_all = y_all_aligned
            else:
                print("Error! Was not able to calculate alignment delay!")
                exit(1)
        else:
            print("Warning! Auto-alignment disabled...")

        if(x_all.size != y_all.size):
            min_size = min(x_all.size, y_all.size)
            #print("Warning! Length for audio files\n\r  %s\n\r  %s\n\rdoes not match, setting both to %d [samples]" % (entry['input'], entry['target'], min_size))
            x_all = np.resize(x_all, min_size)
            y_all = np.resize(y_all, min_size)

        # Noise reduction, using CPU
        if args.denoise:
            y_all = denoise(waveform=y_all)

        # Normalization
        if args.norm:
            in_lvl = peak(x_all)
            y_all = peak(y_all, in_lvl)

        # Default to 70% 15% 15% split
        if not args.csv_file:
            splitted_x = audio_splitter(x_all, [0.70, 0.15, 0.15])
            splitted_y = audio_splitter(y_all, [0.70, 0.15, 0.15])
        else:
            # Csv file to be placed in same dir of input file
            [train_bounds, test_bounds, val_bounds] = parse_csv(os.path.dirname(entry['input']) + '/' + args.csv_file)
            splitted_x = [np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32)]
            splitted_y = [np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32), np.ndarray([0], dtype=np.float32)]
            for bounds in train_bounds:
                splitted_x[0] = np.append(splitted_x[0], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[0] = np.append(splitted_y[0], audio_splitter(y_all, bounds, unit='s'))
            for bounds in test_bounds:
                splitted_x[1] = np.append(splitted_x[1], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[1] = np.append(splitted_y[1], audio_splitter(y_all, bounds, unit='s'))
            for bounds in val_bounds:
                splitted_x[2] = np.append(splitted_x[2], audio_splitter(x_all, bounds, unit='s'))
                splitted_y[2] = np.append(splitted_y[2], audio_splitter(y_all, bounds, unit='s'))

        # Initialize lists to handle the number of parameters
        params_train = []
        params_val = []
        params_test = []

        # Create a list of np arrays of the parameter values
        for val in entry["params"]:
            # Create the parameter arrays
            params_train.append(np.array([val]*len(splitted_x[0]), dtype=np.float32))
            params_test.append(np.array([val]*len(splitted_x[1]), dtype=np.float32))
            params_val.append(np.array([val]*len(splitted_x[2]), dtype=np.float32))

        # Convert the lists to numpy arrays
        params_train = np.array(params_train, dtype=np.float32)
        params_val = np.array(params_val, dtype=np.float32)
        params_test = np.array(params_test, dtype=np.float32)

        # Append the audio and paramters to the full data sets
        all_train_in = np.append(all_train_in, np.append([splitted_x[0]], params_train, axis=0), axis = 1)
        all_train_tg = np.append(all_train_tg, splitted_y[0])
        all_test_in = np.append(all_test_in, np.append([splitted_x[1]], params_test, axis=0), axis = 1)
        all_test_tg = np.append(all_test_tg, splitted_y[1])
        all_val_in = np.append(all_val_in, np.append([splitted_x[2]], params_val, axis=0), axis = 1)
        all_val_tg = np.append(all_val_tg, splitted_y[2])

    print("Saving processed wav files into dataset")

    save_wav("Data/train/" + file_name + "-input.wav", samplerate, all_train_in.T, flatten=False)
    save_wav("Data/test/" + file_name + "-input.wav", samplerate, all_test_in.T, flatten=False)
    save_wav("Data/val/" + file_name + "-input.wav", samplerate, all_val_in.T, flatten=False)

    save_wav("Data/train/" + file_name + "-target.wav", samplerate, all_train_tg)
    save_wav("Data/test/" + file_name + "-target.wav", samplerate, all_test_tg)
    save_wav("Data/val/" + file_name + "-target.wav", samplerate, all_val_tg)

def main(args):
    if args.files:
        if (len(args.files) % 2) and not args.parameterize:
            print("Error: you should provide arguments in pairs see help")
            exit(1)

    if args.parameterize is True:
        conditionedWavParse(args)
    else:
        nonConditionedWavParse(args)

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '-f', nargs='+', help='provide input target files in pairs e.g. guitar_in.wav guitar_tg.wav bass_in.wav bass_tg.wav')
    parser.add_argument('--load_config', '-l',
                  help="File path, to a JSON config file, arguments listed in the config file will replace the defaults", default='RNN-aidadsp-1')
    parser.add_argument('--csv_file', '-csv', default='', help='Name of csv file containing split bounds info')
    parser.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
    parser.add_argument('--parameterize', '-p', action=argparse.BooleanOptionalAction, default=False, help='Perform parameterized training')
    parser.add_argument('--norm', '-n', action=argparse.BooleanOptionalAction, default=False, help='Perform normalization of target tracks so that they will match the volume of the input tracks')
    parser.add_argument('--denoise', '-dn', action=argparse.BooleanOptionalAction, default=False, help='Perform noise removal on target tracks leveraging noisereduce package')

    args = parser.parse_args()
    main(args)
