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
from colab_functions import save_wav, peak
from colab_functions import convert_csv_to_info, get_info_samplerate, scale_info, save_csv, parse_info
from nam_utils import _DataInfo, _calibrate_delay_v_all
import librosa

def WavParse(args):
    print("")
    print("Using config file %s" % args.load_config)
    configs = miscfuncs.json_load(args.load_config, args.config_location)
    file_name, samplerate, csv_file = None, None, None
    try:
        file_name = configs['file_name']
        samplerate = int(configs['samplerate'])
        csv_file = configs['params']['csv']
        params = configs['params']
    except KeyError as e:
        print(f"Config file is missing the key: {e}")
        exit(1)
    print("Using samplerate = %.2f" % samplerate)
    print("Using csv file: %s" % csv_file)
    info = convert_csv_to_info(csv_file)
    info_samplerate = get_info_samplerate(info)
    if info_samplerate != samplerate:
        print("Csv file samplerate = %.2f, desired samplerate = %.2f" % (info_samplerate, samplerate))
        print("Resampling csv file to the desired samplerate")
        info = scale_info(info, scale_factor=float(samplerate)/float(info_samplerate))
        csv_file = csv_file.replace(".csv", f"-{samplerate}.csv")
        save_csv(info, csv_file)

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
        blip_locations = info['blips']
        print(f"Blip locations: {blip_locations}")
        compensation = 250 # [ms]
        compensation_samples = int((compensation / 1000.0) * samplerate)
        first_blips_start = info[blips][0] - compensation_samples
        t_blips = (info[blips][1] + compensation_samples) - first_blips_start

        # Populate _DataInfo
        data_info = _DataInfo(
            major_version=-1,
            rate=samplerate,
            t_blips=t_blips,
            first_blips_start=first_blips_start,
            t_validate=0,
            train_start=0,
            validation_start=0,
            noise_interval=(0, 6_000),
            blip_locations=(tuple(blip_locations),),
        )

        # Calibrate the delay in the input-output pair based on blips
        delay = _calibrate_delay_v_all(data_info, y_all)
        print(f"Calibrated delay: {delay} samples")

        # Delay compensation
        if delay < 0:
            y_all = np.concatenate(np.zeros(abs(delay)), y_all).astype(np.float32)
        elif delay >= 0:
            y_all = y_all[delay:].astype(np.float32)
        else:
            print("Error in calculating delay!")
            raise ValueError

        # Check if the audio files have the same length
        if(x_all.size != y_all.size):
            min_size = min(x_all.size, y_all.size)
            print("Warning! Length for audio files\n\r  %s\n\r  %s\n\rdoes not match, setting both to %d [samples]" % (entry['input'], entry['target'], min_size))
            x_all = np.resize(x_all, min_size)
            y_all = np.resize(y_all, min_size)

        # Noise reduction, using CPU
        if args.denoise:
            y_all = denoise(waveform=y_all, noise_locations=info['noise'], samplerate=samplerate)

        # Normalization
        if args.norm:
            in_lvl = peak(x_all)
            y_all = peak(y_all, in_lvl)

        [train_bounds, test_bounds, val_bounds] = parse_info(csv_file)
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

    print("Done!")

def main(args):
    WavParse(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_config', '-l',
                  help="File path, to a JSON config file, arguments listed in the config file will replace the defaults", default='RNN-aidadsp-1')
    parser.add_argument('--config_location', '-cl', default='Configs', help='Location of the "Configs" directory')
    parser.add_argument('--norm', '-n', action=argparse.BooleanOptionalAction, default=False, help='Perform normalization of target tracks so that they will match the volume of the input tracks')
    parser.add_argument('--denoise', '-dn', action=argparse.BooleanOptionalAction, default=False, help='Perform noise removal on target tracks leveraging noisereduce package')

    args = parser.parse_args()
    main(args)
