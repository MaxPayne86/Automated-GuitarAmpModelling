# Creating a valid dataset for the trainining script
# using wav files provided by user.
# The script will split the audio files into training, testing and validation sets using the csv file provided by the user,
# which contains the timestamps for the train, test and val intervals in samples.
# The csv file syntax is following Reaper region markers export csv format.

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
    file_name, samplerate, csv = None, None, None
    try:
        file_name = configs['file_name']
        samplerate = int(configs['samplerate'])
        csv = configs['params']['csv']
        params = configs['params']
    except KeyError as e:
        print(f"Config file is missing the key: {e}")
        exit(1)
    print("Using samplerate = %.2f" % samplerate)
    print("Using csv file: %s" % csv)
    info = convert_csv_to_info(csv)
    info_samplerate = get_info_samplerate(info)

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

        # Check audio samplerate vs csv samplerate
        if in_rate != info_samplerate:
            print("Error: audio file samplerate = %.2f, csv samplerate = %.2f" % (in_rate, info_samplerate))
            exit(1)

        # Auto-align
        blip_locations = info['blips'][0]
        print(f"Blip locations: {blip_locations}")
        compensation = 250 # [ms]
        compensation_samples = int((compensation / 1000.0) * in_rate)
        first_blips_start = info['blips'][0][0] - compensation_samples
        t_blips = (info['blips'][0][1] + compensation_samples) - first_blips_start
        noise_interval = (first_blips_start, first_blips_start + int(compensation_samples / 4))
        print(f"Noise interval: {noise_interval}")
        # Noise interval needs to be included in the region before the first blip
        assert noise_interval[0] >= first_blips_start and noise_interval[1] <= info['blips'][0][0], "Noise interval is not included in the blips interval!"

        # Populate _DataInfo
        data_info = _DataInfo(
            major_version=-1,
            rate=int(in_rate),
            t_blips=t_blips,
            first_blips_start=first_blips_start,
            t_validate=0,
            train_start=0,
            validation_start=0,
            noise_interval=noise_interval,
            blip_locations=(tuple(blip_locations),),
        )

        # Calibrate the delay in the input-output pair based on blips
        delay = _calibrate_delay_v_all(data_info, y_all)

        # Delay compensation
        if delay < 0:
            y_all = np.concatenate([np.zeros(abs(delay)), y_all]).astype(np.float32)
        elif delay >= 0:
            y_all = y_all[delay:].astype(np.float32)
        else:
            print("Error in calculating delay!")
            raise ValueError

        print(f"Calibrated delay: {delay} samples, {delay / in_rate * 1000} ms")

        # Check if the audio files have the same length
        if(x_all.size != y_all.size):
            min_size = min(x_all.size, y_all.size)
            print("Warning! Length for audio files\n\r  %s\n\r  %s\n\rdoes not match, setting both to %d [samples]" % (entry['input'], entry['target'], min_size))
            x_all = np.resize(x_all, min_size)
            y_all = np.resize(y_all, min_size)

        # Noise reduction, using CPU
        if args.denoise:
            y_all = denoise(waveform=y_all, noise_locations=info['noise'][0], samplerate=int(in_rate))

        # Normalization
        if args.norm:
            in_lvl = peak(x_all)
            y_all = peak(y_all, in_lvl)

        # Check if resample is needed
        if info_samplerate != samplerate:
            print("Csv file samplerate = %.2f, desired samplerate = %.2f" % (info_samplerate, samplerate))
            print("Resampling csv file to the desired samplerate")
            info = scale_info(info, scale_factor=float(samplerate)/float(info_samplerate))
            csv = csv.replace(".csv", f"-{samplerate}.csv")
            save_csv(csv, info)
            print(f"Saved resampled csv file to {csv}")

        if in_rate != samplerate or tg_rate != samplerate:
            print("Input samplerate = %.2f, desired samplerate = %.2f" % (in_rate, samplerate))
            print("Target samplerate = %.2f, desired samplerate = %.2f" % (tg_rate, samplerate))
            print("Resampling files to the desired samplerate")
            x_all = librosa.resample(x_all, orig_sr=in_rate, target_sr=samplerate)
            y_all = librosa.resample(y_all, orig_sr=tg_rate, target_sr=samplerate)

        [train_bounds, test_bounds, val_bounds] = parse_info(info)
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

        if "params" not in entry:
            args.parameterized = False
        else:
            args.parameterized = True

        if args.parameterized:
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

            # Append the audio and parameters to the full data sets
            all_train_in = np.append(all_train_in, np.append([splitted_x[0]], params_train, axis=0), axis = 1)
            all_train_tg = np.append(all_train_tg, splitted_y[0])
            all_test_in = np.append(all_test_in, np.append([splitted_x[1]], params_test, axis=0), axis = 1)
            all_test_tg = np.append(all_test_tg, splitted_y[1])
            all_val_in = np.append(all_val_in, np.append([splitted_x[2]], params_val, axis=0), axis = 1)
            all_val_tg = np.append(all_val_tg, splitted_y[2])
        else:
            all_train_in = np.append(all_train_in, splitted_x[0])
            all_train_tg = np.append(all_train_tg, splitted_y[0])
            all_test_in = np.append(all_test_in, splitted_x[1])
            all_test_tg = np.append(all_test_tg, splitted_y[1])
            all_val_in = np.append(all_val_in, splitted_x[2])
            all_val_tg = np.append(all_val_tg, splitted_y[2])

    print("Saving processed wav files into dataset")

    if args.parameterized:
        save_wav("Data/train/" + file_name + "-input.wav", samplerate, all_train_in.T, flatten=False)
        save_wav("Data/test/" + file_name + "-input.wav", samplerate, all_test_in.T, flatten=False)
        save_wav("Data/val/" + file_name + "-input.wav", samplerate, all_val_in.T, flatten=False)
    else:
        save_wav("Data/train/" + file_name + "-input.wav", samplerate, all_train_in)
        save_wav("Data/test/" + file_name + "-input.wav", samplerate, all_test_in)
        save_wav("Data/val/" + file_name + "-input.wav", samplerate, all_val_in)

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
