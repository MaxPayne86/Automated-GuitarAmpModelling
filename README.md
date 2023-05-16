# Automated-Guitar Amplifier Modelling

This repository contains neural network training scripts and trained models of guitar amplifiers and distortion pedals. The 'Results' directory contains some example recurrent neural network models trained to emulate the ht-1 amplifier and Big Muff Pi fuzz pedal, these models are described in this [conference paper](https://www.dafx.de/paper-archive/2019/DAFx2019_paper_43.pdf)

## Aida DSP contributions

### What we implemented / improved

- a way to export models generated here in a format compatible with [RTNeural](https://github.com/jatinchowdhury18/RTNeural)
- a way to customize the dataset with split bounds that are expressed with a csv file (see prep_wav.py)
- A-Weighting FIR filter coefficients to be used in the loss function pre-emphasis filter see PERCEPTUAL LOSS FUNCTION FOR NEURAL MODELLING OF AUDIO SYSTEMS
- a way to generate an ESR vs time audio track, since ESR is pretty much always NOT evenly distributed across the test audio track
- a Jupyter script .ipynb to perform training with Google Colab
- a Docker container with CUDA support to run the very same .ipynb locally, see aidadsp/pytorch:latest
- a [multi-platform Desktop plugin](https://github.com/AidaDSP/AIDA-X) to run the models generated here on your favourite DAW
- a [lv2 plugin](https://github.com/AidaDSP/aidadsp-lv2) that is a stripped down version of the previous plugin to be used on embedded linux devices such as RPi, MOD Dwarf, AIDA DSP OS, etc
- we now perform model validation during runtime, inside the plugins. This allow us to freely develop the plugin and bring new backend features while at the same time being able to immediately spot regressions as well as improvements in the code that runs the network

### How to use (Google Colab)

Visit the link [here](https://colab.research.google.com/github/AidaDSP/Automated-GuitarAmpModelling/blob/aidadsp_devel/AIDA_X_Model_Trainer.ipynb) to perform the training online

### How to use (Local)

You need to use our docker container on your machine:  aidadsp/pytorch:latest

#### Clone this repository

```
git clone https://github.com/AidaDSP/Automated-GuitarAmpModelling.git
cd Automated-GuitarAmpModelling
git checkout aidadsp_devel
git submodule update --init --recursive
```

From now on, we will refer to <THIS_DIR> as the path where you launched above commands. 

#### Local use, Jupyter Notebook

```
docker run --gpus all -v <THIS_DIR>:/workdir:rw -w /workdir -p 8888:8888 --env JUPYTER_TOKEN=aidadsp -it aidadsp/pytorch:latest
```

Jupyter Web UI will be accessible in your browser at http://127.0.0.1:8888, then simply enter the password (aidadsp)

#### Local use, Bash shell

```
docker run --gpus all -v <THIS_DIR>:/workdir:rw -w /workdir -it --entrypoint /bin/bash aidadsp/pytorch:latest
```

now in the docker container bash shell you can run commands. **Firstly**, you should identify a Configs file that you want to use, tweak it to suits your needs, then you can provide input.wav and target.wav in the following way:

```
python prep_wav.py -f "/path/to/input.wav" ""/path/to/target.wav" -l LSTM-12.json -n
```

where -n would apply normalization (advised). If you want to control which portions of the file are used for train, val, test you can
pass -csv arg just open the script to understand how it works.

**Secondly** you can perform the training passing always the same Configs file where all the infos are stored

```
python dist_model_recnet.py -l LSTM-12.json -slen 24000 --seed 39 -lm 0
```

where -slen would setup the chunk length used during training, here ```24000*1/48000 = 500 [ms]``` considering 48000 Hz sampling rate. For the other params, please open the script.

**Finally** you want to convert the model with all the weights exported from pytorch into a format that is suitable for usage with RTNeural library, which in turns is the engine used by our plugins: [AIDA-X](https://github.com/AidaDSP/AIDA-X) and [aidadsp-lv2](https://github.com/AidaDSP/aidadsp-lv2). You can do it in the following way:

```
python modelToKeras.py -l LSTM-12.json
```

this file would output a file named model_keras.json, which will then be the model to be loaded into the plugins.

#### Explore some hidden features

- we now store metadata inside Config file, so if you save a structure like the following inside your Config file, it will be copied in the final model file

```
metadata = {
    "name": "My Awesome Device Name",
    "samplerate": "48000",
    "source":"Analog / Digital Hw / Plugin / You Name It",
    "style": "Clean / Breakup / High Gain / You Name It",
    "based": "What is based on, aka the name of the device",
    "author": "Who did this model",
    "dataset": "Describe the dataset used to train",
    "license": "CC BY-NC-ND 4.0 / Whatever LICENSE fits your needs"
}
```

- we perform input / target audio track time alignment. For this to work you have to provide into the Config file the following parameters

```
"blip_locations": [12_000, 36_000],
"blip_window": 48_000,
```

the values above are just for reference and are the one used by the blips or counting clicks at the beginning of the track in the current NAM dataset [v1_1_1.wav](https://drive.google.com/file/d/1v2xFXeQ9W2Ks05XrqsMCs2viQcKPAwBk/view?usp=share_link)

- we can express region markers in the input/target tracks that will be used to tag sections of the Dataset that will end up into train, val and test respectively via csv file. To activate csv file just invoke prep_wav.py with -csv option. Then you need to place a .csv file named the same as the input track, so if I have /My/Foo/Dir/input.wav I will create /My/Foo/Dir/input.csv. The csv file needs to be outlined in the following way:

```
#,Name,Start,End,Length,Color
R1,train,50000,7760000,7657222,FF0000
R2,testval,7760000,8592000,832000,00FFFF
```

the example above, which is working for the current NAM dataset, I'm telling that the R1 region will goes into train (RGB: FF0000) while the region R2 will be used for test and validation at the same time (RGBs: 00FF00 + 0000FF = 00FFFF).

- you can not only calculate ESR on an arbitrary audio track for a given model, but you can also obtain an ESR vs time audio track, to be imported in your DAW, which will let you better troubleshoot your model. With the following command:

```
python proc_audio.py -l LSTM-12.json -i /path/to/input.wav -t /path/to/target.wav -o ./proc.wav
```

the script will generate a proc_ESR.wav containing the ESR vs time audio track

### Prerequisites to run the docker container

#### Windows Users

Please follow instructions here https://docs.docker.com/desktop/install/windows-install

#### Mac Users

Please follow instructions here https://docs.docker.com/desktop/install/mac-install

#### Linux Users

Please follow instructions below

##### NVIDIA drivers

```
dpkg -l | grep nvidia-driver
ii  nvidia-driver-510                          510.47.03-0ubuntu0.20.04.1            amd64        NVIDIA driver metapackage

dpkg -S /usr/lib/i386-linux-gnu/libcuda.so
libnvidia-compute-510:i386: /usr/lib/i386-linux-gnu/libcuda.so
```

##### NVIDIA Container Toolkit

```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && distribution="ubuntu20.04" \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

now you can run containers with gpu support

### Dataset

#### NAM Dataset

Since I've received a bunch of request from the NAM community, I leave some infos here. Since the
NAM models at the moment are not compatible with the inference engine used by rt-neural-generic (RTNeural), you can't
use them with our plugin directly. But you can still use our training script and the NAM Dataset, so that you will be able
to use the amplifiers that you are using on NAM with our plugin. In the end, training is 10mins on a Laptop with CUDA.

To do so, I'll leave a reference to NAM Dataset [v1_1_1.wav](https://drive.google.com/file/d/1v2xFXeQ9W2Ks05XrqsMCs2viQcKPAwBk/view?usp=share_link)

## Using this repository
It is possible to use this repository to train your own models. To model a different distortion pedal or amplifier, a dataset recorded from your target device is required, example datasets recorded from the ht1 and Big Muff Pi are contained in the 'Data' directory.

### Cloning this repository

To create a working local copy of this repository, use the following command:

git clone --recurse-submodules https://github.com/Alec-Wright/NeuralGuitarAmpModelling

### Python Environment

Using this repository requires a python environment with the 'pytorch', 'scipy', 'tensorboard' and 'numpy' packages installed.
Additionally this repository uses the 'CoreAudioML' package, which is included as a submodule. Cloining the repo as described in 'Cloning this repository' ensures the CoreAudioML package is also downloaded.

### Processing Audio

The 'proc_audio.py' script loads a neural network model and uses it to process some audio, then saving the processed audio. This is also a good way to check if your python environment is setup correctly. Running the script with no arguments:

python proc_audio.py

will use the default arguments, the script will load the 'model_best.json' file from the directory 'Results/ht1-ht11/' and use it to process the audio file 'Data/test/ht1-input.wav', then save the output audio as 'output.wav'
Different arguments can be used as follows

python proc_audio.py 'path/to/input_audio.wav' 'output_filename.wav' 'Results/path/to/model_best.json'

### Training Script

the 'dist_model_recnet.py' script was used to train the example models in the 'Results' directory. At the top of the file the 'argparser' contains a description of all the training script arguments, as well as their default values. To train a model using the default arguments, simply run the model from the command line as follows:

python dist_model_recnet.py

note that you must run this command from a python environment that has the libraries described in 'Python Environment' installed. To use different arguments in the training script you can change the default arguments directly in 'dist_model_recnet.py', or you can direct the 'dist_model_recnet.py' script to look for a config file that contains different arguments, for example by running the script using the following command:

python dist_model_recnet.py -l "ht11.json"

Where in this case the script will look for the file ht11.json in the the 'Configs' directory. To create custom config files, the ht11.json file provided can be edited in any text editor.

During training, the script will save some data to a folder in the Results directory. These are, the lowest loss achieved on the validation set so far in 'bestvloss.txt', as well as a copy of that model 'model_best.json', and the audio created by that model 'best_val_out.wav'. The neural network at the end of the most recent training epoch is also saved, as 'model.json'. When training is complete the test dataset is processed, and the audio produced and the test loss is also saved to the same directory.

A trained model contained in one of the 'model.json' or 'model_best.json' files can be loaded, see the 'proc_audio.py' script for an example of how this is done.

### Determinism

If determinism is desired, `dist_model_recnet.py` provides an option to seed all of the random number generators used at once. However, if NVIDIA CUDA is used, you must also handle the non-deterministic behavior of CUDA for RNN calculations as is described in the [Rev8 Release Notes](https://docs.nvidia.com/deeplearning/cudnn/release-notes/rel_8.html). The user can eliminate the non-deterministic behavior of cuDNN RNN and multi-head attention APIs, by setting a single buffer size in the CUBLAS_WORKSPACE_CONFIG environmental variable, for example, :16:8 or :4096:2
```
CUBLAS_WORKSPACE_CONFIG=:4096:2
```
or
```
CUBLAS_WORKSPACE_CONFIG=:16:8
```
Note: if you're in google colab, the following goes into a cell
```
!export CUBLAS_WORKSPACE_CONFIG=:4096:2
```

### Tensorboard
The `./dist_model_recnet.py` has been implemented with PyTorch's Tensorboard hooks. To see the data, run:
```
tensorboard --logdir ./TensorboardData
```

### Feedback

This repository is still a work in progress, and I welcome your feedback! Either by raising an Issue or in the 'Discussions' tab
