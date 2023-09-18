# Voice-Conversion-Tests

This repository is dedicated to the implementation of specialized neural networks for voice conversion in the context of a master's thesis. 
Each directory within this repository contains code and resources tailored for training these neural networks.
To effectively train the models, it is essential to provide a training dataset with the following specific structure. Please note that the models exclusively accept audio files in the WAV format.

### Dataset structure
* speaker1
    * --sample1.wav
    * --sample2.wav
    * ...
* speaker2
    * --sample1.wav
    * --sample2.wav
    * ...
* ...

### Directory contents:
* Cyclegan-basic - contains code for early tests for CycleGAN networks for voice conversion
* INCEPTION-VC-VAE-2 - source code for INCEPTION-VC2 model training and inference
* INCEPTION-VC-VAE - contains code for basic INCEPTION-VC model training and inference
* Transformer-VC-Phase - source code for Transformer-VC training and inference with phase processing
* Transformer-VC-modified - contains code for Transformer-VC training and inference with reduced/extended channel size
* Transformer-VC - source code for basic Transformer-VC training and inference
* Transformer-VGG-for-PHASE - contains code for Transformer-VC training and inference with phase processing by VGG model
* Variational-Autencoder-VC - source code for basic VAE-VC model training and inference
* app - contains source code for running web application for voice conversion based on VAE-VC model

To run test, specific Python environment is required with venv installed. 
To create a virtualenv use the following command in specific directory:
`python -m venv <environment-name>`

To activate created environment in Windows operating system, navigate to `<specific-directory>/<environment-name>/Scripts` and type `Activate.bat`.
Once environment run, navigate to`Voice-Conversion-Tests` repository and type `pip install -r requirements.txt`.

To run model training, navigate to model directory from `Directory contents` and run `python main.py`. Change of `config.json` file may be required.
Once training is finished, to run inference on model run `python generate_sound.py`. File `generate_sound.py` change may be also required to set specific paths.
