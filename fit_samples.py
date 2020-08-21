from pathlib import Path
from utils.config import PowerConfig
from utils.tf_models_seq2seq import seq2seqIWGAN

import fire
import numpy as np
import sys


def fit_samples(x: str, y: str, out: str = 'fit_samples_output'):
    '''
    Use this to train GAN with Autoencoder on data (either majority or minority) for sequence labels.

    Inputs:
    x: path to the training data file for GAN model (should be a saved numpy array)
    y: path to the training labels file for GAN model (should be a saved numpy array)
    out: path of the output directory (results will be saved here)
    data: data type to invoke the correct Config file, either 'Power' for power dataset or 'Sentiment' for sentiment dataset
    '''
    output_dir = Path(out)
    output_dir.mkdir(parents=False, exist_ok=True)

    x_train = np.load(x)  # shape: [n_samples, timesteps, n_features]
    y_train = np.load(y)  # shape: [n_samples, timesteps]

    config = PowerConfig()
    config.TIMESTEPS = 8
    config.DATA_DIM = 16

    iwgan = seq2seqIWGAN('Power', config)
    iwgan.iwganAeTrain(x_train, y_train, output_dir)


if __name__ == '__main__':
    fire.Fire(fit_samples)
