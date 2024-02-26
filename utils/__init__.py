import numpy as np
from einops import rearrange
import torch
import yaml
import os
from pathlib import Path
import tempfile
import requests
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn import metrics
import torch.nn.functional as F


def get_root_dir():
    return Path(__file__).parent.parent


def compute_downsample_rate(input_length: int, n_fft: int, downsampled_width: int):
    return (
        round(input_length / (np.log2(n_fft) - 1) / downsampled_width)
        if input_length >= downsampled_width
        else 1
    )


def load_yaml_param_settings(yaml_fname: str):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, "r")
    config = yaml.load(stream, Loader=yaml.FullLoader)
    return config


def time_to_timefreq(x, n_fft: int, C: int):
    """
    x: (B, C, L)
    """
    window = torch.hann_window(n_fft).to(x.device)
    x = rearrange(x, "b c l -> (b c) l")
    x = torch.stft(x, n_fft, normalized=False, return_complex=True, window=window)
    x = torch.view_as_real(x)  # (B, N, T, 2); 2: (real, imag)
    x = rearrange(x, "(b c) n t z -> b (c z) n t ", c=C)  # z=2 (real, imag)
    return x  # (B, C, H, W)


def timefreq_to_time(x, n_fft: int, C: int):
    window = torch.hann_window(n_fft).to(x.device)
    x = rearrange(x, "b (c z) n t -> (b c) n t z", c=C).contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(x, n_fft, normalized=False, return_complex=False, window=window)
    x = rearrange(x, "(b c) l -> b c l", c=C)
    return x


def quantize(z, vq_model, transpose_channel_length_axes=False):
    input_dim = len(z.shape) - 2
    if input_dim == 2:
        h, w = z.shape[2:]
        z = rearrange(z, "b c h w -> b (h w) c")
        z_q, indices, vq_loss, perplexity = vq_model(z)
        z_q = rearrange(z_q, "b (h w) c -> b c h w", h=h, w=w)
    elif input_dim == 1:
        if transpose_channel_length_axes:
            z = rearrange(z, "b c l -> b (l) c")
        z_q, indices, vq_loss, perplexity = vq_model(z)
        if transpose_channel_length_axes:
            z_q = rearrange(z_q, "b (l) c -> b c l")
    else:
        raise ValueError
    return z_q, indices, vq_loss, perplexity


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False


def ssl_config_filename(config, model_type):
    """
    Generates a filename for the SSL configuration based on the provided settings.
    """
    SSL_config = config["SSL"]

    stage1_method, stage1_weight = (
        SSL_config["stage1_method"],
        SSL_config["stage1_weight"],
    )
    stage2_method, stage2_weight = (
        SSL_config["stage2_method"],
        SSL_config["stage2_weight"],
    )

    stage1_text = ""
    stage2_text = ""

    if stage1_method != "":
        stage1_text = f"{stage1_method}_{stage1_weight}_"

    # Only MAGE model has SSL on stage2
    if stage2_method != "" and model_type == "MAGE":
        stage2_text = f"_{stage2_method}_{stage2_weight}"

    filename_parts = [
        part for part in [stage1_text, model_type, stage2_text] if part
    ]  # Filters out empty strings

    return "".join(filename_parts)


def save_model(models_dict: dict, dirname="saved_models", id: str = ""):
    """
    :param models_dict: {'model_name': model, ...}
    """
    try:
        if not os.path.isdir(get_root_dir().joinpath(dirname)):
            os.mkdir(get_root_dir().joinpath(dirname))

        id_ = id[:]
        if id != "":
            id_ = "-" + id_
        for model_name, model in models_dict.items():
            torch.save(
                model.state_dict(),
                get_root_dir().joinpath(dirname, model_name + id_ + ".ckpt"),
            )
    except PermissionError:
        # dirname = tempfile.mkdtemp()
        dirname = tempfile.gettempdir()
        print(
            f"\nThe trained model is saved in the following temporary dirname due to some permission error: {dirname}.\n"
        )

        id_ = id[:]
        if id != "":
            id_ = "-" + id_
        for model_name, model in models_dict.items():
            torch.save(
                model.state_dict(),
                get_root_dir().joinpath(dirname, model_name + id_ + ".ckpt"),
            )


def encode_data(dataloader, encoder, n_fft, vq_model=None, cuda=True):
    """
    Function to encode the data using the encoder and optionally the quantizer.
    It encodes to continous latent variables by default (vq_model=False).
    ---
    returns
    """
    z_list = []  # List to hold all the encoded representations
    y_list = []  # List to hold all the labels/targets

    # Iterate over the entire dataloader
    for batch in dataloader:
        x, y = batch  # Unpack the batch.

        # Perform the encoding
        if cuda:
            x = x.cuda()
        C = x.shape[1]
        xf = time_to_timefreq(x, n_fft, C).to(
            x.device
        )  # Convert time domain to frequency domain
        z = encoder(xf)  # Encode the input

        if vq_model is not None:
            z, _, _, _ = quantize(z, vq_model)
        # Convert the tensors to lists and append to z_list and y_list
        z_list.extend(z.cpu().detach().tolist())
        y_list.extend(
            y.cpu().detach().tolist()
        )  # Make sure to detach y and move to CPU as well

    # Convert lists of lists to 2D tensors
    z_encoded = torch.tensor(z_list)
    ys = torch.tensor(y_list)
    if cuda:
        z_encoded = z_encoded.cuda()
        ys = ys.cuda()

    return z_encoded, ys
