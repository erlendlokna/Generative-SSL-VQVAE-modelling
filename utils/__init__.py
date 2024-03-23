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
import os


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


def shape_match(x, xhat):
    if x.shape[-1] > xhat.shape[-1]:
        x = x[..., : xhat.shape[-1]]
    elif x.shape[-1] < xhat.shape[-1]:
        xhat = xhat[..., : x.shape[-1]]
    else:
        x, xhat = x, xhat

    return x, xhat


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


def timefreq_to_time(x, n_fft: int, C: int, original_length=None):
    window = torch.hann_window(n_fft).to(x.device)
    x = rearrange(x, "b (c z) n t -> (b c) n t z", c=C).contiguous()
    x = torch.view_as_complex(x)
    x = torch.istft(x, n_fft, normalized=False, return_complex=False, window=window)

    if original_length is not None:
        x = x[..., :original_length]

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


def model_filename(config, model_type):
    model_types = {"encoder", "decoder", "vqmodel", "maskgit", "byolmaskgit"}

    assert model_type in model_types, "Non valid model type"

    stage1_ssl_method = config["SSL"]["stage1_method"]
    stage2_ssl_method = config["SSL"]["stage2_method"]

    decorr = config["VQVAE"]["orthogonal_reg_weight"] > 0

    filename_parts = []

    if decorr:
        filename_parts.append("decorr_")

    if stage1_ssl_method:
        filename_parts.append(f"{stage1_ssl_method}_")

    filename_parts.append(model_type)

    if stage2_ssl_method and model_type == "byolmaskgit":
        filename_parts.append(f"_{stage2_ssl_method}")

    return "".join(part for part in filename_parts if part)


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


def encode_data(
    dataloader,
    encoder,
    n_fft,
    vq_model=None,
    avg_pooling=False,
    num_tokens=32,
    device="cuda",
):
    """
    Function to encode the data using the encoder and optionally the quantizer.
    It encodes to continous latent variables by default (vq_model=False).
    ---
    returns
    """
    z_list = []  # List to hold all the encoded representations
    y_list = []  # List to hold all the labels/targets

    # Iterate over the entire dataloader
    counts = torch.zeros(num_tokens, device=device)

    for batch in dataloader:
        x, y = batch  # Unpack the batch.
        if len(x) == 2:
            x = x[0]  # discard the potential augmented view
        x = x.to(device)
        # Perform the encoding
        C = x.shape[1]
        xf = time_to_timefreq(x, n_fft, C).to(
            x.device
        )  # Convert time domain to frequency domain
        z = encoder(xf)  # Encode the input

        if vq_model is not None:
            z, s, _, _ = quantize(z, vq_model)
            counts += torch.bincount(s.flatten(), minlength=32)

        # Convert the tensors to lists and append to z_list and y_list
        z_list.extend(z.cpu().detach().tolist())
        y_list.extend(
            y.cpu().detach().tolist()
        )  # Make sure to detach y and move to CPU as well

    # Convert lists of lists to 2D tensors
    z_encoded = torch.tensor(z_list, device=device)
    ys = torch.tensor(y_list, device=device)

    if avg_pooling:
        z_encoded = F.adaptive_avg_pool2d(z_encoded, (1, 1)).squeeze(-1).squeeze(-1)

    return z_encoded, ys, counts
