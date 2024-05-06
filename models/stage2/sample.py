"""
`python sample.py`

sample
    1) unconditional sampling
    2) class-conditional sampling
"""

import os
from argparse import ArgumentParser

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.stage2.maskgit import MaskGIT
from models.stage2.full_embedding_maskgit import Full_Embedding_MaskGIT
from preprocessing.data_pipeline import build_data_pipeline
from utils import get_root_dir, load_yaml_param_settings, model_filename
from tqdm import tqdm


@torch.no_grad()
def unconditional_sample(
    generative_model,
    n_samples: int,
    device,
    class_index=None,
    batch_size=256,
    return_representations=False,
    guidance_scale=1.0,
):
    """
    Generative model: MaskGIT or MAGE
    """
    n_iters = n_samples // batch_size
    is_residual_batch = False
    if n_samples % batch_size > 0:
        n_iters += 1
        is_residual_batch = True

    x_new = []
    quantize_new = []
    sample_callback = generative_model.iterative_decoding

    for i in tqdm(range(n_iters)):
        # print(f"it: {i+1}/{n_iters}")
        b = batch_size
        if (i + 1 == n_iters) and is_residual_batch:
            b = n_samples - ((n_iters - 1) * batch_size)
        embed_ind = sample_callback(
            num=b, device=device, class_index=class_index, guidance_scale=guidance_scale
        )

        if return_representations:
            x, quantize = generative_model.decode_token_ind_to_timeseries(
                embed_ind, True
            )
            (
                x,
                quantize,
            ) = (
                x.cpu(),
                quantize.cpu(),
            )

            quantize_new.append(quantize)
        else:
            x = generative_model.decode_token_ind_to_timeseries(embed_ind).cpu()

        x_new.append(x)

    x_new = torch.cat(x_new)

    if return_representations:
        quantize_new = torch.cat(quantize_new)
        quantize_new = torch.cat(quantize_new)
        return x_new, quantize_new
    else:
        return x_new


@torch.no_grad()
def conditional_sample(
    generative_model,
    n_samples: int,
    device,
    class_index: int,
    batch_size=256,
    return_representations=False,
    guidance_scale=1.0,
):
    """
    class_index: starting from 0. If there are two classes, then `class_index` ∈ {0, 1}.
    """
    generative_model.transformer.p_unconditional = 0.0

    if return_representations:
        x_new, quantize_new = unconditional_sample(
            generative_model,
            n_samples,
            device,
            class_index,
            batch_size,
            guidance_scale=guidance_scale,
        )
        return x_new, quantize_new
    else:
        x_new = unconditional_sample(
            generative_model,
            n_samples,
            device,
            class_index,
            batch_size,
            guidance_scale=guidance_scale,
        )
        return x_new


def plot_generated_samples(x_new, title: str, max_len=20):
    # x_new: (n_samples, c, length); c=1 (univariate)

    n_samples = x_new.shape[0]
    if n_samples > max_len:
        print(f"`n_samples` is too large for visualization. The maximum is {max_len}.")
        return None

    try:
        fig, axes = plt.subplots(1 * n_samples, 1, figsize=(3.5, 1.7 * n_samples))
        alpha = 0.5
        if n_samples > 1:
            for i, ax in enumerate(axes):
                ax.set_title(title)
                ax.plot(x_new[i, 0, :])
        else:
            axes.set_title(title)
            axes.plot(x_new[0, 0, :])

        plt.tight_layout()
        plt.show()
    except ValueError:
        print(f"`n_samples` is too large for visualization. The maximum is {max_len}.")


def save_generated_samples(x_new: np.ndarray, save: bool, fname: str = None):
    if save:
        fname = "generated_samples.npy" if not fname else fname
        with open(get_root_dir().joinpath("generated_samples", fname), "wb") as f:
            np.save(f, x_new)
            print(
                "numpy matrix of the generated samples are saved as `generated_samples/generated_samples.npy`."
            )


class Sampler(object):
    def __init__(self, real_train_data_loader, config, device, ssl_stage1=False):
        self.config = config
        self.device = device
        self.guidance_scale = self.config["class_guidance"]["guidance_scale"]

        # build MaskGIT
        # train_data_loader = build_data_pipeline(self.config, 'train')
        n_classes = len(np.unique(real_train_data_loader.dataset.Y))
        input_length = real_train_data_loader.dataset.X.shape[-1]

        if ssl_stage1:
            self.maskgit = Full_Embedding_MaskGIT(
                input_length,
                **self.config["MaskGIT"],
                config=self.config,
                n_classes=n_classes,
                finetune_codebook=False,
                device=device,
                load_finetuned_codebook=True,
            ).to(device)
            print("Full embedding MaskGIT sampler loaded..")
        else:
            self.maskgit = MaskGIT(
                input_length,
                **self.config["MaskGIT"],
                config=self.config,
                n_classes=n_classes,
            ).to(device)
            print("MaskGIT sampler loaded..")

        # load
        dataset_name = self.config["dataset"]["dataset_name"]
        model_name = "fullembed-maskgit-finetuned" if ssl_stage1 else "maskgit"
        ckpt_fname = os.path.join(
            "saved_models",
            f"{model_filename(self.config, model_name)}-{dataset_name}.ckpt",
        )
        saved_state = torch.load(ckpt_fname)
        try:
            self.maskgit.load_state_dict(saved_state)
        except:
            saved_state_renamed = (
                {}
            )  # need it to load the saved model from the odin server.
            for k, v in saved_state.items():
                if ".ff." in k:
                    saved_state_renamed[k.replace(".ff.", ".net.")] = v
                else:
                    saved_state_renamed[k] = v
            saved_state = saved_state_renamed
            self.maskgit.load_state_dict(saved_state)

        # inference mode
        self.maskgit.eval()

    @torch.no_grad()
    def unconditional_sample(
        self,
        n_samples: int,
        class_index=None,
        batch_size=256,
        return_representations=False,
    ):
        return unconditional_sample(
            self.maskgit,
            n_samples,
            self.device,
            class_index,
            batch_size,
            return_representations,
            self.guidance_scale,
        )

    @torch.no_grad()
    def conditional_sample(
        self,
        n_samples: int,
        class_index: int,
        batch_size=256,
        return_representations=False,
    ):
        """
        class_index: starting from 0. If there are two classes, then `class_index` ∈ {0, 1}.
        """
        return conditional_sample(
            self.maskgit,
            n_samples,
            self.device,
            class_index,
            batch_size,
            return_representations,
            self.guidance_scale,
        )

    def sample(self, kind: str, n_samples: int, class_index: int, batch_size: int):
        if kind == "unconditional":
            x_new = self.unconditional_sample(
                n_samples, None, batch_size
            )  # (b c l); b=n_samples, c=1 (univariate)
        elif kind == "conditional":
            x_new = self.conditional_sample(
                n_samples, class_index, batch_size
            )  # (b c l); b=n_samples, c=1 (univariate)
        else:
            raise ValueError
        return x_new
