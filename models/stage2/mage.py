import copy
import numpy as np
import math
from pathlib import Path
import tempfile
from typing import Union
from collections import deque
from torch import nn
import torch.nn.functional as F

import torch

from einops import repeat, rearrange
from typing import Callable
from scipy.stats import truncnorm

from models.stage2.transformers import AutoEncoderTransformer
from models.stage1.encoder_decoder import VQVAEEncoder, VQVAEDecoder
from models.stage1.vq import VectorQuantize

from utils import (
    compute_downsample_rate,
    get_root_dir,
    freeze,
    timefreq_to_time,
    time_to_timefreq,
    quantize,
    model_filename,
)


class MAGE(nn.Module):
    """
    references:
    """

    def __init__(
        self,
        input_length: int,
        choice_temperature: int,
        stochastic_sampling: int,
        T: int,
        config: dict,
        n_classes: int,
        **kwargs,
    ):
        super().__init__()
        self.choice_temperature = choice_temperature
        self.T = T
        self.config = config
        self.n_classes = n_classes

        self.mask_token_ids = config["VQVAE"]["codebook"]["size"]
        self.gamma = self.gamma_func("cosine")
        dataset_name = config["dataset"]["dataset_name"]

        # define encoder, decoder, vq_models
        dim = config["encoder"]["dim"]
        in_channels = config["dataset"]["in_channels"]
        downsampled_width = config["encoder"]["downsampled_width"]
        self.n_fft = config["VQVAE"]["n_fft"]
        downsample_rate = compute_downsample_rate(
            input_length, self.n_fft, downsampled_width
        )

        self.encoder = VQVAEEncoder(
            dim, 2 * in_channels, downsample_rate, config["encoder"]["n_resnet_blocks"]
        )
        self.decoder = VQVAEDecoder(
            dim, 2 * in_channels, downsample_rate, config["decoder"]["n_resnet_blocks"]
        )
        self.vq_model = VectorQuantize(
            dim, config["VQVAE"]["codebook"]["size"], **config["VQVAE"]
        )
        # load trained models for encoder, decoder, and vq_models
        stage1_ssl_method = config["SSL"]["stage1_method"]
        self.load(
            self.encoder,
            get_root_dir().joinpath("saved_models"),
            f"{model_filename(config, 'encoder')}-{dataset_name}.ckpt",
        )
        print(f"{stage1_ssl_method} encoder loaded")
        self.load(
            self.decoder,
            get_root_dir().joinpath("saved_models"),
            f"{model_filename(config, 'decoder')}-{dataset_name}.ckpt",
        )
        print(f"{stage1_ssl_method} decoder loaded")
        self.load(
            self.vq_model,
            get_root_dir().joinpath("saved_models"),
            f"{model_filename(config, 'vqmodel')}-{dataset_name}.ckpt",
        )
        print(f"{stage1_ssl_method} vqmodel loaded")

        # freeze the models for encoder, decoder, and vq_model
        freeze(self.encoder)
        freeze(self.decoder)
        freeze(self.vq_model)

        # evaluation model for encoder, decoder, and vq_model
        self.encoder.eval()
        self.decoder.eval()
        self.vq_model.eval()

        # token lengths
        self.num_tokens = self.encoder.num_tokens.item()

        # latent space dim
        self.H_prime = self.encoder.H_prime.item()
        self.W_prime = self.encoder.W_prime.item()

        # Masking generator
        mask_ratio_mu = config["MAGE"]["mask_ratio"]["mu"]
        mask_ratio_std = config["MAGE"]["mask_ratio"]["std"]
        self.mask_ratio_min = config["MAGE"]["mask_ratio"]["min"]
        mask_ratio_max = config["MAGE"]["mask_ratio"]["max"]
        self.mask_ratio_generator = truncnorm(
            (self.mask_ratio_min - mask_ratio_mu) / mask_ratio_std,
            (mask_ratio_max - mask_ratio_mu) / mask_ratio_std,
            loc=mask_ratio_mu,
            scale=mask_ratio_std,
        )

        # pretrained discrete tokens
        embed = nn.Parameter(copy.deepcopy(self.vq_model._codebook.embed))

        # Encoder Decoder Bidirectional Transformer
        self.autoencoder_transformer = AutoEncoderTransformer(
            self.num_tokens,
            config["VQVAE"]["codebook"]["size"],
            config["VQVAE"]["codebook"]["dim"],
            **config["MAGE"]["prior_model"],
            n_classes=n_classes,
            pretrained_tok_emb=embed,
        )

        # stochastic codebook sampling
        self.vq_model._codebook.sample_codebook_temp = stochastic_sampling

    def load(self, model, dirname, fname):
        """
        model: instance
        path_to_saved_model_fname: path to the ckpt file (i.e., trained model)
        """
        print(dirname)
        print(fname)
        try:
            model.load_state_dict(torch.load(dirname.joinpath(fname)))
        except FileNotFoundError:
            dirname = Path(tempfile.gettempdir())
            model.load_state_dict(torch.load(dirname.joinpath(fname)))

    @torch.no_grad()
    def encode_to_z_q(self, x, encoder: VQVAEEncoder, vq_model: VectorQuantize):
        """
        x: (B, C, L)
        """
        C = x.shape[1]
        xf = time_to_timefreq(x, self.n_fft, C)  # (B, C, H, W)
        z = encoder(xf)  # (b c h w)
        z_q, indices, vq_loss, perplexity = quantize(
            z, vq_model
        )  # (b c h w), (b (h w) h), ...
        return z_q, indices

    def forward(self, x, y, return_summaries: bool = False):
        """
        x: (B, C, L)
        y: (B, 1)
        """
        device = x.device
        _, s = self.encode_to_z_q(x, self.encoder, self.vq_model)  # (b n)

        # --- Generating mask and drop ---
        s_M1, token_all_mask1, token_drop_mask1 = self.generate_mage_mask_drop(
            s, device
        )
        s_M2, token_all_mask2, token_drop_mask2 = self.generate_mage_mask_drop(
            s, device
        )

        # --- Encode-Decode transformers ---
        logits1, summary1 = self.autoencoder_transformer(
            s_M1, y, token_all_mask1, token_drop_mask1
        )
        logits2, summary2 = self.autoencoder_transformer(
            s_M2, y, token_all_mask2, token_drop_mask2
        )

        logits = [logits1, logits2]
        summaries = [summary1, summary2]
        target = s

        if return_summaries:
            return logits, summaries, target
        else:
            return logits, target

    def generate_mage_mask_drop(self, s, device):
        # Method used in MAGE paper.
        s_M = s.clone()
        # sample a masking ratio from a truncated Gaussian distribution
        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        drop_rate = self.mask_ratio_min

        # calculate the number of tokens to mask and to drop
        bsz, seq_len = s_M.size()

        n_masks = int(np.ceil(mask_rate * seq_len))
        n_drops = int(np.ceil(drop_rate * seq_len))

        while True:
            noise = torch.rand(s.shape, device=device)  # (b n)
            sorted_noise, _ = torch.sort(noise, dim=1)

            cutoff_drop = sorted_noise[:, n_drops - 1 : n_drops]
            cutoff_mask = sorted_noise[:, n_masks - 1 : n_masks]

            token_drop_mask = (noise <= cutoff_drop).float()
            token_all_mask = (noise <= cutoff_mask).float()
            if (
                token_drop_mask.sum() == bsz * n_drops
                and token_all_mask.sum() == bsz * n_masks
            ):
                break
            else:
                print("whoopsie, universe not happy..")

        s_M[token_all_mask.nonzero(as_tuple=True)] = self.mask_token_ids  # (b n)

        return s_M, token_all_mask, token_drop_mask

    @torch.no_grad()
    def summarize(self, x, y=None):
        # Ensure x is a PyTorch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        _, s = self.encode_to_z_q(x, self.encoder, self.vq_model)
        summary = self.autoencoder_transformer.summarize(s, y)
        return summary

    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r**2
        elif mode == "cubic":
            return lambda r: 1 - r**3
        else:
            raise NotImplementedError

    def create_input_tokens_normal(self, num, num_tokens, mask_token_ids, device):
        # Initialize blank tokens and create masked tokens by multiplying with mask_token_ids
        blank_tokens = torch.ones((num, num_tokens), device=device)
        masked_tokens = mask_token_ids * blank_tokens

        # Create a mask with all True values, indicating all positions are masked
        mask = torch.ones((num, num_tokens), dtype=torch.bool, device=device)

        return masked_tokens.to(torch.int64), mask

    def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device="cpu"):
        """
        mask_len: (b 1)
        probs: (b n); also for the confidence scores

        This version keeps `mask_len` exactly.
        """

        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            """
            Gumbel max trick: https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch
            """
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        confidence = torch.log(probs + 1e-5) + temperature * gumbel_noise(probs).to(
            device
        )  # Gumbel max trick; 1e-5 for numerical stability; (b n)
        mask_len_unique = int(mask_len.unique().item())
        masking_ind = torch.topk(
            confidence, k=mask_len_unique, dim=-1, largest=False
        ).indices  # (b k)
        masking = torch.zeros_like(confidence).to(device)  # (b n)
        for i in range(masking_ind.shape[0]):
            masking[i, masking_ind[i].long()] = 1.0
        masking = masking.bool()
        return masking

    def sample(
        self,
        s: torch.Tensor,
        unknown_number_in_the_beginning,
        class_condition: Union[torch.Tensor, None],
        init_masking,
        guidance_scale: float,
        gamma: Callable,
        device,
    ):
        masking = init_masking

        for t in range(self.T):
            logits, _ = self.autoencoder_transformer(
                embed_ind=s,
                class_condition=class_condition,
                token_all_mask=masking,
            )  # (b n codebook_size) == (b n K)
            if isinstance(class_condition, torch.Tensor):
                logits_null, _ = self.autoencoder_transformer(
                    embed_ind=s, class_condition=None, token_all_mask=masking
                )
                logits = logits_null + guidance_scale * (logits - logits_null)

            sampled_ids = torch.distributions.categorical.Categorical(
                logits=logits
            ).sample()  # (b n)
            unknown_map = (
                s == self.mask_token_ids
            )  # which tokens need to be sampled; (b n)
            sampled_ids = torch.where(
                unknown_map, sampled_ids, s
            )  # keep the previously-sampled tokens; (b n)

            # create masking according to `t`
            ratio = 1.0 * (t + 1) / self.T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)  # convert logits into probs; (b n K)
            selected_probs = torch.gather(
                probs, dim=-1, index=sampled_ids.unsqueeze(-1)
            ).squeeze()  # get probability for the selected tokens; p(\hat{s}(t) | \hat{s}_M(t)); (b n)

            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(device)
            selected_probs = torch.where(
                unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS
            )  # assign inf probability to the previously-selected tokens; (b n)

            mask_len = torch.unsqueeze(
                torch.floor(unknown_number_in_the_beginning * mask_ratio), 1
            )  # number of tokens that are to be masked;  (b,)
            mask_len = torch.clip(
                mask_len, min=0.0
            )  # `mask_len` should be equal or larger than zero.

            # Adds noise for randomness
            masking = self.mask_by_random_topk(
                mask_len,
                selected_probs,
                temperature=self.choice_temperature * (1.0 - ratio),
                device=device,
            )

            # Masks tokens with lower confidence.
            s = torch.where(masking, self.mask_token_ids, sampled_ids)  # (b n)

        return s

    @torch.no_grad()
    def iterative_decoding(
        self,
        num=1,
        mode="cosine",
        class_index=None,
        device="cpu",
        guidance_scale: float = 1.0,
    ):
        """
        It performs the iterative decoding and samples token indices.
        :param num: number of samples
        :return: sampled token indices
        """
        s, init_masks = self.create_input_tokens_normal(
            num, self.num_tokens, self.mask_token_ids, device
        )  # (b n)

        unknown_number_in_the_beginning = torch.sum(
            s == self.mask_token_ids, dim=-1
        )  # (b,)

        gamma = self.gamma_func(mode)
        class_condition = (
            repeat(torch.Tensor([class_index]).int().to(device), "i -> b i", b=num)
            if class_index != None
            else None
        )  # (b 1)

        s = self.sample(
            s,
            unknown_number_in_the_beginning,
            class_condition,
            init_masks,
            guidance_scale,
            gamma,
            device,
        )

        return s

    def decode_token_ind_to_timeseries(
        self, s: torch.Tensor, return_representations: bool = False
    ):
        #
        # It takes token embedding indices and decodes them to time series.
        #:param s: token embedding index
        #:param return_representations:
        #:return:
        #

        vq_model = self.vq_model
        decoder = self.decoder

        quantize = F.embedding(s, vq_model._codebook.embed)  # (b n d)
        quantize = vq_model.project_out(quantize)  # (b n c)

        quantize = rearrange(quantize, "b n c -> b c n")  # (b c n) == (b c (h w))

        # print("quantize.shape before reshaping:", quantize.shape)
        # print(self.H_prime, self.W_prime)

        quantize = rearrange(
            quantize, "b c (h w) -> b c h w", h=self.H_prime, w=self.W_prime
        )

        uhat = decoder(quantize)

        xhat = timefreq_to_time(
            uhat, self.n_fft, self.config["dataset"]["in_channels"]
        )  # (B, C, L)

        if return_representations:
            return xhat, quantize
        else:
            return xhat

    def compute_confidence_score(
        self, s, mask_token_ids, vq_model, transformer, class_condition
    ):
        confidence_scores = torch.zeros_like(s).float()  # (b n)
        for n in range(confidence_scores.shape[-1]):
            s_m = copy.deepcopy(s)  # (b n)
            s_m[:, n] = (
                mask_token_ids  # (b n); masking the n-th token to measure the confidence score for that token.
            )
            logits = transformer(s_m, class_condition=class_condition)  # (b n K)
            logits = torch.nn.functional.softmax(logits, dim=-1)  # (b n K)

            true_tokens = s[:, n]  # (b,)
            logits = logits[:, n]  # (b, K)
            pred_tokens = logits.argmax(dim=-1)  # (b,)

            z_q_true = vq_model._codebook.embed[true_tokens]  # (b, dim)
            z_q_pred = vq_model._codebook.embed[pred_tokens]  # (b, dim)
            dist = torch.sum((z_q_true - z_q_pred) ** 2, dim=-1)  # (b,)
            confidence_scores[:, n] = -1 * dist  # confidence score for the n-th token
        confidence_scores = torch.nn.functional.softmax(
            confidence_scores, dim=-1
        )  # (b n)
        return confidence_scores
