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

from models.stage2.transformers import (
    BidirectionalTransformer,
    FullEmbedBidirectionalTransformer,
)
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


class Full_Embedding_MaskGIT(nn.Module):
    """
    references:
        1. https://github.com/ML4ITS/TimeVQVAE/blob/main/generators/maskgit.py'
        2. https://github.com/dome272/MaskGIT-pytorch/blob/cff485ad3a14b6ed5f3aa966e045ea2bc8c68ad8/transformer.py#L11
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
        ssl_method = config["SSL"]["stage1_method"]

        self.load(
            self.encoder,
            get_root_dir().joinpath("saved_models"),
            f"{model_filename(config, 'encoder')}-{dataset_name}.ckpt",
        )
        print(f"{ssl_method} encoder loaded")
        self.load(
            self.decoder,
            get_root_dir().joinpath("saved_models"),
            f"{model_filename(config, 'decoder')}-{dataset_name}.ckpt",
        )
        print(f"{ssl_method} decoder loaded")
        self.load(
            self.vq_model,
            get_root_dir().joinpath("saved_models"),
            f"{model_filename(config, 'vqmodel')}-{dataset_name}.ckpt",
        )
        print(f"{ssl_method} vqmodel loaded")

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

        # pretrained discrete tokens
        embed = nn.Parameter(copy.deepcopy(self.vq_model._codebook.embed))

        self.transformer = FullEmbedBidirectionalTransformer(
            self.H_prime * self.W_prime,
            config["VQVAE"]["codebook"]["size"],
            config["VQVAE"]["codebook"]["dim"],
            **config["MaskGIT"]["prior_model"],
            n_classes=n_classes,
            pretrained_tok_emb=embed,
        )

        # stochastic codebook sampling
        self.vq_model._codebook.sample_codebook_temp = stochastic_sampling

        self.mask_token_ids = config["VQVAE"]["codebook"]["size"]
        self.mask_emb = nn.Parameter(
            torch.randn(
                dim,
            )
        )

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

    def forward(self, x, y):
        """
        x: (B, C, L)
        y: (B, 1)
        straight from [https://github.com/dome272/MaskGIT-pytorch/blob/main/transformer.py]
        """
        device = x.device
        z_q, s = self.encode_to_z_q(x, self.encoder, self.vq_model)  # (b n)

        z_q = rearrange(z_q, "b c h w -> b (h w) c")  # (b, h*w, c)

        # randomly sample `t`
        t = np.random.uniform(0, 1)

        n_masks = math.floor(self.gamma(t) * s.shape[1])
        rand = torch.rand(s.shape, device=device)  # (b n)
        mask = torch.zeros(s.shape, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=rand.topk(n_masks, dim=1).indices, value=True)

        mask_emb = self.mask_emb.view(1, 1, -1)  # reshape to (1, 1, c)
        z_q_M = torch.where(mask.unsqueeze(-1), mask_emb, z_q)

        # prediction
        logits = self.transformer(
            z_q_M.detach(), class_condition=y
        )  # (b n codebook_size)
        target = s  # (b n)

        return logits, target

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

    # def create_input_tokens_normal(self, num, num_embeddings, device):
    #     """
    #     returns masked tokens
    #     """

    #     blank_embeddings = self.mask_emb.repeat(num_embeddings * num)
    #     blank_embeddings = blank_embeddings.view(
    #         num, num_embeddings, self.mask_emb.shape[0]
    #     )

    #     return blank_embeddings.to(torch.float32)

    def create_input_tokens_normal(self, num, num_tokens, mask_token_ids, device):
        """
        returns masked tokens
        """
        blank_tokens = torch.ones((num, num_tokens), device=device)
        masked_tokens = mask_token_ids * blank_tokens
        return masked_tokens.to(torch.int64)

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

    def sample_good(
        self,
        s: torch.Tensor,
        unknown_number_in_the_beginning,
        class_condition: Union[torch.Tensor, None],
        guidance_scale: float,
        gamma: Callable,
        device,
        stats: bool = False,
    ):

        probs_array = []
        s_array = []
        entropy_array = []
        selected_entropy_array = []

        for t in range(self.T):

            z_q = self.s_to_z_q(s, self.vq_model, self.mask_token_ids, self.mask_emb)

            logits = self.transformer(
                z_q, class_condition=class_condition
            )  # (b n codebook_size) == (b n K)

            if isinstance(class_condition, torch.Tensor):
                logits_null = self.transformer(z_q, class_condition=None)
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

            if stats:
                s_array.append(s)
                probs_array.append(selected_probs)

                # Filter out `inf` and `nan` values before entropy calculation
                finite_probs = probs[torch.isfinite(probs)].view(
                    probs.size(0), probs.size(1), -1
                )  # Reshape back to original after filtering
                finite_selected_probs = selected_probs[torch.isfinite(selected_probs)]

                # avoid log 0
                epsilon = 1e-5

                # Calculate entropy for finite probabilities
                entropy = -torch.sum(
                    finite_probs * torch.log(finite_probs + epsilon), dim=-1
                )
                selected_entropy = -torch.sum(
                    finite_selected_probs * torch.log(finite_selected_probs + epsilon),
                    dim=-1,
                )

                entropy_array.append(entropy)
                selected_entropy_array.append(selected_entropy)

        if stats:
            return s_array, probs_array, entropy_array, selected_entropy_array
        return s

    @torch.no_grad()
    def s_to_z_q(self, s, vq_model, mask_token_ids, mask_emb):
        # s: (b, n) containing masked tokens
        # z_q : (b, n, d)
        unmasked_map = ~(s == mask_token_ids)  # b n
        unmasked_s = s[unmasked_map].reshape(s.shape[0], -1)  # b n_unmasked
        unmasked_z_q = vq_model.project_out(
            F.embedding(unmasked_s, vq_model._codebook.embed)
        )  # b n_unmasked d. Unmasked z_q's

        # Create a tensor filled with mask_emb
        z_q = mask_emb.repeat(s.shape[0], s.shape[1], 1)

        # Flatten unmasked_map and unmasked_z_q to match the dimensions
        unmasked_map = unmasked_map.view(-1)
        unmasked_z_q = unmasked_z_q.view(-1, unmasked_z_q.shape[-1])

        # Replace the unmasked locations in z_q with unmasked_z_q
        z_q.view(-1, z_q.shape[-1])[unmasked_map] = unmasked_z_q

        # print(torch.sum(z_q == mask_emb, dim=1)
        return z_q

    @torch.no_grad()
    def iterative_decoding(
        self,
        num=1,
        mode="cosine",
        class_index=None,
        device="cpu",
        guidance_scale: float = 1.0,
        stats: bool = False,
    ):
        """
        It performs the iterative decoding and samples token indices.
        :param num: number of samples
        :return: sampled token indices
        """
        s = self.create_input_tokens_normal(
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

        if stats:
            s_array, probs, entropy, sel_entropy = self.sample_good(
                s,
                unknown_number_in_the_beginning,
                class_condition,
                guidance_scale,
                gamma,
                device,
                stats=True,
            )
            return (s_array, probs, entropy, sel_entropy)

        s = self.sample_good(
            s,
            unknown_number_in_the_beginning,
            class_condition,
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
