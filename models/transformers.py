import math
from typing import Union

import torch
import torch.nn as nn
import numpy as np
from einops import repeat
from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder


def load_pretrained_tok_emb(
    pretrained_tok_emb, tok_emb, freeze_pretrained_tokens: bool
):
    """
    :param pretrained_tok_emb: pretrained token embedding from stage 1
    :param tok_emb: token embedding of the transformer
    :return:
    """
    with torch.no_grad():
        if pretrained_tok_emb != None:
            tok_emb.weight[:-1, :] = pretrained_tok_emb
            if freeze_pretrained_tokens:
                tok_emb.weight[:-1, :].requires_grad = False


class BidirectionalTransformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        codebook_size: int,
        embed_dim: int,
        hidden_dim: int,
        n_layers: int,
        heads: int,
        ff_mult: int,
        use_rmsnorm: bool,
        p_unconditional: float,
        n_classes: int,
        pretrained_tok_emb: nn.Parameter = None,
        freeze_pretrained_tokens: bool = False,
        **kwargs
    ):
        """
        :param num_tokens:
        :param codebook_sizes:
        :param embed_dim:
        :param hidden_dim:
        :param n_layers:
        :param heads:
        :param ff_mult:
        :param use_rmsnorm:
        :param p_unconditional:
        :param n_classes:
        :param pretrained_tok_emb: if given, the embedding of the transformer is initialized with the pretrained embedding from stage 1
        :param freeze_pretrained_tokens:
        :param num_tokens:
        :param kwargs:
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim
        out_dim = embed_dim

        # token embeddings
        self.tok_emb = nn.Embedding(
            codebook_size + 1, embed_dim
        )  # `+1` is for mask-token
        load_pretrained_tok_emb(
            pretrained_tok_emb, self.tok_emb, freeze_pretrained_tokens
        )

        # transformer
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.class_condition_emb = nn.Embedding(
            n_classes + 1, in_dim
        )  # `+1` is for no-condition
        self.blocks = ContinuousTransformerWrapper(
            dim_in=in_dim,
            dim_out=in_dim,
            max_seq_len=self.num_tokens + 1,
            attn_layers=TFEncoder(
                dim=hidden_dim,
                depth=n_layers,
                heads=heads,
                use_rmsnorm=use_rmsnorm,
                ff_mult=ff_mult,
                use_abs_pos_emb=False,
            ),
        )
        self.Token_Prediction = nn.Sequential(
            *[
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim, eps=1e-12),
            ]
        )
        self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))
        self.ln = nn.LayerNorm(in_dim, eps=1e-12)
        self.drop = nn.Dropout(p=0.0)

    def class_embedding(
        self, class_condition: Union[None, torch.Tensor], batch_size: int, device
    ):
        if isinstance(class_condition, torch.Tensor):
            # if condition is given (conditional sampling)
            conditional_ind = (
                torch.rand(class_condition.shape).to(device) > self.p_unconditional
            )
            class_uncondition = repeat(
                torch.Tensor([self.n_classes]).long().to(device),
                "i -> b i",
                b=batch_size,
            )  # (b 1)

            class_condition = torch.where(
                conditional_ind, class_condition.long(), class_uncondition
            )  # (b 1)
        else:
            # if condition is not given (unconditional sampling)
            class_uncondition = repeat(
                torch.Tensor([self.n_classes]).long().to(device),
                "i -> b i",
                b=batch_size,
            )  # (b 1)
            class_condition = class_uncondition
        cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
        return cls_emb

    def forward(self, embed_ind, class_condition: Union[None, torch.Tensor] = None):
        device = embed_ind.device

        token_embeddings = self.tok_emb(embed_ind)  # (b n dim)
        cls_emb = self.class_embedding(
            class_condition, embed_ind.shape[0], device
        )  # (b 1 dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = self.drop(
            self.ln(token_embeddings + position_embeddings)
        )  # (b, n, dim)
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
        embed = self.blocks(embed)  # (b, 1+n, dim)
        embed = self.Token_Prediction(embed)[:, 1:, :]  # (b, n, dim)

        logits = (
            torch.matmul(embed, self.tok_emb.weight.T) + self.bias
        )  # (b, n, codebook_size+1)
        logits = logits[
            :, :, :-1
        ]  # remove the logit for the mask token.  # (b, n, codebook_size)
        return logits


class AutoEncoderTransformer(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        codebook_size: int,
        embed_dim: int,
        hidden_dim: int,
        encoder_layers: int,
        decoder_layers: int,
        heads: int,
        ff_mult: int,
        use_rmsnorm: bool,
        p_unconditional: float,
        n_classes: int,
        pretrained_tok_emb: nn.Parameter = None,
        freeze_pretrained_tokens: bool = False,
    ):
        """
        Contains two bidirectional transformers, one for encoding and one for decoding.

        The encoded representation gets masked with a summary vector before being passed to the decoder.
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim
        out_dim = embed_dim

        # Token Embeddings (+1 for mask token)
        self.tok_emb = nn.Embedding(
            codebook_size + 1, embed_dim
        )  # `+1` is for mask-token
        load_pretrained_tok_emb(
            pretrained_tok_emb, self.tok_emb, freeze_pretrained_tokens
        )

        # Class Conditional Embedding (+1 for unconditional scenario)
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)

        # positional embedding
        self.pos_emb = nn.Embedding(num_tokens + 1, in_dim)

        # Summary Embedding (learnable parameter). Randomly initialized
        # self.summary = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.summary_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.summary_emb, std=0.02)

        # Encoder Transformer
        self.encoder_blocks = ContinuousTransformerWrapper(
            dim_in=in_dim,
            dim_out=in_dim,
            max_seq_len=num_tokens + 2,  # Adjust for summary and class embedding
            attn_layers=TFEncoder(
                dim=hidden_dim,
                depth=encoder_layers,
                heads=heads,
                use_rmsnorm=use_rmsnorm,
                ff_mult=ff_mult,
                use_abs_pos_emb=False,  # Using relative positional embeddings
            ),
        )

        # Decoder Transformer
        self.decoder_blocks = ContinuousTransformerWrapper(
            dim_in=in_dim,
            dim_out=in_dim,
            max_seq_len=num_tokens + 1,  # Adjust for class embedding
            attn_layers=TFEncoder(
                dim=hidden_dim,
                depth=decoder_layers,
                heads=heads,
                use_rmsnorm=use_rmsnorm,
                ff_mult=ff_mult,
                use_abs_pos_emb=False,
            ),
        )

        # Final Projection to logits
        self.Token_Prediction = nn.Sequential(
            *[
                nn.Linear(in_features=in_dim, out_features=out_dim),
                nn.GELU(),
                nn.LayerNorm(out_dim, eps=1e-12),
            ]
        )

        self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))
        self.ln = nn.LayerNorm(in_dim, eps=1e-12)
        self.drop = nn.Dropout(p=0.0)

    def class_embedding(
        self,
        class_condition: Union[None, torch.Tensor],
        batch_size: int,
        device,
    ):
        self.class_condition_emb = self.class_condition_emb.to(device)
        if class_condition is not None:
            class_condition = class_condition.to(device)

        # returns a class embedding based on the class condition. If no condition is given, it returns an unconditional class embedding.
        # It also will sometimes turn off the class embedding (unconditional sampling) with p_unconditional.
        if isinstance(class_condition, torch.Tensor):
            # if condition is given (conditional sampling)
            conditional_ind = (
                torch.rand(class_condition.shape) > self.p_unconditional
            ).to(device)
            class_uncondition = repeat(
                torch.Tensor([self.n_classes]).long(),
                "i -> b i",
                b=batch_size,
            ).to(
                device
            )  # (b 1)

            class_condition = torch.where(
                conditional_ind,
                class_condition.long().to(device),
                class_uncondition,
            ).to(
                device
            )  # (b 1)
        else:
            # if condition is not given (unconditional sampling)
            class_uncondition = repeat(
                torch.Tensor([self.n_classes]).long().to(device),
                "i -> b i",
                b=batch_size,
            )  # (b 1)
            class_condition = class_uncondition

        cls_emb = self.class_condition_emb(class_condition.long())  # (b 1 dim)
        return cls_emb

    def forward_encoder(
        self,
        unmasked_token_emb,
        class_emb,
    ):
        # Creates a encoded representation and a summary.
        n = unmasked_token_emb.size(1)
        position_emb = self.pos_emb.weight[:n, :]
        summary_emb = self.summary_emb.repeat(unmasked_token_emb.size(0), 1, 1)

        encoder_emb = self.drop(self.ln(unmasked_token_emb + position_emb))
        encoder_emb = torch.cat((class_emb, summary_emb, encoder_emb), dim=1)
        encoder_emb = self.encoder_blocks(encoder_emb)

        latent = encoder_emb[:, 2:, :]
        summary = encoder_emb[:, 1, :]
        return latent, summary

    def forward_decoder(
        self,
        class_emb,
        padded_latent,
    ):
        # Takes in padded latent. I.e representation from encoder.
        n = padded_latent.size(1)
        position_emb = self.pos_emb.weight[:n, :]

        decoder_emb = self.drop(self.ln(padded_latent + position_emb))
        decoder_emb = torch.cat((class_emb, decoder_emb), dim=1)
        decoder_emb = self.decoder_blocks(decoder_emb)
        # logits prediction given transformer representation:
        decoder_emb = self.Token_Prediction(decoder_emb)[:, 1:, :]
        logits = torch.matmul(decoder_emb, self.tok_emb.weight.T) + self.bias
        return logits

    def forward(
        self,
        embed_ind,
        class_condition: Union[None, torch.Tensor] = None,
        masks=None,
    ):
        device = embed_ind.device

        # embed_ind.to(device)
        batch_size = embed_ind.size(0)

        token_emb = self.tok_emb(embed_ind).to(device)
        cls_emb = self.class_embedding(class_condition, batch_size, device)

        # Filter unmasked tokens
        unmasked_tokens = self.filter_unmasked_tokens(token_emb, masks, device)

        # Encoding to latent representation and summary
        latent, summary = self.forward_encoder(unmasked_tokens, cls_emb)

        # Reconstruct latent representation with summary at masked positions,
        # and latent at unmasked positions.
        summary_latent_emb = self.pad_latent_representation(
            latent, summary, masks, token_emb.shape, device
        ).to(device)

        # Decoding to logits prediction
        logits = self.forward_decoder(cls_emb, summary_latent_emb).to(device)

        return logits, summary

    @torch.no_grad()
    def summarize(self, embed_ind, class_condition: Union[None, torch.Tensor] = None):
        # Gives a "summary" of the embed_ind. Used for downstream classification.
        device = embed_ind.device
        batch_size = embed_ind.size(0)

        token_emb = self.tok_emb(embed_ind)
        cls_emb = self.class_embedding(class_condition, batch_size, device)
        # Here I assume we are interested in the summary without prior knowledge of class. Therefor None is standard.
        # A unconditioned summary is interesting.
        _, summary = self.forward_encoder(token_emb, cls_emb)

        return self.ln(summary)

    def filter_unmasked_tokens(self, token_emb, masks, device):
        if masks is None:
            return token_emb

        # Assuming masks is a boolean tensor indicating masked positions
        masks_expanded = masks.unsqueeze(-1).expand_as(token_emb)
        unmasked_tokens = token_emb[~masks_expanded].view(
            token_emb.shape[0], -1, token_emb.shape[-1]
        )

        return unmasked_tokens.to(device)

    def pad_latent_representation(
        self, latent, summary, masks, token_emb_shape, device
    ):
        device = latent.device
        reconstructed = torch.zeros(token_emb_shape, device=device)
        if masks is None:
            # If no masks are provided, use the latent as is, without needing to place the summary
            reconstructed[:] = latent
        else:
            masks_expanded = masks.unsqueeze(-1).expand_as(reconstructed)
            # Place latent representations back into their original positions
            reconstructed[~masks_expanded] = latent.reshape(-1)  #
            # Fill masked positions with the summary embedding
            summary_expanded = summary.unsqueeze(1).expand(-1, token_emb_shape[1], -1)
            reconstructed[masks_expanded] = summary_expanded[masks_expanded]
        return reconstructed.to(device)
