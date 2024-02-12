import math
from typing import Union

import torch
import torch.nn as nn
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


class IntegratedBidirectionalTransformer(nn.Module):
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
        Contains a encoder transformer and a decoder transformer.
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

        # self.latent_tok_emb = nn.Embedding(codebook_size + 1, embed_dim)

        # Class Conditional Embedding (+1 for unconditional scenario)
        self.class_condition_emb = nn.Embedding(n_classes + 1, embed_dim)

        # Positional Embeddings (+2 to account for summary and class embeddings)
        self.pos_emb = nn.Embedding(num_tokens + 2, embed_dim)

        # Summary Embedding (learnable parameter). Randomly initialized
        # self.summary_emb = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.summary_emb = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.summary_emb = nn.Embedding(num_embeddings=1, embedding_dim=embed_dim)

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

    def forward(
        self,
        embed_ind,
        class_condition: Union[None, torch.Tensor] = None,
        return_summary: bool = False,
        return_latent_tokens: bool = False,
    ):
        device = embed_ind.device
        batch_size = embed_ind.size(0)

        # Token embeddings
        token_embeddings = self.tok_emb(embed_ind)

        # Class and summary embeddings
        class_embeddings = self.class_embedding(class_condition, batch_size, device)
        summary_embedding = self.summary_emb.repeat(batch_size, 1, 1)

        # Positional embeddings for encoder. +2 to account for summary and class embeddings.
        position_embeddings_enc = self.pos_emb(
            torch.arange(self.num_tokens + 2, device=device)
        ).unsqueeze(0)
        # Positional embeddings for decoder. +1 to account for class embedding.
        position_embeddings_dec = self.pos_emb(
            torch.arange(self.num_tokens + 1, device=device)
        ).unsqueeze(0)

        # --- Encoder embedding ---
        encoder_embed = torch.cat(
            (class_embeddings, summary_embedding, token_embeddings), dim=1
        )
        encoder_embed += position_embeddings_enc
        encoder_embed = self.drop(self.ln(encoder_embed))
        encoder_embed = self.encoder_blocks(encoder_embed)

        # extracting summary and latent representation. Normalized, as done in paper on MAGE architecture:
        summary = self.ln(encoder_embed[:, 1, :])
        latent_token_embedding = encoder_embed[:, 2:, :]

        # --- Decoder embedding.---
        decoder_embed = torch.cat((class_embeddings, latent_token_embedding), dim=1)
        decoder_embed += position_embeddings_dec
        decoder_embed = self.drop(self.ln(decoder_embed))
        decoder_embed = self.decoder_blocks(decoder_embed)
        # --- logits prediction ---:
        decoder_embed = self.Token_Prediction(decoder_embed[:, 1:, :])
        logits = (
            torch.matmul(decoder_embed, self.tok_emb.weight.T) + self.bias
        )  # (b, n, codebook_size+1)logits =

        logits = logits[
            :, :, :-1
        ]  # remove the logit for the mask token.  # (b, n, codebook_size)

        # Return the summary and/or latent token embeddings
        if return_summary and return_latent_tokens:
            return logits, summary, latent_token_embedding
        elif return_summary:
            return logits, summary
        elif return_latent_tokens:
            return logits, latent_token_embedding
        else:
            return logits
