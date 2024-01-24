import math
from typing import Union

import torch
import torch.nn as nn
from einops import repeat
from x_transformers import ContinuousTransformerWrapper, Encoder as TFEncoder


def load_pretrained_tok_emb(pretrained_tok_emb, tok_emb, freeze_pretrained_tokens: bool):
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
    def __init__(self,
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
                 num_tokens_l: int = None,
                 **kwargs):
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
        :param num_tokens_l:
        :param kwargs:
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.n_classes = n_classes
        self.p_unconditional = p_unconditional
        in_dim = embed_dim 
        out_dim = embed_dim

        # token embeddings
        self.tok_emb = nn.Embedding(codebook_size + 1, embed_dim)  # `+1` is for mask-token
        load_pretrained_tok_emb(pretrained_tok_emb, self.tok_emb, freeze_pretrained_tokens)
        
        # transformer
        self.pos_emb = nn.Embedding(self.num_tokens + 1, in_dim)
        self.class_condition_emb = nn.Embedding(n_classes + 1, in_dim)  # `+1` is for no-condition
        self.blocks = ContinuousTransformerWrapper(dim_in=in_dim,
                                                   dim_out=in_dim,
                                                   max_seq_len=self.num_tokens + 1,
                                                   attn_layers=TFEncoder(
                                                       dim=hidden_dim,
                                                       depth=n_layers,
                                                       heads=heads,
                                                       use_rmsnorm=use_rmsnorm,
                                                       ff_mult=ff_mult,
                                                       use_abs_pos_emb=False,
                                                   ))
        self.Token_Prediction = nn.Sequential(*[
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim, eps=1e-12)
        ])
        self.bias = nn.Parameter(torch.zeros(self.num_tokens, codebook_size + 1))
        self.ln = nn.LayerNorm(in_dim, eps=1e-12)
        self.drop = nn.Dropout(p=0.)

    def class_embedding(self, class_condition: Union[None, torch.Tensor], batch_size: int, device):
        if isinstance(class_condition, torch.Tensor):
            # if condition is given (conditional sampling)
            conditional_ind = torch.rand(class_condition.shape).to(device) > self.p_unconditional
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            class_condition = torch.where(conditional_ind, class_condition.long(), class_uncondition)  # (b 1)
        else:
            # if condition is not given (unconditional sampling)
            class_uncondition = repeat(torch.Tensor([self.n_classes]).long().to(device), 'i -> b i', b=batch_size)  # (b 1)
            class_condition = class_uncondition
        cls_emb = self.class_condition_emb(class_condition)  # (b 1 dim)
        return cls_emb

    def forward(self, embed_ind, class_condition: Union[None, torch.Tensor] = None):
        device = embed_ind.device

        token_embeddings = self.tok_emb_l(embed_ind)  # (b n dim)
        cls_emb = self.class_embedding(class_condition, embed_ind.shape[0], device)  # (b 1 dim)

        n = token_embeddings.shape[1]
        position_embeddings = self.pos_emb.weight[:n, :]
        embed = self.drop(self.ln(token_embeddings + position_embeddings))  # (b, n, dim)
        embed = torch.cat((cls_emb, embed), dim=1)  # (b, 1+n, dim)
        embed = self.blocks(embed)  # (b, 1+n, dim)
        embed = self.Token_Prediction(embed)[:, 1:, :]  # (b, n, dim)

        logits = torch.matmul(embed, self.tok_emb_l.weight.T) + self.bias  # (b, n, codebook_size+1)
        logits = logits[:, :, :-1]  # remove the logit for the mask token.  # (b, n, codebook_size)
        return logits
