"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch

import torch.nn as nn
from deepspeed import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from fairscale.nn import checkpoint_wrapper, auto_wrap, wrap
from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.utilities import rank_zero_info
from torch.nn import functional as F
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    I believe I could have just used torch.nn.MultiheadAttention but their documentation
    is all but absent and code ugly so I don't trust it, rolling my own here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(pl.LightningModule):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self,
                 vocab_size,
                 weight_decay=0.1,
                 betas=(0.9, 0.95),
                 learning_rate=3e-4,
                 n_embd=768,
                 block_size=128,
                 embd_pdrop=0.1,
                 n_layer=12,
                 n_head=4,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 checkpoint=False,
                 should_auto_wrap=False,
                 should_wrap=False
                 ):
        super().__init__()
        # auto creates self.hparams from the method signature
        self.save_hyperparameters()

        # in lightning the "config" is hparams (for hyperparameters)
        self.config = self.hparams

        # input embedding stem
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, n_embd))
        self.drop = nn.Dropout(embd_pdrop)

        # decoder head
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)

        self.block_size = self.config.block_size
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_block_size(self):
        return self.block_size

    def configure_sharded_model(self) -> None:
        blocks = []
        for x in range(self.config.n_layer):
            layer = Block(self.config)
            if self.hparams.checkpoint and not self.use_deepspeed:
                layer = checkpoint_wrapper(layer)
            if self.hparams.should_auto_wrap:
                layer = auto_wrap(layer, reshard_after_forward=True)
            elif self.hparams.should_wrap:
                layer = wrap(layer, reshard_after_forward=True)
            blocks.append(layer)
        if self.hparams.should_wrap or self.hparams.should_auto_wrap:
            self.blocks = wrap(nn.Sequential(*blocks))
        else:
            self.blocks = nn.Sequential(*blocks)

    def configure_optimizers(self):
        rank_zero_info(self.trainer.model)
        # create the optimizer
        if self.use_deepspeed and self.cpu_offload:
            return DeepSpeedCPUAdam(
                self.trainer.model.parameters(),
                lr=self.hparams.learning_rate,
                betas=self.hparams.betas
            )
        return FusedAdam(
            self.trainer.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas
        )

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        if self.hparams.should_wrap or self.hparams.should_auto_wrap:
            x = self.blocks(x)
        else:
            for block in self.blocks:
                if self.hparams.checkpoint and self.use_deepspeed:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                else:
                    x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        # same action as inference
        logits = self(idx)

        if not hasattr(self, 'done'):
            rank_zero_info(self.trainer.model)
            self.done = True

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        self.log('train_loss', loss)
        return loss

    @property
    def use_deepspeed(self):
        return isinstance(self.trainer.accelerator.training_type_plugin, DeepSpeedPlugin)

    @property
    def cpu_offload(self):
        return self.use_deepspeed and (self.trainer.accelerator.training_type_plugin.config['zero_optimization'].get(
            'offload_optimizer') or self.trainer.accelerator.training_type_plugin.config['zero_optimization'].get(
            'offload_param'))
