import math
import os
import time

import numpy as np
import torch
from pytorch_lightning import Trainer, Callback
from pytorch_lightning import seed_everything
from torch.utils.data import Dataset, DataLoader

from mingpt.lr_decay import LearningRateDecayCallback
from mingpt.model import GPT


class CUDACallback(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        max_memory = torch.tensor(int(max_memory), dtype=torch.int, device=trainer.root_gpu)
        epoch_time = torch.tensor(int(epoch_time), dtype=torch.int, device=trainer.root_gpu)

        torch.distributed.all_reduce(max_memory, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(epoch_time, op=torch.distributed.ReduceOp.SUM)

        world_size = torch.distributed.get_world_size()

        print(f"Average Epoch time: {epoch_time.item() / float(world_size):.2f} seconds")
        print(f"Average Peak memory {max_memory.item() / float(world_size):.2f}MiB")


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i + self.block_size + 1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


if __name__ == '__main__':
    seed_everything(42)
    block_size = 128  # spatial extent of the model for its context

    if not os.path.exists("input.txt"):
        os.system("wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")

    # you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
    text = open('input.txt', 'r').read()  # don't worry we won't run out of file handles
    train_dataset = CharDataset(text, block_size)  # one line of poem is roughly 50 characters
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4)

    model = GPT(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        n_layer=15,
        n_head=16,
        n_embd=3072,
        learning_rate=6e-4
    )

    # scheduler
    lr_decay = LearningRateDecayCallback(
        learning_rate=6e-4,
        warmup_tokens=512 * 20,
        final_tokens=00 * len(train_dataset) * block_size
    )

    trainer = Trainer(
        gpus=4,
        precision=16,
        accelerator='ddp',
        max_epochs=1,
        gradient_clip_val=1.0,
        callbacks=[lr_decay, CUDACallback()],
    )
    trainer.fit(model, train_loader)
