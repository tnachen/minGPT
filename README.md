# minGPT Lightning Benchmark

![mingpt](mingpt.jpg)

Modified [Andrej's](https://github.com/karpathy/minGPT) and [William's](https://github.com/williamFalcon/minGPT) awesome code to create a simple benchmarking script.

### Usage

```
pip install -r requirements.txt
```

We benchmark a few different models with different configurations. Below is an example of some of the configurations we can run:

```bash
# 1.6B parameters
python benchmark.py --n_layer 14 --n_head 16 --n_embd 3072 --gpus 8 --precision 16 --accelerator ddp --limit_train_batches 120
```

```bash
# 2.5B parameters
python benchmark.py --n_layer 22 --n_head 16 --n_embd 3072 --gpus 8 --plugins deepspeed --precision 16 --limit_train_batches 120
```

```bash
# 13B parameters, only possible with DeepSpeed
python benchmark.py --n_layer 16 --n_head 16 --n_embd 8192 --gpus 8 --plugins deepspeed --precision 16 --limit_train_batches 120
```

### Results

Results were collected on an 8 GPU A100 server.

#### DDP vs DeepSpeed

The first set of results we collected were using a model size that fit training with DDP (roughly 1.6B parameters). 
When using DeepSpeed, I noticed that for the first 20 batches the optimizer step was skipped as infs were detected.

Command:
```bash
1.6B
python benchmark.py --n_layer 14 --n_head 16 --n_embd 3072 --gpus 8 --accelerator ddp --precision 16 --limit_train_batches 120
```

##### DDP
```
Average Epoch time: 40.27 seconds
Average Peak memory 35834.96MiB
```
##### DeepSpeed Default (With ZeRO-Offload)
```
Average Epoch time: 357.26 seconds
Average Peak memory 9993.60MiB
```
##### DeepSpeed With ZeRO, no Offload
```
Average Epoch time: 18.41 seconds
Average Peak memory 12625.53MiB
```
##### DeepSpeed Without ZeRO-Offload (requires instantiating the plugin, example below)
```
Average Epoch time: 33.27 seconds
Average Peak memory 30698.40MiB
```

#### Maximum DeepSpeed!

My attempt to get the fit the largest model I could train on this machine reasonably:

```
12.9B
python benchmark.py --n_layer 16 --n_head 16 --n_embd 8192 --gpus 8 --plugins deepspeed --precision 16 --limit_train_batches 120

Average Epoch time: 1700.02 seconds
Average Peak memory 35430.12MiB
```

#### Instantiating DeepSpeed

When modifying defaults, we have to specify the ``DeepSpeedPlugin`` as input, so I made the modification as such to the benchmark script and adjusted parameters when necessary:

```python
from pytorch_lightning.plugins import DeepSpeedPlugin

...
trainer = Trainer.from_argparse_args(
    args,
    max_epochs=1,
    gradient_clip_val=1.0,
    plugins=[DeepSpeedPlugin(zero_optimization=False)], # Pass in my own custom deepspeed plugin to turn off ZeRO-Offload
    callbacks=[lr_decay, CUDACallback()],
)
```

### References

Code:

- [openai/gpt-2](https://github.com/openai/gpt-2) has the model but not the training code, and in TensorFlow
- [openai/image-gpt](https://github.com/openai/image-gpt) has some more modern gpt-3 like modification in its code, good reference as well
- huggingface/transformers has a [language-modeling example](https://github.com/huggingface/transformers/tree/master/examples/language-modeling). It is full-featured but as a result also somewhat challenging to trace. E.g. some large functions have as much as 90% unused code behind various branching statements that is unused in the default setting of simple language modeling.
- [Teddy Koker/image-gpu in PyTorch Lightning](https://github.com/teddykoker/image-gpt)

Papers + some implementation notes:

#### Improving Language Understanding by Generative Pre-Training (GPT-1)

- Our model largely follows the original transformer work
- We trained a 12-layer decoder-only transformer with masked self-attention heads (768 dimensional states and 12 attention heads). For the position-wise feed-forward networks, we used 3072 dimensional inner states.
- Adam max learning rate of 2.5e-4. (later GPT-3 for this model size uses 6e-4)
- LR decay: increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule
- We train for 100 epochs on minibatches of 64 randomly sampled, contiguous sequences of 512 tokens.
- Since layernorm is used extensively throughout the model, a simple weight initialization of N(0, 0.02) was sufficient
- bytepair encoding (BPE) vocabulary with 40,000 merges
- residual, embedding, and attention dropouts with a rate of 0.1 for regularization.
- modified version of L2 regularization proposed in (37), with w = 0.01 on all non bias or gain weights
- For the activation function, we used the Gaussian Error Linear Unit (GELU).
- We used learned position embeddings instead of the sinusoidal version proposed in the original work
- For finetuning: We add dropout to the classifier with a rate of 0.1. learning rate of 6.25e-5 and a batchsize of 32. 3 epochs. We use a linear learning rate decay schedule with warmup over 0.2% of training. λ was set to 0.5.
- GPT-1 model is 12 layers and d_model 768, ~117M params

#### Language Models are Unsupervised Multitask Learners (GPT-2)

- LayerNorm was moved to the input of each sub-block, similar to a pre-activation residual network
- an additional layer normalization was added after the final self-attention block.
- modified initialization which accounts for the accumulation on the residual path with model depth is used. We scale the weights of residual layers at initialization by a factor of 1/√N where N is the number of residual layers. (weird because in their released code i can only find a simple use of the old 0.02... in their release of image-gpt I found it used for c_proj, and even then only for attn, not for mlp. huh. https://github.com/openai/image-gpt/blob/master/src/model.py)
- the vocabulary is expanded to 50,257
- increase the context size from 512 to 1024 tokens
- larger batchsize of 512 is used
- GPT-2 used 48 layers and d_model 1600 (vs. original 12 layers and d_model 768). ~1.542B params

#### Language Models are Few-Shot Learners (GPT-3)

- GPT-3: 96 layers, 96 heads, with d_model of 12,288 (175B parameters).
- GPT-1-like: 12 layers, 12 heads, d_model 768 (125M)
- We use the same model and architecture as GPT-2, including the modified initialization, pre-normalization, and reversible tokenization described therein
- we use alternating dense and locally banded sparse attention patterns in the layers of the transformer, similar to the Sparse Transformer
- we always have the feedforward layer four times the size of the bottleneck layer, dff = 4 ∗ dmodel
- all models use a context window of nctx = 2048 tokens.
- Adam with β1 = 0.9, β2 = 0.95, and eps = 10−8
- All models use weight decay of 0.1 to provide a small amount of regularization. (NOTE: GPT-1 used 0.01 I believe, see above)
- clip the global norm of the gradient at 1.0
- Linear LR warmup over the first 375 million tokens. Then use cosine decay for learning rate down to 10% of its value, over 260 billion tokens.
- gradually increase the batch size linearly from a small value (32k tokens) to the full value over the first 4-12 billion tokens of training, depending on the model size.
- full 2048-sized time context window is always used, with a special END OF DOCUMENT token delimiter

### License

MIT
