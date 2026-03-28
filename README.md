# Julia microgpt

A dependency-free Julia port of [@karpathy](https://github.com/karpathy)'s [microgpt.py](https://github.com/karpathy/makemore).

The most atomic way to train and run inference for a GPT in pure Julia. No shortcuts. Every operation is tracked as a scalar `Value` node through a hand-rolled autograd engine.

## What it does
Trains a small character-level GPT on a list of names, then generates new hallucinated names at inference time. The entire algorithm: dataset, tokeniser, autograd, transformer, optimiser, inference, lives in a single file.

## Requirements
- Julia 1.9+
- Standard library only (`Random`, `Downloads`) — no external packages needed for training
- `BenchmarkTools.jl` for benchmarking only (optional)

To install BenchmarkTools:

```
] add BenchmarkTools
```

## Usage
### Run training + inference
```bash
julia microgpt.jl
```

The script will:

1. Download `names.txt` automatically on first run
2. Train for 1000 steps, printing loss at each step
3. Generate 20 hallucinated names at inference time
4. 

### Expected output
```
num docs: 32033
vocab size: 27
num params: 4192
step 1000 / 1000 | loss 2.786
--- inference (new, hallucinated names) ---
sample  1: jarare
sample  2: charil
...
```

## How it works
The implementation follows GPT-2's architecture with a few simplifications:

**Autograd**: a `Value` struct wraps every scalar, storing its data, gradient, children, and local gradients. `backward!()` performs a topological sort of the computation graph and applies the chain rule in reverse.

**Tokenizer**: character-level. Each unique character in the dataset gets a token ID. A special `BOS` token marks the beginning and end of each name.

**Transformer**: a single-layer GPT with:
- Token + positional embeddings
- RMSNorm (instead of LayerNorm)
- Multi-head causal self-attention with KV cache
- MLP block with ReLU (instead of GeLU)
- No biases anywhere

**Optimizer**: Adam with linear learning rate decay.

**Inference**: temperature-scaled softmax with weighted categorical sampling, implemented without any dependencies.

### Architecture hyperparameters
|Parameter|Value|
|---|---|
|Layers|1|
|Embedding dim|16|
|Attention heads|4|
|Head dim|4|
|Block size|16|
|Total params|5408|

## Benchmarks
Measured on Apple MacBook Pro (M-series), full training loop of 1000 steps:

|Implementation|Total training time|Single forward pass|
|---|---|---|
|Python (microgpt.py)|61.5s|1.5ms|
|Julia (microgpt.jl)|37.1s|2.9ms|
|**Speedup**|**~1.7x**|—|

The single forward pass microbenchmark is misleading here, it only tests one token with an empty KV cache and no backward graph. The full training loop (forward + backward + Adam over 5000+ params) is the meaningful comparison, and Julia wins by ~1.7x.

Julia benchmark run with `@time`, Python with `time.perf_counter()`.
## File structure

```
microgpt.jl     # the entire algorithm, one file
input.txt       # auto-downloaded on first run
```

---

## Credits
Based on [microgpt.py](https://github.com/karpathy/makemore) by Andrej Karpathy. Julia port by Vuk.
