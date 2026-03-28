"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Julia.
This file is the complete algorithm.
Everything else is just efficiency.

Julia port of @karpathy's microgpt.py
"""

using BenchmarkTools

using Random, Downloads

Random.seed!(42) # Let there be order among chaos

# Dataset
input_file = "input.txt"
if !isfile(input_file)
    Downloads.download(
        "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt",
        input_file
    )
end
docs = [strip(line) for line in eachline(input_file) if !isempty(strip(line))]
shuffle!(docs)
println("num docs: $(length(docs))")

# Tokenizer
uchars  = sort(collect(Set(join(docs))))            # unique chars → token ids 0..n-1 (1-based in Julia)
BOS     = length(uchars) + 1                        # 1-based BOS token id
vocab_size = length(uchars) + 1
println("vocab size: $vocab_size")

char_index(ch) = findfirst(==(ch), uchars)          # char → 1-based token id

# Autograd — scalar Value with reverse-mode autodiff
mutable struct Value
    data        :: Float64
    grad        :: Float64
    children    :: Vector{Value}
    local_grads :: Vector{Float64}
end

Value(data::Real) = Value(Float64(data), 0.0, Value[], Float64[])
Value(data::Real, children, local_grads) =
    Value(Float64(data), 0.0, collect(Value, children), collect(Float64, local_grads))

# Operator overloads
import Base: +, -, *, /, ^, log, exp

+(a::Value, b::Value) = Value(a.data + b.data, (a, b), (1.0, 1.0))
+(a::Value, b::Real)  = a + Value(b)
+(a::Real,  b::Value) = Value(a) + b

*(a::Value, b::Value) = Value(a.data * b.data, (a, b), (b.data, a.data))
*(a::Value, b::Real)  = a * Value(b)
*(a::Real,  b::Value) = Value(a) * b

-(a::Value, b::Value) = a + (b * -1.0)
-(a::Value, b::Real)  = a + Value(-b)
-(a::Real,  b::Value) = Value(a) + (b * -1.0)
-(a::Value)           = a * -1.0

/(a::Value, b::Value) = a * b^(-1.0)
/(a::Value, b::Real)  = a * Value(b)^(-1.0)
/(a::Real,  b::Value) = Value(a) * b^(-1.0)

^(a::Value, b::Real)  = Value(a.data^b, (a,), (b * a.data^(b - 1),))

log(a::Value) = Value(log(a.data),  (a,), (1.0 / a.data,))
exp(a::Value) = Value(exp(a.data),  (a,), (exp(a.data),))
relu(a::Value) = Value(max(0.0, a.data), (a,), (Float64(a.data > 0),))

function backward!(root::Value)
    topo    = Value[]
    visited = IdSet{Value}()
    function build_topo!(v)
        if v ∉ visited
            push!(visited, v)
            for c in v.children; build_topo!(c); end
            push!(topo, v)
        end
    end
    build_topo!(root)
    root.grad = 1.0
    for v in reverse(topo)
        for (child, lg) in zip(v.children, v.local_grads)
            child.grad += lg * v.grad
        end
    end
end

# Model parameters
const n_layer    = 1
const n_embd     = 16
const block_size = 16
const n_head     = 4
const head_dim   = n_embd ÷ n_head

mat(nout, nin, std=0.08) =
    [[Value(randn() * std) for _ in 1:nin] for _ in 1:nout]

state_dict = Dict{String, Vector{Vector{Value}}}(
    "wte"     => mat(vocab_size, n_embd),
    "wpe"     => mat(block_size, n_embd),
    "lm_head" => mat(vocab_size, n_embd),
)
for i in 0:(n_layer-1)
    state_dict["layer$i.attn_wq"] = mat(n_embd, n_embd)
    state_dict["layer$i.attn_wk"] = mat(n_embd, n_embd)
    state_dict["layer$i.attn_wv"] = mat(n_embd, n_embd)
    state_dict["layer$i.attn_wo"] = mat(n_embd, n_embd)
    state_dict["layer$i.mlp_fc1"] = mat(4 * n_embd, n_embd)
    state_dict["layer$i.mlp_fc2"] = mat(n_embd, 4 * n_embd)
end
params = [p for mat in values(state_dict) for row in mat for p in row]
println("num params: $(length(params))")

# Model architecture
function linear(x::Vector{Value}, w::Vector{Vector{Value}})
    [sum(wi .* x) for wi in w]          # dot product of each output row with x
end

function softmax(logits::Vector{Value})
    max_val = maximum(l.data for l in logits)
    exps    = [(l - max_val) |> exp for l in logits]
    total   = sum(exps)
    [e / total for e in exps]
end

function rmsnorm(x::Vector{Value})
    n   = length(x)
    ms  = sum(xi * xi for xi in x) / n
    scl = (ms + 1e-5)^(-0.5)
    [xi * scl for xi in x]
end

function gpt(token_id::Int, pos_id::Int,
             keys::Vector{Vector{Vector{Value}}},
             values_kv::Vector{Vector{Vector{Value}}})

    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = [t + p for (t, p) in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in 0:(n_layer-1)
        # 1) Multi-head Attention
        x_res = x
        x = rmsnorm(x)
        q = linear(x, state_dict["layer$li.attn_wq"])
        k = linear(x, state_dict["layer$li.attn_wk"])
        v = linear(x, state_dict["layer$li.attn_wv"])
        push!(keys[li+1], k)
        push!(values_kv[li+1], v)

        x_attn = Value[]
        for h in 0:(n_head-1)
            hs   = h * head_dim
            q_h  = q[hs+1 : hs+head_dim]
            k_h  = [ki[hs+1:hs+head_dim] for ki in keys[li+1]]
            v_h  = [vi[hs+1:hs+head_dim] for vi in values_kv[li+1]]
            T    = length(k_h)
            attn_logits  = [sum(q_h[j] * k_h[t][j] for j in 1:head_dim) / sqrt(head_dim) for t in 1:T]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in 1:T) for j in 1:head_dim]
            append!(x_attn, head_out)
        end
        x = linear(x_attn, state_dict["layer$li.attn_wo"])
        x = [a + b for (a, b) in zip(x, x_res)]

        # 2) MLP block
        x_res = x
        x = rmsnorm(x)
        x = linear(x, state_dict["layer$li.mlp_fc1"])
        x = [relu(xi) for xi in x]
        x = linear(x, state_dict["layer$li.mlp_fc2"])
        x = [a + b for (a, b) in zip(x, x_res)]
    end

    linear(x, state_dict["lm_head"])
end

# Adam optimizer buffers
const learning_rate = 0.01
const beta1         = 0.85
const beta2         = 0.99
const eps_adam      = 1e-8

m_buf = zeros(Float64, length(params))
v_buf = zeros(Float64, length(params))

# Training loop
num_steps = 1000
training_start = time()

for step in 1:num_steps
    doc    = docs[mod1(step, length(docs))]
    tokens = [BOS; [char_index(ch) for ch in doc]; BOS]
    n      = min(block_size, length(tokens) - 1)

    keys_kv   = [Vector{Value}[] for _ in 1:n_layer]
    values_kv = [Vector{Value}[] for _ in 1:n_layer]
    losses    = Value[]

    for pos_id in 1:n
        token_id = tokens[pos_id]
        target_id = tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys_kv, values_kv)
        probs  = softmax(logits)
        push!(losses, -log(probs[target_id]))
    end

    loss = sum(losses) * (1.0 / n)
    backward!(loss)

    lr_t = learning_rate * (1 - (step - 1) / num_steps)
    for (i, p) in enumerate(params)
        m_buf[i] = beta1 * m_buf[i] + (1 - beta1) * p.grad
        v_buf[i] = beta2 * v_buf[i] + (1 - beta2) * p.grad^2
        m_hat    = m_buf[i] / (1 - beta1^step)
        v_hat    = v_buf[i] / (1 - beta2^step)
        p.data  -= lr_t * m_hat / (sqrt(v_hat) + eps_adam)
        p.grad   = 0.0
    end

    print("\rstep $(lpad(step, 4)) / $num_steps | loss $(round(loss.data, digits=4))")
end

# Inference
# Pure-Julia weighted categorical sampler (no dependencies)
function weighted_sample(weights::Vector{Float64})
    r = rand() * sum(weights)
    cumsum = 0.0
    for (i, w) in enumerate(weights)
        cumsum += w
        cumsum >= r && return i
    end
    return length(weights)
end

temperature = 0.5
println("\n--- inference (new, hallucinated names) ---")
for sample_idx in 1:20
    keys_kv   = [Vector{Value}[] for _ in 1:n_layer]
    values_kv = [Vector{Value}[] for _ in 1:n_layer]
    token_id  = BOS
    chars     = Char[]
    for pos_id in 1:block_size
        logits   = gpt(token_id, pos_id, keys_kv, values_kv)
        probs    = softmax([l * (1.0 / temperature) for l in logits])
        token_id = weighted_sample([p.data for p in probs])
        token_id == BOS && break
        push!(chars, uchars[token_id])
    end
    println("sample $(lpad(sample_idx, 2)): $(String(chars))")
end

keys_kv   = [Vector{Value}[] for _ in 1:n_layer]
values_kv = [Vector{Value}[] for _ in 1:n_layer]

gpt(BOS, 1, keys_kv, values_kv)  # warm-up

@btime gpt($BOS, 1, $keys_kv, $values_kv)

println("\nTotal training time: $(round(time() - training_start, digits=2))s")