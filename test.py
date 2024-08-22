import torch
import sentencepiece
import time
from typing import Optional
from torch import nn, Tensor
from dataclasses import dataclass
from torch.nn import functional as F


device = 'cuda' if torch.cuda.is_available() else 'cpu'
max_new_tokens = 500
transformer_configs = {"7B": dict(n_layer=32, n_head=32, dim=4096)}


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def sample(logits, temperature: float = 1.0):
    logits = logits[0,-1] / max(temperature, 1e-5)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    idx_next = torch.argmax(probs / torch.empty_like(probs).exponential_(1), dim=-1, keepdim=True).to(dtype=torch.int)
    return idx_next, probs


@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str):
        config = [config for config in transformer_configs if config.lower() in str(name).lower()]
        if len(config) > 1:
            config.sort(key=len, reverse=True)
        return cls(**transformer_configs[config[0]])


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (max_batch_size, n_heads, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        assert input_pos.shape[0] == k_val.shape[2]

        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.max_batch_size = -1
        self.max_seq_length = -1

    def setup_caches(self, max_batch_size, max_seq_length):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        dtype = self.output.weight.dtype
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype)
        self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        mask = self.causal_mask[None, None, input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)

        for i, layer in enumerate(self.layers):
            x = layer(x, input_pos, freqs_cis, mask)
        x = self.norm(x)
        logits = self.output(x)
        return logits

    @classmethod
    def from_name(cls, name: str):
        return cls(ModelArgs.from_name(name))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))

        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)

        k = k.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        v = v.repeat_interleave(self.n_head // self.n_local_heads, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        return y


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000,
    dtype: torch.dtype = torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


class tokentoolkit():
    def __init__(self, path):
        self.tokenizer = sentencepiece.SentencePieceProcessor(path)

    def encode(self, prompt):
        tokens = [self.tokenizer.bos_id()] + self.tokenizer.EncodeAsIds(prompt)
        return torch.tensor(tokens, dtype=int, device=device)
    
    def decode(self, tokens):
        output = self.tokenizer.DecodeIds(tokens.tolist())
        return output


def load_model(path):
    with torch.device('meta'):
        model = Transformer.from_name(path)
    checkpoint = torch.load(path +'/model.pth', mmap=True, weights_only=True)
    model.load_state_dict(checkpoint, assign=True)
    model.to(device=device, dtype=torch.bfloat16)
    return model


def generate_once(model, current_token, input_pos):
    new_token = model(current_token, input_pos)
    return sample(new_token)


@torch.no_grad()
def generate(model:Transformer, encoded:torch.Tensor):
    input_size = encoded.size(0)
    max_seq_length = input_size + max_new_tokens
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=max_seq_length)

    sequence = torch.empty(max_seq_length, dtype=int, device=device)
    sequence[:input_size] = encoded
    input_pos = torch.arange(0, input_size, device=device)

    next_token = sample(model(encoded.view(1,-1),input_pos))[0].clone()
    sequence[input_size] = next_token
    input_pos = torch.tensor([input_size], dtype=torch.int, device=device)

    for i in range(1, max_new_tokens):
        with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
            current_token = next_token.view(1,-1)
            next_token, _ = generate_once(model, current_token, input_pos)
            input_pos += 1
            sequence[input_size + i] = next_token.clone()

    return sequence


def speedup():
    global generate_once
    generate_once = torch.compile(generate_once, mode="reduce-overhead", fullgraph=True)


def main(prompt='Hello, my name is', setting=2, output=False):
    if setting not in [0,1,2]:
        print('setting error!')
        return
    
    print(f'Using device={device}')
    path = './checkpoints/meta-llama/Llama-2-7b-chat-hf'

    tokenizer = tokentoolkit(path + '/tokenizer.model')
    encoded = tokenizer.encode(prompt)

    t0 = time.time()
    model = load_model(path)
    torch.cuda.synchronize(device)
    t1 = time.time()
    print(f'Time to load model: {t1-t0:.02f} seconds')

    if setting != 0:
        tokens = generate(model, encoded)
        out = tokenizer.decode(tokens)
        if output:
            print(out)
        t2 = time.time()
        print(f'Time to generate without compilation: {t2-t1:.02f} seconds')
        print(f'tokens per second: {tokens.size(0) / (t2-t1):.02f} per second')
    else:
        t2 = t1

    if setting == 1: 
        return

    speedup()
    torch.cuda.synchronize(device)
    generate(model, encoded)
    t3 = time.time()
    print(f'Time to compile: {t3-t2:.02f} seconds')

    tokens = generate(model, encoded)
    out = tokenizer.decode(tokens)
    if output:
        print(out)
    t4 = time.time()
    print(f'Time to generate with compilation: {t4-t3:.02f} seconds')
    print(f'tokens per second: {tokens.size(0) / (t4-t3):.02f} per second')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompt', type=str, default='Hello, my name is')
    parser.add_argument('--setting', type=int, default=2)
    parser.add_argument('--output', type=bool, default=False)
    

    args = parser.parse_args()
    main(args.prompt, args.setting, args.output)