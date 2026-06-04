"""
PCLN Stage 3 — Frozen LLM Test
Replaces ONE attention layer in GPT-2 with Percolative Message-Passing.
Measures: PPL change vs baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, json, os, sys
from scipy.sparse.csgraph import connected_components
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(OUT_DIR, "test_results_frozen_llm.json")

# ============================================================
class PercolativeMsgPassNoSkip(nn.Module):
    """Message-passing WITHOUT skip-connection (transformer adds residual)."""
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.d_head))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, adj):
        N, d = x.shape
        q = self.W_q(x).view(N, self.n_heads, self.d_head)
        k = self.W_k(x).view(N, self.n_heads, self.d_head)
        v = self.W_v(x).view(N, self.n_heads, self.d_head)

        edge_index = adj.nonzero(as_tuple=False)
        src, dst = edge_index[:, 0], edge_index[:, 1]
        E = len(src)

        if E == 0:
            return torch.zeros(N, d, device=x.device, dtype=x.dtype)

        edge_feat = torch.cat([q[src], k[dst]], dim=-1)
        e = (edge_feat * self.a.unsqueeze(0)).sum(dim=-1)
        e = self.leaky_relu(e)

        out = torch.zeros(N, self.n_heads, self.d_head, device=x.device, dtype=x.dtype)
        for h in range(self.n_heads):
            e_h = e[:, h]
            max_e = torch.zeros(N, device=x.device)
            max_e.scatter_reduce_(0, dst, e_h, reduce='amax', include_self=False)
            exp_e = torch.exp(e_h - max_e[dst])
            sum_exp = torch.zeros(N, device=x.device)
            sum_exp.scatter_add_(0, dst, exp_e)
            alpha = exp_e / (sum_exp[dst] + 1e-8)
            weighted_v = v[dst, h, :] * alpha.unsqueeze(-1)
            out[:, h, :].scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.d_head), weighted_v)

        return self.out_proj(out.reshape(N, d))


def build_percolation_graph(K_emb, t_factor=1.02):
    """Build percolation adjacency at Δ_c * t_factor."""
    dists = torch.cdist(K_emb, K_emb, p=2)

    # Find critical threshold
    for t in np.linspace(0.5, 2.5, 60):
        adj_test = (dists < float(t)).cpu().numpy()
        np.fill_diagonal(adj_test, 0)
        _, labels = connected_components(adj_test, directed=False)
        if np.bincount(labels).max() / len(K_emb) > 0.5:
            t_c = t
            break

    t_use = t_c * t_factor
    adj = (dists < float(t_use))
    adj.fill_diagonal_(False)
    return adj, t_c, t_use


# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")

    # ========== CONFIG ==========
    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    REPLACE_LAYER = 6     # Which attention layer to replace (0-indexed, 0-11 for GPT-2)
    TRAIN_STEPS = 400
    TRAIN_LR = 0.005
    T_FACTOR = 1.05       # Threshold multiplier above Δ_c

    # ========== LOAD MODEL ==========
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    n_layers = len(model.transformer.h) if hasattr(model, 'transformer') else len(model.model.layers)
    d_model = model.config.n_embd if hasattr(model.config, 'n_embd') else model.config.hidden_size
    n_heads = model.config.n_head if hasattr(model.config, 'n_head') else model.config.num_attention_heads

    REPLACE_LAYER = min(REPLACE_LAYER, n_layers // 2)  # middle layer
    print(f"  Layers: {n_layers}, d_model: {d_model}, heads: {n_heads}")
    print(f"  Replacing layer: {REPLACE_LAYER}")

    # ========== GET TARGET LAYER ==========
    if hasattr(model, 'transformer'):
        target_layer = model.transformer.h[REPLACE_LAYER]
    else:
        target_layer = model.model.layers[REPLACE_LAYER]

    test_texts = [
        "The capital of France is Paris. It is known for the Eiffel Tower",
        "Machine learning is a subset of artificial intelligence that focuses on",
        "In quantum physics, the Schrödinger equation describes how the quantum state",
        "The history of the Roman Empire spans over a thousand years, beginning with",
        "Deep neural networks have revolutionized computer vision by enabling",
    ]

    def compute_ppl(model, texts):
        """Compute perplexity on given texts."""
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt").to(device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].numel()
                total_tokens += inputs["input_ids"].numel()
        return np.exp(total_loss / total_tokens)

    # ========== BASELINE PPL ==========
    print("\nComputing baseline PPL...")
    t0 = time.time()
    baseline_ppl = compute_ppl(model, test_texts)
    print(f"  Baseline PPL: {baseline_ppl:.4f}  ({time.time()-t0:.1f}s)")

    # ========== CAPTURE TARGET LAYER OUTPUT ==========
    print(f"\nCapturing attention output at layer {REPLACE_LAYER}...")

    # Determine attention attribute name (differs between GPT-2 and Qwen)
    if hasattr(target_layer, 'attn'):
        attn_attr = 'attn'
        # GPT-2 uses c_proj
        has_c_proj = True
    elif hasattr(target_layer, 'self_attn'):
        attn_attr = 'self_attn'
        # Qwen uses o_proj
        has_c_proj = False
    else:
        raise AttributeError(f"Cannot find attention module in layer. Available: {dir(target_layer)}")

    attn_module = getattr(target_layer, attn_attr)

    # Hook to capture input and attention output
    captured_attn_out = None
    captured_input = None

    orig_attn_forward = attn_module.forward

    def hooked_attn(hidden_states, **kwargs):
        nonlocal captured_attn_out, captured_input
        # hidden_states: (B, T, d_model)
        captured_input = hidden_states.detach().clone()

        # Call original attention
        attn_output = orig_attn_forward(hidden_states, **kwargs)

        # We can't easily capture Q,K,V from GPT2Attention without patching deeper
        # So we use hidden_states (input to attention) as proxy for K
        captured_attn_out = attn_output[0].detach().clone() if isinstance(attn_output, tuple) else attn_output.detach().clone()
        return attn_output

    attn_module.forward = hooked_attn

    # Run one forward pass to capture
    sample_text = " ".join(test_texts)
    inputs = tokenizer(sample_text[:512], return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        _ = model(**inputs)

    # Restore
    attn_module.forward = orig_attn_forward

    B, T, D = captured_input.shape
    print(f"  Captured: B={B}, T={T}, d_model={D}")

    # Use captured_input as proxy for K embeddings (it goes into attention after LayerNorm)
    K_proxy = captured_input[0]  # (T, d_model)
    K_proxy = K_proxy / K_proxy.norm(dim=-1, keepdim=True)  # normalize

    # ========== BUILD PERCOLATION GRAPH ==========
    print("\nBuilding percolation graph...")
    adj, t_c, t_use = build_percolation_graph(K_proxy, t_factor=T_FACTOR)
    n_edges = adj.sum().item()
    sparsity = 1.0 - n_edges / (T * T)
    print(f"  Δ_c={t_c:.4f}, t_used={t_use:.4f}, edges={n_edges}/{T*T}, sparsity={sparsity:.2%}")

    # ========== TRAIN MESSAGE-PASSING ==========
    print(f"\nTraining message-passing ({TRAIN_STEPS} steps)...")
    mp_module = PercolativeMsgPassNoSkip(d_model, n_heads=n_heads).to(device)
    target_attn_out = captured_attn_out[0].float()  # (T, d_model), force float32
    x_train = K_proxy.float()  # force float32 for training

    opt = torch.optim.Adam(mp_module.parameters(), lr=TRAIN_LR)
    losses = []

    t0 = time.time()
    for step in range(TRAIN_STEPS):
        opt.zero_grad()
        pred = mp_module(x_train, adj)
        loss = F.mse_loss(pred, target_attn_out)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    train_time = time.time() - t0

    cos_final = F.cosine_similarity(pred, target_attn_out, dim=-1).mean().item()
    print(f"  Final loss: {losses[-1]:.6f}, cos_sim: {cos_final:.4f}, time: {train_time:.1f}s")

    # ========== REPLACE LAYER AND TEST ==========
    print(f"\nReplacing attention layer {REPLACE_LAYER} and measuring PPL...")

    # Determine output projection
    if has_c_proj:
        out_proj = attn_module.c_proj
    else:
        out_proj = attn_module.o_proj

    mp_module.eval()

    def percolative_forward(hidden_states, **kwargs):
        """Replacement forward: LayerNorm -> PercolativeMessagePassing -> out_proj"""
        x = hidden_states[0] if hidden_states.dim() == 3 else hidden_states
        T_cur, D_cur = x.shape

        x_norm = x.float()  # cdist requires float32 on CUDA
        x_norm = x_norm / (x_norm.norm(dim=-1, keepdim=True) + 1e-8)
        dists_cur = torch.cdist(x_norm, x_norm, p=2)
        adj_cur = (dists_cur < float(t_use))
        adj_cur.fill_diagonal_(False)

        mp_out = mp_module(x_norm, adj_cur).to(x.dtype)  # match model dtype
        attn_output = out_proj(mp_out)

        return (attn_output, None)

    # Replace
    orig_forward = attn_module.forward
    attn_module.forward = percolative_forward

    # Measure PPL
    t0 = time.time()
    pcln_ppl = compute_ppl(model, test_texts)
    ppl_time = time.time() - t0

    # Restore
    attn_module.forward = orig_forward

    delta_ppl = (pcln_ppl - baseline_ppl) / baseline_ppl * 100
    print(f"  PCLN PPL:   {pcln_ppl:.4f}  (Δ={delta_ppl:+.1f}%)")
    print(f"  Baseline:   {baseline_ppl:.4f}")

    # ========== VERDICT ==========
    print("\n" + "="*60)
    print("VERDICT")
    print("="*60)

    if delta_ppl < 5:
        verdict = "✅ PCLN VIABLE — PPL degradation < 5%"
    elif delta_ppl < 15:
        verdict = "⚠️ PCLN DEGRADES but may be trainable"
    elif delta_ppl < 50:
        verdict = "⚠️ SIGNIFICANT degradation — needs better training or more layers"
    else:
        verdict = "❌ PCLN FAILS on real LLM — catastrophic PPL explosion"

    print(f"  {verdict}")
    print(f"  Baseline PPL: {baseline_ppl:.2f}")
    print(f"  PCLN PPL:     {pcln_ppl:.2f}")
    print(f"  Δ:            {delta_ppl:+.1f}%")

    # ========== SAVE ==========
    results = {
        "model": MODEL_NAME,
        "replace_layer": REPLACE_LAYER,
        "n_layers": n_layers, "d_model": d_model, "n_heads": n_heads,
        "T": T, "t_critical": float(t_c), "t_used": float(t_use),
        "edges": n_edges, "sparsity": sparsity,
        "train_steps": TRAIN_STEPS, "train_lr": TRAIN_LR,
        "train_final_loss": losses[-1],
        "train_cos_sim": cos_final,
        "baseline_ppl": baseline_ppl,
        "pcln_ppl": pcln_ppl,
        "delta_ppl_pct": delta_ppl,
        "verdict": verdict,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {RESULTS_PATH}")

if __name__ == "__main__":
    main()
