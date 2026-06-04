"""
Percolative Attention — Stage 2: Message-Passing on Percolation Graph
Compares: Full Attention vs Masked Softmax vs Graph Message-Passing

KEY FIXES from v1:
- Skip-connection: out = x + mp(x, adj)
- Fast training (100 Adam steps) to learn attention on percolation graph
- No dense graphs (max 200k edges to avoid OOM)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time, json, os
from scipy.sparse.csgraph import connected_components

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(OUT_DIR, "test_results_msgpass.json")

# ============================================================
class PercolativeMessagePassing(nn.Module):
    """GAT-style message passing on percolation graph + skip connection."""
    def __init__(self, d_model, n_heads=4, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * self.d_head))
        nn.init.xavier_uniform_(self.a)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
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

        # Edge features
        edge_feat = torch.cat([q[src], k[dst]], dim=-1)  # (E, H, 2*d_h)
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
            alpha = self.dropout(alpha)
            weighted_v = v[dst, h, :] * alpha.unsqueeze(-1)
            out[:, h, :].scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.d_head), weighted_v)

        out = out.reshape(N, d)
        return self.out_proj(out) + x  # SKIP CONNECTION


def full_attention(Q, K, V, scale=None):
    N, H, d_h = Q.shape
    if scale is None:
        scale = d_h ** 0.5
    scores = torch.einsum('nhd,mhd->hnm', Q, K) / scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum('hnm,mhd->nhd', attn, V)
    return out, attn


def masked_softmax_attention(Q, K, V, adj, scale=None):
    N, H, d_h = Q.shape
    if scale is None:
        scale = d_h ** 0.5
    scores = torch.einsum('nhd,mhd->hnm', Q, K) / scale
    mask = ~adj.unsqueeze(0).expand(H, -1, -1)
    scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0)
    out = torch.einsum('hnm,mhd->nhd', attn, V)
    return out, attn


def train_mp_module(mp_module, x, adj, out_full_flat, steps=100, lr=0.01):
    """Quick training: learn to reconstruct full attention output via message-passing."""
    opt = torch.optim.Adam(mp_module.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        opt.zero_grad()
        out_pred = mp_module(x, adj)
        loss = F.mse_loss(out_pred, out_full_flat)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses[-1] if losses else float('inf')


def top20_overlap(attn_a, attn_b, n_heads):
    """Vectorized top-20 overlap between two attention maps (H, N, N)."""
    _, top_a = attn_a.topk(20, dim=-1)  # (H, N, 20)
    _, top_b = attn_b.topk(20, dim=-1)  # (H, N, 20)
    overlap_sum = 0.0
    for h in range(n_heads):
        idx_a = top_a[h]  # (N, 20)
        idx_b = top_b[h]  # (N, 20)
        match = (idx_a.unsqueeze(-1) == idx_b.unsqueeze(1)).any(dim=-1)
        overlap_sum += match.float().mean().item()
    return overlap_sum / n_heads


# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    N = 2048
    d_model = 256
    n_heads = 4
    d_head = d_model // n_heads
    train_steps = 100
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ============ Build Percolation Graph ============
    print("\n" + "="*60)
    print("BUILDING PERCOLATION GRAPH")
    print("="*60)

    K_emb = torch.randn(N, d_model, device=device)
    K_emb = K_emb / K_emb.norm(dim=-1, keepdim=True)

    t0 = time.time()
    dists = torch.cdist(K_emb, K_emb, p=2)
    torch.cuda.synchronize()
    print(f"  Distances: {time.time()-t0:.3f}s")

    # Find critical threshold
    for t in np.linspace(0.8, 1.8, 50):
        adj_test = (dists < float(t)).cpu().numpy()
        np.fill_diagonal(adj_test, 0)
        _, labels = connected_components(adj_test, directed=False)
        if np.bincount(labels).max() / N > 0.5:
            t_critical = t
            break
    print(f"  Δ_c ≈ {t_critical:.4f}")

    # Build graphs — skip dense ones (>200k edges)
    graph_configs = [
        ('t=Δ_c-0.02', t_critical - 0.02),
        ('t=Δ_c',      t_critical),
        ('t=Δ_c+0.02', t_critical + 0.02),
        ('t=Δ_c+0.05', t_critical + 0.05),
    ]

    graphs = {}
    for name, t_val in graph_configs:
        adj = (dists < float(t_val))
        adj.fill_diagonal_(False)
        n_edges = adj.sum().item()
        if n_edges > 200000:
            print(f"  {name}: t={t_val:.4f} — SKIPPED (edges={n_edges} > 200k)")
            continue
        graphs[name] = {
            'adj': adj, 't_val': t_val,
            'edges': n_edges,
            'sparsity': 1.0 - n_edges / (N * N),
            'avg_deg': n_edges / N,
        }
        print(f"  {name}: t={t_val:.4f}, edges={n_edges}, "
              f"sparsity={graphs[name]['sparsity']:.2%}, avg_deg={graphs[name]['avg_deg']:.1f}")

    # ============ Input Features ============
    x = torch.randn(N, d_model, device=device)
    Q = x.view(N, n_heads, d_head)
    K = x.view(N, n_heads, d_head)
    V = x.view(N, n_heads, d_head)
    scale = d_head ** 0.5

    # Full attention baseline
    torch.cuda.synchronize()
    t0 = time.time()
    out_full, attn_full = full_attention(Q, K, V, scale)
    torch.cuda.synchronize()
    t_full = time.time() - t0
    out_full_flat = out_full.reshape(N, d_model)
    print(f"\n  Full Attention: {t_full:.4f}s\n")

    # ============ Compare Methods ============
    results = []

    for graph_name, g in graphs.items():
        adj = g['adj']
        sparsity = g['sparsity']
        print(f"  {graph_name} (sparsity={sparsity:.2%}):")

        # --- Masked Softmax ---
        torch.cuda.synchronize()
        t0 = time.time()
        out_masked, attn_masked = masked_softmax_attention(Q, K, V, adj, scale)
        torch.cuda.synchronize()
        t_masked = time.time() - t0
        out_masked_flat = out_masked.reshape(N, d_model)

        cos_masked = F.cosine_similarity(out_full_flat, out_masked_flat, dim=-1).mean().item()
        overlap_masked = top20_overlap(attn_full, attn_masked, n_heads)
        print(f"    Masked:   cos={cos_masked:.4f}, top20={overlap_masked:.2%}, time={t_masked:.4f}s")

        # --- Message-Passing (fresh module, trained) ---
        mp_module = PercolativeMessagePassing(d_model, n_heads=n_heads).to(device)

        # Train
        t_train_start = time.time()
        final_loss = train_mp_module(mp_module, x, adj, out_full_flat, steps=train_steps, lr=0.01)
        t_train = time.time() - t_train_start

        # Inference
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out_mp = mp_module(x, adj)
        torch.cuda.synchronize()
        t_mp = time.time() - t0

        cos_mp = F.cosine_similarity(out_full_flat, out_mp, dim=-1).mean().item()
        print(f"    MsgPass:  cos={cos_mp:.4f}, time={t_mp:.4f}s, train={t_train:.2f}s, final_loss={final_loss:.6f}")

        results.append({
            'graph': graph_name, 't_value': g['t_val'],
            'edges': g['edges'], 'sparsity': sparsity, 'avg_deg': g['avg_deg'],
            'masked_cos_sim': cos_masked, 'masked_top20_overlap': overlap_masked,
            'masked_time': t_masked,
            'mp_cos_sim': cos_mp, 'mp_time': t_mp,
            'mp_train_time': t_train, 'mp_final_loss': final_loss,
        })

    # ============ Summary ============
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for r in results:
        gain = r['mp_cos_sim'] - r['masked_cos_sim']
        print(f"  {r['graph']:15s}  masked={r['masked_cos_sim']:.4f}  "
              f"msgpass={r['mp_cos_sim']:.4f}  Δ={gain:+.4f}  "
              f"sparsity={r['sparsity']:.2%}")

    # Verdict
    best_mp = max(results, key=lambda r: r['mp_cos_sim'])
    print(f"\n  Best message-passing: {best_mp['graph']} "
          f"(cos={best_mp['mp_cos_sim']:.4f}, sparsity={best_mp['sparsity']:.2%})")

    if best_mp['mp_cos_sim'] > best_mp['masked_cos_sim'] + 0.05:
        print(f"  ✅ Message-Passing BEATS masked softmax!")
    elif best_mp['mp_cos_sim'] > 0.5:
        print(f"  ⚠️ Both work, message-passing viable at high sparsity")
    else:
        print(f"  ❌ Even trained message-passing can't match full attention at this sparsity")

    # Save
    output = {
        'device': str(device), 'N': N, 'd_model': d_model, 'n_heads': n_heads,
        't_critical': float(t_critical), 't_full': t_full, 'train_steps': train_steps,
        'results': results,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n[SAVED] {RESULTS_PATH}")

if __name__ == "__main__":
    main()
