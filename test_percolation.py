"""
Percolative Attention — Synthetic Test (Stage 1)
Tests: RGG critical point, giant component emergence, sparse attention validity.

GPU-accelerated via torch cuda.
"""
import torch
import numpy as np
import time, json, sys, os
from scipy.sparse.csgraph import connected_components

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(OUT_DIR, "test_results.json")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"DEVICE: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ============ CONFIG ============
    N = 2048          # tokens
    d = 128           # embedding dim
    n_thresholds = 50
    thresholds = np.linspace(0.05, 2.5, n_thresholds)
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ============ TEST 1: Percolation Threshold ============
    print("\n" + "="*60)
    print("TEST 1: PERCOLATION THRESHOLD ON RANDOM EMBEDDINGS")
    print("="*60)

    # Generate normalized embeddings (simulating real LLM embedding distribution)
    embeddings = torch.randn(N, d, device=device)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)

    # Compute all pairwise distances (GPU)
    t0 = time.time()
    dists = torch.cdist(embeddings, embeddings, p=2)
    torch.cuda.synchronize()
    print(f"  [GPU] Pairwise distances ({N}×{N}, d={d}): {time.time()-t0:.3f}s")

    # Find percolation thresholds
    giant_sizes = []
    edges_list = []
    for t in thresholds:
        adj = (dists < float(t)).cpu().numpy()
        np.fill_diagonal(adj, 0)
        n_edges = adj.sum()
        n_components, labels = connected_components(adj, directed=False)
        giant = np.bincount(labels).max()
        giant_sizes.append(giant / N)
        edges_list.append(n_edges)

    # Find critical point (giant > 0.5)
    critical_t = None
    for t, gs in zip(thresholds, giant_sizes):
        if gs > 0.5 and critical_t is None:
            critical_t = t
        marker = " <-- CRITICAL" if gs > 0.5 and abs(t - (critical_t or 0)) < 0.01 else ""
        print(f"  t={t:.4f}  giant={gs:.4f}  edges={edges_list[list(thresholds).index(t)]:>8d}{marker}")

    print(f"\n  Critical threshold Δ_c ≈ {critical_t:.4f}")

    # ============ TEST 2: Structure Preservation ============
    print("\n" + "="*60)
    print("TEST 2: ATTENTION STRUCTURE PRESERVATION UNDER PERCOLATION MASK")
    print("="*60)

    Q = torch.randn(N, d, device=device)
    K = torch.randn(N, d, device=device)
    V = torch.randn(N, d, device=device)

    # Full attention
    scale = d ** 0.5
    scores_full = torch.matmul(Q, K.T) / scale
    attn_full = torch.softmax(scores_full, dim=-1)
    out_full = torch.matmul(attn_full, V)

    # Percolative attention at different thresholds
    # Test attention quality at thresholds spanning critical region
    # t_c = 1.25 — use narrow window around it
    for t_val in [1.23, 1.26, 1.28, 1.30, 1.32, 1.35, 1.40]:
        adj_bool = (dists < float(t_val))
        # zero out diagonal
        adj_bool.fill_diagonal_(False)
        
        mask = adj_bool  # bool tensor

        n_edges = mask.sum().item()
        sparsity = 1.0 - n_edges / (N * N)

        # Masked attention
        masked_scores = scores_full.clone()
        masked_scores[~mask] = float('-inf')
        attn_sparse = torch.softmax(masked_scores, dim=-1)
        attn_sparse = torch.nan_to_num(attn_sparse)
        out_sparse = torch.matmul(attn_sparse, V)

        # Cosine similarity between full and sparse outputs (per token)
        cos_sims = torch.nn.functional.cosine_similarity(
            out_full, out_sparse, dim=-1
        )
        mean_cos = cos_sims.mean().item()

        # Top-K overlap in attention (top 20)
        _, topk_full = attn_full.topk(20, dim=-1)
        _, topk_sparse = attn_sparse.topk(20, dim=-1)
        overlap = sum(
            len(set(kf.tolist()) & set(ks.tolist())) / 20
            for kf, ks in zip(topk_full, topk_sparse)
        ) / N

        print(f"  t={t_val:.4f}  sparsity={sparsity:.2%}  "
              f"cos_sim={mean_cos:.4f}  top20_overlap={overlap:.2%}")

    # ============ TEST 3: Complexity Scaling ============
    print("\n" + "="*60)
    print("TEST 3: COMPLEXITY SCALING (theoretical O(N²) vs O(N log N))")
    print("="*60)

    Ns = [256, 512, 1024, 2048]
    times_full = []
    times_sparse_count = []

    for n_test in Ns:
        # Generate embeddings
        emb = torch.randn(n_test, d, device=device)
        emb = emb / emb.norm(dim=-1, keepdim=True)

        # Full attention timing
        Qs = torch.randn(n_test, d, device=device)
        Ks = torch.randn(n_test, d, device=device)
        Vs = torch.randn(n_test, d, device=device)

        torch.cuda.synchronize()
        t0 = time.time()
        scores = torch.matmul(Qs, Ks.T) / scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, Vs)
        torch.cuda.synchronize()
        t_full = time.time() - t0
        times_full.append(t_full)

        # Sparse count: build RGG mask at critical threshold
        dsts = torch.cdist(emb, emb, p=2)
        adj = (dsts < float(critical_t)).float()
        adj.fill_diagonal_(0)
        n_edges = adj.sum().item()

        torch.cuda.synchronize()
        t0 = time.time()
        scores_s = torch.matmul(Qs, Ks.T) / scale
        masked = scores_s.masked_fill(adj == 0, float('-inf'))
        attn_s = torch.softmax(masked, dim=-1)
        attn_s = torch.nan_to_num(attn_s)
        out_s = torch.matmul(attn_s, Vs)
        torch.cuda.synchronize()
        t_sparse = time.time() - t0
        times_sparse_count.append(t_sparse)

        # Theoretical: edges should scale as N·k where k ≈ avg degree
        avg_degree = n_edges / n_test
        print(f"  N={n_test:>5d}  full={t_full:.4f}s  masked={t_sparse:.4f}s  "
              f"edges={n_edges:>10.0f}  avg_deg={avg_degree:.1f}")

    # ============ SAVE RESULTS ============
    results = {
        "device": str(device),
        "N": N, "d": d, "seed": seed,
        "critical_threshold": float(critical_t),
        "giant_sizes": [float(g) for g in giant_sizes],
        "thresholds": [float(t) for t in thresholds],
        "edges_at_critical": int(edges_list[list(thresholds).index(critical_t)]),
        "scaling": {
            "N": Ns,
            "times_full": [float(t) for t in times_full],
            "times_masked": [float(t) for t in times_sparse_count],
        }
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {RESULTS_PATH}")

    # ============ SUMMARY ============
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Δ_c        = {critical_t:.4f}")
    print(f"  Edges @Δ_c = {results['edges_at_critical']} "
          f"({results['edges_at_critical']/(N*N):.2%} of N²)")
    print(f"  Verdict    = {'✅ CRITICAL POINT FOUND' if critical_t else '❌ NO CRITICAL POINT'}")

if __name__ == "__main__":
    main()
