import sys
import os

# Adding the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch, torch.nn as nn, numpy as np, math, pickle, time
import matplotlib
matplotlib.use('Agg')

import reps

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# =============================================================================
# SO(3) sampler — fully on-device, no CPU<->GPU transfers
# =============================================================================

EPS = 1e-6

def sample_unit_sphere_batch(n, device):
    """
    Muller's method: uniform unit vectors on S^2, allocated directly on device.
    All randn calls go to the target device so no host-device copies occur.
    """
    v     = torch.randn(n, 3, device=device)
    norms = v.norm(dim=1, keepdim=True)
    # Resample any degenerate vectors (astronomically rare) without leaving device
    bad = (norms < 1e-10).squeeze(1)
    while bad.any():
        v[bad]     = torch.randn(int(bad.sum()), 3, device=device)
        norms[bad] = v[bad].norm(dim=1, keepdim=True)
        bad        = (norms < 1e-10).squeeze(1)
    return v / norms


def expmap_so3_batch(u):
    """
    Rodrigues exponential map R^3 -> SO(3), fully on u.device.

    exp(K) = I + p*K + q*K^2
      p = sin(beta)/beta         (Taylor series near beta=0)
      q = (1-cos(beta))/beta^2  (Taylor series near beta=0)

    K is built with torch.stack (not in-place slice assignment) so the
    function is autograd-safe and works cleanly with CUDA async execution.
    """
    n   = u.shape[0]
    dev = u.device

    beta  = u.norm(dim=1)          # (n,)
    small = beta < EPS

    p = torch.where(small,
                    1 - beta**2/6  + beta**4/120,    # Taylor
                    torch.sin(beta) / beta)            # exact

    q = torch.where(small,
                    0.5 - beta**2/24 + beta**4/720,  # Taylor
                    (1 - torch.cos(beta)) / beta**2)  # exact

    a, b, c = u[:, 0], u[:, 1], u[:, 2]
    z = torch.zeros(n, device=dev)

    # Build skew-symmetric K via stack — no in-place writes
    K = torch.stack([
        torch.stack([ z, -c,  b], dim=1),
        torch.stack([ c,  z, -a], dim=1),
        torch.stack([-b,  a,  z], dim=1),
    ], dim=1)                                    # (n, 3, 3)

    K2 = torch.bmm(K, K)
    I  = torch.eye(3, device=dev).unsqueeze(0).expand(n, -1, -1)

    return I + p.view(n, 1, 1) * K + q.view(n, 1, 1) * K2   # (n, 3, 3)


def random_so3_batch(n, device=None):
    """
    Sample n rotation matrices uniformly from SO(3), entirely on `device`.
    Steps:
      1. Uniform axis on S^2 via Muller's method  (on device)
      2. Uniform angle beta ~ U[0, pi]            (on device)
      3. axis-angle vector u = axis * beta
      4. Rodrigues exponential map -> SO(3)
    """
    if device is None:
        device = torch.device('cpu')
    axes  = sample_unit_sphere_batch(n, device)           # (n, 3)
    betas = torch.rand(n, 1, device=device) * math.pi    # (n, 1)
    return expmap_so3_batch(axes * betas)                 # (n, 3, 3)


# =============================================================================
# Network
# =============================================================================

class EncoderMLP(nn.Module):
    def __init__(self, odim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9,   128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, odim),
        )
    def forward(self, x): return self.net(x)


def geodesic_deg(M, Mp):
    Mpp = M @ Mp.transpose(-1, -2)
    tr  = Mpp[:,0,0] + Mpp[:,1,1] + Mpp[:,2,2]
    return torch.acos(torch.clamp((tr-1)/2, -1+1e-6, 1-1e-6)) * 180 / math.pi

# =============================================================================
# Training
# =============================================================================

TOTAL     = 100
BS        = 64
LOG_EVERY = 10

# Larger eval batch — GPU can handle it in one shot
EVAL_BATCH = 10_000
EVAL_N     = 100_000

results = {}
for name, g_fn, f_fn, dim in reps.REPS:
    print(f"\n{'='*50}", flush=True)
    print(f"Training {name} ({dim}D output)...", flush=True)
    t0 = time.time()

    net = EncoderMLP(dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)

    # Use CUDA streams for async data generation + forward pass overlap
    # (no-op on CPU — torch.cuda.Stream() is a no-op wrapper there)
    mean_errors = []
    for it in range(1, TOTAL + 1):
        if it == 10_001:
            for pg in opt.param_groups:
                pg['lr'] = 1e-6

        # Sampling happens entirely on device — no PCIe transfer
        M  = random_so3_batch(BS, device)
        r  = net(M.reshape(BS, 9))
        Mp = f_fn(r)

        loss = geodesic_deg(M, Mp).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % LOG_EVERY == 0:
            # torch.no_grad eval on GPU — single device sync
            with torch.no_grad():
                err = geodesic_deg(M, Mp).mean().item()
            elapsed = time.time() - t0
            eta     = elapsed / it * (TOTAL - it)
            mean_errors.append((it, err))
            print(f"  [{name}] iter {it:>7d} | err {err:.3f}° | "
                  f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m", flush=True)

    # Evaluate: larger batches to keep GPU saturated
    net.eval()
    all_errs = []
    with torch.no_grad():
        for _ in range(EVAL_N // EVAL_BATCH):
            M  = random_so3_batch(EVAL_BATCH, device)
            Mp = f_fn(net(M.reshape(EVAL_BATCH, 9)))
            all_errs.append(geodesic_deg(M, Mp).cpu().numpy())
    final_errors = np.concatenate(all_errs)

    results[name] = {"mean_errors": mean_errors, "final_errors": final_errors}
    print(f"  [{name}] FINAL: mean={final_errors.mean():.2f}° "
          f"max={final_errors.max():.2f}° std={final_errors.std():.2f}°", flush=True)

    savefile = "./data/" + name + ".pkl"
    pickle.dump(results[name], open(savefile, "wb"))
    print("\nResults saved.", flush=True)

