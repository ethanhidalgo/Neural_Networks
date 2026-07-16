"""
Section 5.2 — Pose Estimation for 3D Point Clouds
===================================================
Siamese PointNet network trained to estimate the rotation between
a reference point cloud and a randomly-rotated target point cloud.

Architecture (from paper):
  - Shared PointNet encoder Φ : R^(N×3) → R^1024
      Conv1d(3→64) → LeakyReLU
      Conv1d(64→128) → LeakyReLU
      Conv1d(128→1024) → LeakyReLU
      AdaptiveMaxPool1d → R^1024
  - Concatenate z_r and z_t → R^2048
  - MLP decoder: 2048 → 512 → 512 → D
  - Fixed mapping f: R^D → SO(3)

Training:
  - 2,290 airplane .pts point clouds in ./points/
  - At each iteration: pick one reference cloud, apply 10 random
    rotations to get 10 target clouds (batch size = 10)
  - Loss: L2 between predicted and ground-truth rotation matrices
  - 2,600,000 iterations

Evaluation:
  - 400 test point clouds in ./points_test/
  - Each augmented with 100 random rotations → 40,000 test samples
  - Metric: geodesic error in degrees

Representations: 6D, Quaternion, Axis-angle, Euler

Output: pointcloud_results.png  (3-panel figure matching sanity-test style)
        pointcloud_results.pkl
"""

import os, glob, math, time, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# =============================================================================
# 1.  Point cloud loader
# =============================================================================

def load_pts(path):
    """Load a .pts or .xyz file (plain x y z per line, no header).
    Returns (N, 3) float32 numpy array."""
    pts = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue   # skip header lines if any
    return np.array(pts, dtype=np.float32)


def load_pointcloud_folder(folder):
    """Return list of (N,3) float32 arrays for every .pts/.xyz file in folder."""
    files = []
    for ext in ("*.pts", "*.xyz"):
        files.extend(glob.glob(os.path.join(folder, ext)))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .pts/.xyz files found in {folder!r}")
    clouds = [load_pts(f) for f in files]
    print(f"  Loaded {len(clouds)} point clouds from {folder}", flush=True)
    return clouds

# =============================================================================
# 2.  Random SO(3) sampling (axis-angle, no cuda dependency)
# =============================================================================

def random_so3_batch(n, dev=None):
    """
    Sample n rotation matrices uniformly from SO(3), entirely on `dev`.
    Uses Muller's method for a uniform axis on S^2 then a uniform angle in
    [0, pi] — all tensors allocated directly on the target device so there
    is no CPU->GPU transfer per call.
    """
    if dev is None:
        dev = torch.device('cpu')
    # Uniform axis via Muller's method: isotropic Gaussian + normalize
    axes  = torch.randn(n, 3, device=dev)
    axes  = axes / axes.norm(dim=1, keepdim=True).clamp(min=1e-8)
    # Uniform angle in [0, pi]
    theta = torch.rand(n, device=dev) * math.pi
    sin   = torch.sin(theta)
    qw = torch.cos(theta)
    qx = axes[:, 0] * sin
    qy = axes[:, 1] * sin
    qz = axes[:, 2] * sin
    xx=(qx*qx).view(n,1); yy=(qy*qy).view(n,1); zz=(qz*qz).view(n,1)
    xy=(qx*qy).view(n,1); xz=(qx*qz).view(n,1); yz=(qy*qz).view(n,1)
    xw=(qx*qw).view(n,1); yw=(qy*qw).view(n,1); zw=(qz*qw).view(n,1)
    r0 = torch.cat((1-2*yy-2*zz, 2*xy-2*zw, 2*xz+2*yw), 1)
    r1 = torch.cat((2*xy+2*zw, 1-2*xx-2*zz, 2*yz-2*xw), 1)
    r2 = torch.cat((2*xz-2*yw, 2*yz+2*xw, 1-2*xx-2*yy), 1)
    return torch.cat((r0.view(n,1,3), r1.view(n,1,3), r2.view(n,1,3)), 1)  # (n,3,3)

# =============================================================================
# Rotation representations
# =============================================================================

def g_6d(M):
    return M[:, :, :2].reshape(-1, 6)

def f_6d(r):
    a1, a2 = r[:, 0:3], r[:, 3:6]
    b1 = nn.functional.normalize(a1, dim=1)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)

def g_quat(M):
    B, dev = M.shape[0], M.device
    t    = M[:,0,0] + M[:,1,1] + M[:,2,2] + 1
    q_nz = torch.stack([M[:,2,1]-M[:,1,2], M[:,0,2]-M[:,2,0],
                         M[:,1,0]-M[:,0,1], t], dim=1)
    sgn32 = torch.sign(M[:,2,1])
    sgn32 = torch.where(sgn32 == 0, torch.ones_like(sgn32), sgn32)
    def ci(i):
        v = M[:,i,0] + M[:,0,i]
        return torch.where(v > 0,  torch.ones(B, device=dev),
               torch.where(v < 0, -torch.ones(B, device=dev), sgn32**(i+1)))
    q_z = torch.stack([
        torch.sqrt(torch.clamp(M[:,0,0]+1, min=1e-8)),
        ci(1)*torch.sqrt(torch.clamp(M[:,1,1]+1, min=1e-8)),
        ci(2)*torch.sqrt(torch.clamp(M[:,2,2]+1, min=1e-8)),
        torch.zeros(B, device=dev)], dim=1)
    return torch.where((t.abs() > 1e-7).unsqueeze(1), q_nz, q_z)

def f_quat(q):
    q = nn.functional.normalize(q, dim=1)
    x, y, z, w = q[:,0], q[:,1], q[:,2], q[:,3]
    R = torch.zeros(q.shape[0], 3, 3, device=q.device)
    R[:,0,0]=1-2*y*y-2*z*z; R[:,0,1]=2*x*y-2*z*w; R[:,0,2]=2*x*z+2*y*w
    R[:,1,0]=2*x*y+2*z*w;   R[:,1,1]=1-2*x*x-2*z*z; R[:,1,2]=2*y*z-2*x*w
    R[:,2,0]=2*x*z-2*y*w;   R[:,2,1]=2*y*z+2*x*w;   R[:,2,2]=1-2*x*x-2*y*y
    return R

def g_axisangle(M):
    trace = M[:,0,0]+M[:,1,1]+M[:,2,2]
    theta = torch.acos(torch.clamp((trace-1)/2, -1+1e-6, 1-1e-6))
    ax    = torch.stack([M[:,2,1]-M[:,1,2], M[:,0,2]-M[:,2,0], M[:,1,0]-M[:,0,1]], dim=1)
    ax    = ax / (2*torch.sin(theta).unsqueeze(1) + 1e-8)
    return ax * theta.unsqueeze(1)

def f_axisangle(v):
    theta = v.norm(dim=1, keepdim=True).clamp(min=1e-8)
    ax = v / theta; theta = theta.squeeze(1)
    c, s, t = torch.cos(theta), torch.sin(theta), 1 - torch.cos(theta)
    x, y, z = ax[:,0], ax[:,1], ax[:,2]
    R = torch.zeros(v.shape[0], 3, 3, device=v.device)
    R[:,0,0]=t*x*x+c; R[:,0,1]=t*x*y-s*z; R[:,0,2]=t*x*z+s*y
    R[:,1,0]=t*x*y+s*z; R[:,1,1]=t*y*y+c; R[:,1,2]=t*y*z-s*x
    R[:,2,0]=t*x*z-s*y; R[:,2,1]=t*y*z+s*x; R[:,2,2]=t*z*z+c
    return R

def g_euler(M):
    sy  = torch.sqrt(M[:,0,0]**2 + M[:,1,0]**2)
    sg  = sy < 1e-6
    x   = torch.atan2(M[:,2,1], M[:,2,2])
    y   = torch.atan2(-M[:,2,0], sy)
    z   = torch.atan2(M[:,1,0], M[:,0,0])
    xs  = torch.atan2(-M[:,1,2], M[:,1,1])
    return torch.stack([torch.where(sg, xs, x), y,
                        torch.where(sg, torch.zeros_like(z), z)], dim=1)

def f_euler(e):
    cx, sx = torch.cos(e[:,0]), torch.sin(e[:,0])
    cy, sy = torch.cos(e[:,1]), torch.sin(e[:,1])
    cz, sz = torch.cos(e[:,2]), torch.sin(e[:,2])
    R = torch.zeros(e.shape[0], 3, 3, device=e.device)
    R[:,0,0]=cy*cz; R[:,0,1]=cz*sx*sy-cx*sz; R[:,0,2]=cx*cz*sy+sx*sz
    R[:,1,0]=cy*sz; R[:,1,1]=cx*cz+sx*sy*sz;  R[:,1,2]=cx*sy*sz-cz*sx
    R[:,2,0]=-sy;   R[:,2,1]=cy*sx;            R[:,2,2]=cx*cy
    return R


def g_svd(M):
    """(B,3,3) -> (B,9): flatten the matrix (identity mapping into representation space)."""
    return M.reshape(-1, 9)

def f_svd(r):
    """
    SVDO+(M) from Eq. 2: projects a 9D vector onto SO(3).
      1. Reshape to (B,3,3)
      2. SVD: M = U Sigma V^T
      3. Replace singular values with diag(1,...,1, det(UV^T))
         so the result is guaranteed to be in SO(3) (det=+1).
      4. Return U Sigma' V^T
    """
    B = r.shape[0]
    M = r.reshape(B, 3, 3)
    U, _, Vh = torch.linalg.svd(M)          # U: (B,3,3), Vh: (B,3,3)
    # det(UV^T) is +1 or -1; multiply last column of U to enforce det=+1
    det = torch.linalg.det(U @ Vh)          # (B,)
    # Build Sigma': diag(1, 1, det(UV^T))
    sigma_prime = torch.ones(B, 3, device=r.device)
    sigma_prime[:, 2] = det                  # last singular value = ±1
    # U @ diag(sigma_prime) @ Vh
    R = U * sigma_prime.unsqueeze(1) @ Vh   # (B,3,3)
    return R

REPS = [
    ("6D",         g_6d,        f_6d,        6),
    ("Quaternion", g_quat,      f_quat,      4),
    ("Axis-angle", g_axisangle, f_axisangle, 3),
    ("Euler",      g_euler,     f_euler,     3),
    ("SVD",        g_svd,       f_svd,       9),
]

# =============================================================================
# 4.  Network
# =============================================================================

class PointNetEncoder(nn.Module):
    """Shared Φ: R^(N×3) → R^1024  (per paper: 3×64×128×1024 + maxpool)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64,   kernel_size=1), nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=1), nn.LeakyReLU(0.2),
            nn.Conv1d(128,1024,kernel_size=1), nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(1),
        )
    def forward(self, x):
        """x: (B, N, 3) -> (B, 1024)"""
        return self.net(x.transpose(1, 2)).squeeze(-1)


class SiameseRotNet(nn.Module):
    """
    Weight-sharing Siamese PointNet.
    Encoder: shared PointNetEncoder (→ 1024 each)
    Decoder MLP: 2048 → 512 → 512 → D  (per paper)
    """
    def __init__(self, rep_dim):
        super().__init__()
        self.encoder = PointNetEncoder()
        self.decoder = nn.Sequential(
            nn.Linear(2048, 512), nn.LeakyReLU(0.2),
            nn.Linear(512,  512), nn.LeakyReLU(0.2),
            nn.Linear(512,  rep_dim),
        )
        self.rep_dim = rep_dim

    def forward(self, pc_ref, pc_tgt):
        """
        pc_ref, pc_tgt: (B, N, 3)
        Returns: (B, rep_dim)
        """
        z_r = self.encoder(pc_ref)   # (B, 1024)
        z_t = self.encoder(pc_tgt)   # (B, 1024)
        z   = torch.cat([z_r, z_t], dim=1)   # (B, 2048)
        return self.decoder(z)        # (B, rep_dim)

# =============================================================================
# 5.  Geodesic error
# =============================================================================

def geodesic_deg(R1, R2):
    """(B,3,3), (B,3,3) -> (B,) in degrees"""
    M   = R1 @ R2.transpose(-1, -2)
    tr  = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
    cos = torch.clamp((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    return torch.acos(cos) * 180 / math.pi

# =============================================================================
# 6.  Training
# =============================================================================

def train_one_rep(name, f_fn, rep_dim, train_clouds,
                  total_iters=2_600_000, batch_size=10):
    """
    At each iteration:
      1. Pick one random reference cloud
      2. Sample batch_size rotation matrices
      3. Apply rotations to get batch_size target clouds
      4. Forward through Siamese net → predicted rotation
      5. L2 loss between predicted and GT rotation matrices
    """
    print(f"\n{'='*60}\nTraining [{name}]", flush=True)

    net = SiameseRotNet(rep_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    net.train()

    log_every  = max(1, total_iters // 200)
    mean_losses = []
    t0 = time.time()

    # Pre-convert all training clouds to GPU tensors once — avoids a
    # CPU->GPU transfer and a numpy->torch conversion on every iteration.
    train_tensors = [
        torch.FloatTensor(pc).to(device) for pc in train_clouds
    ]

    for it in range(1, total_iters + 1):
        # --- pick random reference cloud (already on device) ---
        pc_ref_1 = train_tensors[np.random.randint(len(train_tensors))]
        point_num = pc_ref_1.shape[0]

        # pc_ref: (batch, N, 3)
        pc_ref = pc_ref_1.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

        # gt rotations sampled directly on device — no transfer needed
        gt_R = random_so3_batch(batch_size, device)

        # pc_tgt = R @ pc_ref  (apply per-point)
        # gt_R: (B,3,3), pc_ref: (B,N,3)
        pc_tgt = torch.bmm(
            gt_R.unsqueeze(1).expand(-1, point_num, -1, -1).reshape(-1, 3, 3),
            pc_ref.reshape(-1, 3, 1)
        ).reshape(batch_size, point_num, 3)

        # forward
        pred_rep = net(pc_ref, pc_tgt)          # (B, rep_dim)
        pred_R   = f_fn(pred_rep)               # (B, 3, 3)

        # L2 loss on rotation matrices (paper: "minimize L2 loss between
        # output and ground-truth rotation matrices")
        loss = ((pred_R - gt_R) ** 2).mean()

        opt.zero_grad(set_to_none=True)  # frees grad buffers instead of zeroing
        loss.backward()
        opt.step()

        if it % log_every == 0:
            elapsed = time.time() - t0
            eta     = elapsed / it * (total_iters - it)
            mean_losses.append((it, loss.item()))
            print(f"  [{name}] {it:>9d}/{total_iters} | "
                  f"loss {loss.item():.6f} | "
                  f"{elapsed/60:.1f}m elapsed | ETA {eta/60:.1f}m", flush=True)

    return net, mean_losses

# =============================================================================
# 7.  Evaluation
# =============================================================================

def evaluate(net, f_fn, rep_dim, test_clouds, n_aug=100):
    """
    For each test cloud, apply n_aug random rotations and measure
    geodesic error between GT and predicted rotation.
    Returns (n_test * n_aug,) numpy array of errors in degrees.
    """
    net.eval()
    all_errors = []

    # Pre-convert test clouds to device tensors once
    test_tensors = [torch.FloatTensor(pc).to(device) for pc in test_clouds]

    with torch.no_grad():
        for pc_ref_1 in test_tensors:
            point_num = pc_ref_1.shape[0]
            pc_ref = pc_ref_1.unsqueeze(0).expand(n_aug, -1, -1).contiguous()

            gt_R = random_so3_batch(n_aug, device)
            pc_tgt = torch.bmm(
                gt_R.unsqueeze(1).expand(-1, point_num, -1, -1).reshape(-1, 3, 3),
                pc_ref.reshape(-1, 3, 1)
            ).reshape(n_aug, point_num, 3)

            pred_rep = net(pc_ref, pc_tgt)
            pred_R   = f_fn(pred_rep)

            errs = geodesic_deg(pred_R, gt_R).cpu().numpy()
            all_errors.append(errs)

    return np.concatenate(all_errors)

# =============================================================================
# 8.  Main
# =============================================================================

if __name__ == "__main__":
    TRAIN_DIR   = "./small_points"
    TEST_DIR    = "./small_points_test"
    # TODO: Change back to reasonable number
    TOTAL_ITERS = 50_000
    BATCH_SIZE  = 10    # 10 rotations per reference cloud per iteration
    N_AUG       = 100   # augmentations per test cloud

    print("Loading training point clouds...", flush=True)
    train_clouds = load_pointcloud_folder(TRAIN_DIR)

    print("Loading test point clouds...", flush=True)
    test_clouds  = load_pointcloud_folder(TEST_DIR)

    results = {}
    for name, f_fn, rep_dim in REPS:
        net, mean_losses = train_one_rep(
            name, f_fn, rep_dim, train_clouds, TOTAL_ITERS, BATCH_SIZE)
        final_errors = evaluate(net, f_fn, rep_dim, test_clouds, N_AUG)
        results[name] = {"mean_losses": mean_losses, "final_errors": final_errors}
        e = final_errors
        print(f"  [{name}] mean={e.mean():.2f}°  max={e.max():.2f}°  "
              f"std={e.std():.2f}°", flush=True)

    pickle.dump(results, open("pointcloud_results.pkl", "wb"))
    print("\nResults saved → pointcloud_results.pkl", flush=True)

    # =========================================================================
    # 9.  Plot — same 3-panel layout as sanity test and IK experiment
    # =========================================================================
    COLORS = {"6D": "red", "Quaternion": "green", "Axis-angle": "cyan", "Euler": "blue", "SVD": "magenta"}
    STYLES = {"6D": "-", "Quaternion": "-", "Axis-angle": "-", "Euler": "-", "SVD": "--"}
    ORDER = ["6D", "Quaternion", "Axis-angle", "Euler", "SVD"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.text(0.01, 0.98, "Pose Estimation — 3D Point Clouds",
             fontsize=13, va='top', fontweight='normal')

    # (a) mean loss during training
    ax = axes[0]
    for name in ORDER:
        xs = [x[0] for x in results[name]["mean_losses"]]
        ys = [x[1] for x in results[name]["mean_losses"]]
        ax.plot(xs, ys, color=COLORS[name], linestyle=STYLES[name], linewidth=1.5, label=name)
    ax.set_xlim(0, TOTAL_ITERS)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x)//1000}k"))
    ax.set_xlabel("a. Mean loss during iterations.", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)

    # (b) percentile of geodesic error at end of training
    ax = axes[1]
    pcts = np.linspace(0, 100, 1000)
    for name in ORDER:
        vals = np.percentile(results[name]["final_errors"], pcts)
        ax.semilogy(pcts, vals, color=COLORS[name], linewidth=1.5, label=name)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda y, _: f"{y:g}°"))
    ax.set_xlabel("b. Percentile of errors at final iteration.", fontsize=9)
    ax.legend(fontsize=8, loc='upper left')
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, which='both')

    # (c) summary table
    ax = axes[2]; ax.axis('off')
    col_labels = ["", "Mean(°)", "Max(°)", "Std(°)"]
    rows = []
    for name in ORDER:
        e = results[name]["final_errors"]
        rows.append([name, f"{e.mean():.2f}", f"{e.max():.2f}", f"{e.std():.2f}"])
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:           cell.set_text_props(fontweight='bold')
        if r == 1 and c > 0: cell.set_text_props(fontweight='bold')  # 6D row
    ax.set_xlabel("c. Errors at final iteration.", fontsize=9)

    plt.tight_layout()
    plt.savefig("pointcloud_results.png", dpi=150, bbox_inches='tight')
    print("Plot saved → pointcloud_results.png", flush=True)