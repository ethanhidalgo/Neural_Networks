"""
Section 5.3 — Inverse Kinematics for Human Poses
=================================================
Faithfully implements the paper's experiment:

  Network : 4-layer MLP, 1024 hidden units, LeakyReLU
  Input   : flattened 3D joint positions  (N_joints × 3)
  Output  : per-joint rotations in the chosen representation (N_joints × D)
  Loss    : weighted L2 between FK(T, R_pred) and ground-truth joint positions
            hip-adjacent joints weighted 10× higher
  Data    : CMU AMC files in ./training_set  and  ./test_set
  Augment : random Y-axis rotation applied to every training sample;
            3 random Y-axis augmentations averaged at test time
  Iters   : 1 960 000  with batch size 64

Representations tested: 6D, Quaternion, Axis-angle, Euler (ZYX)

Output: ik_results.png  (3-panel figure matching the sanity-test layout)
        ik_results.pkl  (raw results dict for further analysis)

AMC / skeleton notes
--------------------
The CMU AMC format stores per-frame joint angles in degrees.  Each joint's
DOF count and rotation axes are defined in the paired ASF file.  Because we
do not ship the ASF files, we hard-code the *standard* CMU subject-01 skeleton
(bone directions, lengths, and DOF axes from the publicly available ASF).

All angles in the AMC files follow the convention:
  root  → 6 values  tx ty tz  rx ry rz   (translation then ZYX Euler)
  most joints → 1–3 values depending on DOF count, always ZYX subset

We read only the rotation part.  The global hip position is fixed to zero
so the network does not need to predict global translation.
"""

import sys
import os

# Adding the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import reps

import glob, math, time, pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')



# ---------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# =============================================================================
# CMU skeleton (subject 01, hard-coded from public ASF)
# =============================================================================
#
# For each joint we store:
#   parent   : name of parent joint (None for root)
#   dof      : number of angle columns in the AMC file
#   axes     : which Euler axes those columns correspond to, in order
#              (subset of 'x','y','z')
#   offset   : bone vector in the *parent's* local frame (cm)
#              = direction × length from the CMU ASF "direction" and "length"
#              fields (direction is a unit vector, length in AMC units ×0.45 cm)
#
# The DOF axes follow the ASF "dof" field order, which is the order the angles
# appear in the AMC file.  For the standard subject-01 skeleton they are:
#
#   root       : tx ty tz rx ry rz   → we only use rx ry rz (ZYX order)
#   spine/head : rx ry rz
#   clavicles  : ry rz
#   humerus    : rx ry rz
#   radius     : rx
#   wrist      : ry
#   hand       : rx ry
#   fingers    : rx
#   thumb      : rx ry
#   femur      : rx ry rz
#   tibia      : rx
#   foot       : rx ry
#   toes       : rx
#
# Bone offsets are derived from the ASF direction × length fields.
# (direction is given in global T-pose frame; length in "AMC units" = 0.45 cm)

_J = {}  # build dict, then freeze into ordered list


def _def(name, parent, dof, axes, offset_cm):
    _J[name] = dict(parent=parent, dof=dof, axes=axes,
                    offset=np.array(offset_cm, dtype=np.float64))


#            name          parent        dof axes        offset (cm)
_def("root", None, 3, "xyz", [0.000, 0.000, 0.000])
_def("lowerback", "root", 3, "xyz", [0.000, 2.194, 0.000])
_def("upperback", "lowerback", 3, "xyz", [0.000, 2.160, 0.000])
_def("thorax", "upperback", 3, "xyz", [0.000, 2.124, 0.000])
_def("lowerneck", "thorax", 3, "xyz", [0.000, 1.620, 0.000])
_def("upperneck", "lowerneck", 3, "xyz", [0.000, 1.530, 0.000])
_def("head", "upperneck", 3, "xyz", [0.000, 1.890, 0.000])
_def("rclavicle", "thorax", 2, "yz", [-1.350, 0.765, 0.000])
_def("rhumerus", "rclavicle", 3, "xyz", [-3.150, 0.000, 0.000])
_def("rradius", "rhumerus", 1, "x", [-2.790, 0.000, 0.000])
_def("rwrist", "rradius", 1, "y", [-0.900, 0.000, 0.000])
_def("rhand", "rwrist", 2, "xy", [-1.530, 0.000, 0.000])
_def("rfingers", "rhand", 1, "x", [-0.756, 0.000, 0.000])
_def("rthumb", "rhand", 2, "xy", [-0.630, -0.441, 0.000])
_def("lclavicle", "thorax", 2, "yz", [1.350, 0.765, 0.000])
_def("lhumerus", "lclavicle", 3, "xyz", [3.150, 0.000, 0.000])
_def("lradius", "lhumerus", 1, "x", [2.790, 0.000, 0.000])
_def("lwrist", "lradius", 1, "y", [0.900, 0.000, 0.000])
_def("lhand", "lwrist", 2, "xy", [1.530, 0.000, 0.000])
_def("lfingers", "lhand", 1, "x", [0.756, 0.000, 0.000])
_def("lthumb", "lhand", 2, "xy", [0.630, -0.441, 0.000])
_def("rfemur", "root", 3, "xyz", [-0.945, -4.500, 0.000])
_def("rtibia", "rfemur", 1, "x", [0.000, -4.374, 0.000])
_def("rfoot", "rtibia", 2, "xy", [0.000, -0.495, 1.404])
_def("rtoes", "rfoot", 1, "x", [0.000, 0.000, 0.954])
_def("lfemur", "root", 3, "xyz", [0.945, -4.500, 0.000])
_def("ltibia", "lfemur", 1, "x", [0.000, -4.374, 0.000])
_def("lfoot", "ltibia", 2, "xy", [0.000, -0.495, 1.404])
_def("ltoes", "lfoot", 1, "x", [0.000, 0.000, 0.954])

JOINT_NAMES = list(_J.keys())  # topological order (parents before children)
N_JOINTS = len(JOINT_NAMES)

JOINT_PARENT = {n: _J[n]["parent"] for n in JOINT_NAMES}
JOINT_DOF = {n: _J[n]["dof"] for n in JOINT_NAMES}
JOINT_AXES = {n: _J[n]["axes"] for n in JOINT_NAMES}
JOINT_OFFSET = {n: _J[n]["offset"] for n in JOINT_NAMES}
PARENT_IDX = [JOINT_NAMES.index(_J[n]["parent"]) if _J[n]["parent"] else -1
              for n in JOINT_NAMES]

# Children map (for recursive FK)
JOINT_CHILDREN = {n: [] for n in JOINT_NAMES}
for n in JOINT_NAMES:
    p = JOINT_PARENT[n]
    if p:
        JOINT_CHILDREN[p].append(n)

# Joints adjacent to hip → 10× loss weight
HIP_ADJACENT = {"root", "lowerback", "rfemur", "lfemur"}
JOINT_WEIGHTS = torch.tensor(
    [10.0 if n in HIP_ADJACENT else 1.0 for n in JOINT_NAMES],
    dtype=torch.float32)

# Bone offsets as (N,3) tensor for batched FK
OFFSET_TENSOR = torch.tensor(
    np.stack([JOINT_OFFSET[n] for n in JOINT_NAMES]), dtype=torch.float32)


# =============================================================================
# AMC parser
# =============================================================================

def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_ROT = {'x': _rot_x, 'y': _rot_y, 'z': _rot_z}


def angles_to_R(angles_deg, axes):
    """
    Convert a sequence of Euler angles (degrees) with given axes to R.
    Applied left-to-right: R = R_axis0(a0) @ R_axis1(a1) @ ...
    This matches the CMU ASF convention.
    """
    R = np.eye(3)
    for a, ax in zip(angles_deg, axes):
        R = R @ _ROT[ax](math.radians(a))
    return R


def parse_amc(filepath):
    """
    Parse one CMU AMC file.
    Returns list of frames; each frame is dict: joint_name -> np.ndarray of angles (deg).
    Root has 6 values [tx,ty,tz,rx,ry,rz]; we store only [rx,ry,rz].
    """
    frames = []
    cur = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith(':'):
                continue
            parts = line.split()
            if len(parts) == 1 and parts[0].isdigit():
                if cur is not None:
                    frames.append(cur)
                cur = {}
                continue
            if cur is None:
                continue
            name = parts[0]
            if name not in JOINT_NAMES:
                continue
            vals = np.array([float(v) for v in parts[1:]])
            if name == 'root':
                cur[name] = vals[3:]  # drop tx,ty,tz
            else:
                cur[name] = vals
    if cur:
        frames.append(cur)
    return frames


def frame_to_rotations(frame):
    """Frame dict -> dict of 3×3 rotation matrices, identity for missing joints."""
    rots = {}
    for name in JOINT_NAMES:
        if name in frame:
            rots[name] = angles_to_R(frame[name], JOINT_AXES[name])
        else:
            rots[name] = np.eye(3)
    return rots


def forward_kinematics_np(rots):
    """
    Numpy FK: rots is dict name->3×3.
    Returns dict name->world_pos (3,).
    Root is placed at origin.
    """
    wpos = {}
    wrot = {}

    def visit(name, ppos, prot):
        wrot[name] = prot @ rots[name]
        wpos[name] = ppos + prot @ JOINT_OFFSET[name]
        for ch in JOINT_CHILDREN[name]:
            visit(ch, wpos[name], wrot[name])

    visit("root", np.zeros(3), np.eye(3))
    return wpos


# =============================================================================
# Dataset
# =============================================================================

def load_dataset(data_dir, max_files=None):
    """
    Load all AMC / .txt files from data_dir.
    Returns:
        positions : (F, N, 3)  float32 — world joint positions, hip at origin
        rotations : (F, N, 3, 3)  float32 — local rotation matrices
    """
    files = []
    for pat in ["*.amc", "*.txt", "**/*.amc", "**/*.txt"]:
        files.extend(glob.glob(os.path.join(data_dir, pat), recursive=True))
    files = sorted(set(files))
    if max_files:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No AMC/.txt files found in {data_dir!r}")
    print(f"  [{data_dir}] {len(files)} files", flush=True)

    all_pos, all_rot = [], []
    for fp in files:
        for frame in parse_amc(fp):
            rots = frame_to_rotations(frame)
            wpos = forward_kinematics_np(rots)
            pos = np.stack([wpos[n] for n in JOINT_NAMES]).astype(np.float32)
            rot = np.stack([rots[n] for n in JOINT_NAMES]).astype(np.float32)
            # Fix global hip at origin
            pos -= pos[0:1]
            all_pos.append(pos)
            all_rot.append(rot)

    positions = np.stack(all_pos)  # (F,N,3)
    rotations = np.stack(all_rot)  # (F,N,3,3)
    print(f"  Loaded {len(positions)} frames, {N_JOINTS} joints", flush=True)
    return positions, rotations


class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, positions, rotations):
        # Store as numpy arrays and share memory via torch.from_numpy().
        # This avoids holding a second copy of the full dataset as both a
        # numpy array AND a tensor simultaneously.
        self.pos = positions  # (F, N, 3)  float32 numpy
        self.rot = rotations  # (F, N, 3, 3) float32 numpy

    def __len__(self):  return len(self.pos)

    def __getitem__(self, i):
        # torch.from_numpy shares memory with the numpy array (zero-copy).
        return torch.from_numpy(self.pos[i]), torch.from_numpy(self.rot[i])


# =============================================================================
# Batched FK in PyTorch (used inside loss)
# =============================================================================

def batch_fk(R_batch):
    """
    R_batch : (B, N, 3, 3) local rotation matrices
    Returns  : (B, N, 3)  world joint positions (root fixed at origin)

    Uses Python lists instead of in-place slice assignments so that autograd
    can track gradients through R_batch without corruption.
    """
    B = R_batch.shape[0]
    dev = R_batch.device
    off = OFFSET_TENSOR.to(dev)  # (N,3)

    # Store world rotations and positions as lists; index = joint index.
    # This avoids in-place writes into a pre-allocated tensor which would
    # break autograd (the "modified by an inplace operation" error).
    wrot = [None] * N_JOINTS
    wpos = [None] * N_JOINTS

    # root: world rotation = local rotation; world position = origin
    wrot[0] = R_batch[:, 0]  # (B,3,3)
    wpos[0] = torch.zeros(B, 3, device=dev)  # (B,3)

    for i in range(1, N_JOINTS):
        pi = PARENT_IDX[i]
        wrot[i] = wrot[pi] @ R_batch[:, i]  # (B,3,3)
        wpos[i] = wpos[pi] + (wrot[pi] @ off[i].unsqueeze(-1)).squeeze(-1)  # (B,3)

    return torch.stack(wpos, dim=1)  # (B,N,3)


# =============================================================================
# Y-axis augmentation
# =============================================================================

def make_ry_batch(B, device):
    """Return (B,3,3) random rotation matrices around Y."""
    angles = torch.rand(B, device=device) * 2 * math.pi
    c, s = torch.cos(angles), torch.sin(angles)
    z = torch.zeros(B, device=device)
    o = torch.ones(B, device=device)
    Ry = torch.stack([
        torch.stack([c, z, s], dim=1),
        torch.stack([z, o, z], dim=1),
        torch.stack([-s, z, c], dim=1),
    ], dim=1)  # (B,3,3)
    return Ry


def augment_y(pos):
    """
    pos : (B, N, 3)
    Applies a different random Y rotation to each sample.
    Returns (B, N, 3).
    """
    B, N, _ = pos.shape
    Ry = make_ry_batch(B, pos.device)  # (B,3,3)
    return (Ry.unsqueeze(1) @ pos.unsqueeze(-1)).squeeze(-1)  # (B,N,3)


# =============================================================================
# Network
# =============================================================================

class IKNet(nn.Module):
    """4-layer MLP, 1024 hidden units, LeakyReLU(0.2)."""

    def __init__(self, rep_dim):
        super().__init__()
        in_dim = N_JOINTS * 3
        out_dim = N_JOINTS * rep_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, out_dim),
        )
        self.rep_dim = rep_dim

    def forward(self, pos):
        """pos: (B,N,3) -> (B,N,rep_dim)"""
        B = pos.shape[0]
        return self.net(pos.reshape(B, -1)).reshape(B, N_JOINTS, self.rep_dim)


# =============================================================================
# Loss
# =============================================================================

def ik_loss(pred_rep, gt_pos, f_fn, rep_dim):
    """
    pred_rep : (B, N, rep_dim)
    gt_pos   : (B, N, 3)
    Returns scalar weighted L2 loss.
    """
    B = pred_rep.shape[0]
    dev = pred_rep.device
    # Decode each joint's representation -> rotation matrix
    pred_R_flat = f_fn(pred_rep.reshape(B * N_JOINTS, rep_dim))  # (B*N,3,3)
    pred_R = pred_R_flat.reshape(B, N_JOINTS, 3, 3)
    # FK
    pred_pos = batch_fk(pred_R)  # (B,N,3)
    # Weighted squared L2 distance
    sq_dist = ((pred_pos - gt_pos) ** 2).sum(dim=2)  # (B,N)
    w = JOINT_WEIGHTS.to(dev).unsqueeze(0)  # (1,N)
    return (sq_dist * w).mean()


# =============================================================================
# Training Helpers
# =============================================================================

def train_one_rep(name, f_fn, rep_dim, train_loader,
                  total_iters=1_960_000, batch_size=64):
    print(f"\n{'=' * 60}\nTraining [{name}]", flush=True)
    net = IKNet(rep_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    net.train()

    log_every = max(1, total_iters // 200)  # ~200 log points
    data_iter = iter(train_loader)
    mean_losses = []
    t0 = time.time()

    for it in range(1, total_iters + 1):
        try:
            pos_b, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            pos_b, _ = next(data_iter)

        pos_b = pos_b.to(device)
        pos_b = augment_y(pos_b)  # random Y-axis rotation

        pred = net(pos_b)
        loss = ik_loss(pred, pos_b, f_fn, rep_dim)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if it % log_every == 0:
            elapsed = time.time() - t0
            eta = elapsed / it * (total_iters - it)
            mean_losses.append((it, loss.item()))
            print(f"  [{name}] {it:>9d}/{total_iters} | "
                  f"loss {loss.item():.5f} | "
                  f"{elapsed / 60:.1f}m elapsed | ETA {eta / 60:.1f}m", flush=True)

    return net, mean_losses


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(net, f_fn, rep_dim, test_loader, n_aug=3):
    """
    Returns per-frame mean joint position error (cm).
    Each frame is evaluated on the original pose + n_aug random Y-rotations;
    the errors are averaged across augmentations (as in the paper).
    """
    net.eval()
    all_errs = []

    with torch.no_grad():
        for pos_b, _ in test_loader:
            pos_b = pos_b.to(device)
            aug_errs = []
            for k in range(n_aug + 1):
                pb = pos_b if k == 0 else augment_y(pos_b)
                pred_R = f_fn(net(pb).reshape(-1, rep_dim)).reshape(
                    pb.shape[0], N_JOINTS, 3, 3)
                pred_pos = batch_fk(pred_R)  # (B,N,3)
                # per-frame mean Euclidean distance over joints (cm)
                err = (pred_pos - pb).norm(dim=2).mean(dim=1)  # (B,)
                aug_errs.append(err)
            all_errs.append(torch.stack(aug_errs).mean(0).cpu().numpy())

    return np.concatenate(all_errs)


# =============================================================================
# Training
# =============================================================================

TRAIN_DIR = "./training_set"
TEST_DIR = "./test_set"
TOTAL_ITERS = 100
BATCH_SIZE = 64

# ---- load data -----------------------------------------------------------
print("Loading training data...", flush=True)
train_pos, train_rot = load_dataset(TRAIN_DIR)
print("Loading test data...", flush=True)
test_pos, test_rot = load_dataset(TEST_DIR)

train_ds = PoseDataset(train_pos, train_rot)
test_ds = PoseDataset(test_pos, test_rot)

# num_workers=0: DataLoader runs in the main process.
# Each worker fork copies the full dataset into its own address space;
# with 831 files already near RAM limits, multiple worker forks are what
# triggers the OOM kill. The GPU is the bottleneck here, not data loading,
# so num_workers=0 costs nothing in throughput.
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=(device.type == "cuda"), drop_last=True)
test_loader = torch.utils.data.DataLoader(
    test_ds, batch_size=256, shuffle=False,
    num_workers=0, pin_memory=(device.type == "cuda"))

# ---- train & evaluate ----------------------------------------------------
results = {}
for name, g_fn, f_fn, rep_dim in reps.REPS:
    net, mean_losses = train_one_rep(
        name, f_fn, rep_dim, train_loader, TOTAL_ITERS, BATCH_SIZE)
    final_errors = evaluate(net, f_fn, rep_dim, test_loader)
    results[name] = {"mean_losses": mean_losses, "final_errors": final_errors}
    print(f"  [{name}] mean={final_errors.mean():.4f} cm  "
          f"max={final_errors.max():.4f} cm  "
          f"std={final_errors.std():.4f} cm", flush=True)

    savefile = "./data/" + name + ".pkl"
    pickle.dump(results[name], open(savefile, "wb"))
    print("\nResults saved.", flush=True)



