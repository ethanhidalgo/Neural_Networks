import torch, torch.nn as nn, numpy as np, math, pickle, time



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
