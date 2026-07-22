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




def _rot_around_axis(x, theta):
    """
    Rotation matrix for rotating by angle theta around unit axis x.
    x: (B,3), theta: (B,) -> (B,3,3)
    Uses Rodrigues formula: R = cos(t)*I + sin(t)*K + (1-cos(t))*(x x^T)
    """
    B = x.shape[0]; dev = x.device
    c = torch.cos(theta); s = torch.sin(theta); t = 1 - c
    K = torch.zeros(B, 3, 3, device=dev)
    K[:, 0, 1] = -x[:, 2]; K[:, 0, 2] =  x[:, 1]
    K[:, 1, 0] =  x[:, 2]; K[:, 1, 2] = -x[:, 0]
    K[:, 2, 0] = -x[:, 1]; K[:, 2, 1] =  x[:, 0]
    I   = torch.eye(3, device=dev).unsqueeze(0)
    xxT = x.unsqueeze(2) * x.unsqueeze(1)
    return c.view(B,1,1)*I + s.view(B,1,1)*K + t.view(B,1,1)*xxT


def g_5d(M):
    """
    Reverse mapping SO(3) -> R^5.
    R = [x  y  z] (columns).

      v1,v2,v3 = components of x (first column)

      a = 1/(1+x[3]),  b = -a*x[1]*x[2]   (1-indexed notation from paper)

      y' = [1-a*x[1]^2,  b,  -x[1]]^T     (reference second column at theta=0)

      theta = signed angle from y' to y, measured around axis x:
        cos(theta) = y'_n . y
        sin(theta) = (y'_n x y) . x   (cross product projected onto x for sign)

      v4 = sin(theta),  v5 = cos(theta)
    """
    x = M[:, :, 0]   # (B,3) first column
    y = M[:, :, 1]   # (B,3) second column

    a = 1.0 / (1.0 + x[:, 2].clamp(min=-1+1e-6))
    b = -a * x[:, 0] * x[:, 1]

    y_prime = torch.stack([1 - a*x[:, 0]**2, b, -x[:, 0]], dim=1)

    norm_yp   = y_prime.norm(dim=1, keepdim=True).clamp(min=1e-8)
    y_prime_n = y_prime / norm_yp

    cos_t = (y_prime_n * y).sum(dim=1).clamp(-1+1e-6, 1-1e-6)
    cross = torch.cross(y_prime_n, y, dim=1)
    sin_t = (cross * x).sum(dim=1)

    return torch.stack([x[:, 0], x[:, 1], x[:, 2], sin_t, cos_t], dim=1)


def f_5d(v):
    """
    Forward mapping R^5 -> SO(3).

      x     = normalize([v1, v2, v3])
      theta = atan2(v4, v5)   i.e. atan2(sin, cos)

      a = 1/(1+x[3]),  b = -a*x[1]*x[2]

      y' = [1-a*x[1]^2,  b,  -x[1]]^T
      z' = [b,  1-a*x[2]^2,  -x[2]]^T

      R_x(theta): rotation by theta around axis x (Rodrigues formula)

      y = R_x(theta) @ y'
      z = R_x(theta) @ z'

      R = [x  y  z]  (as column matrix)
    """
    B   = v.shape[0]
    dev = v.device

    x     = nn.functional.normalize(v[:, 0:3], dim=1)
    theta = torch.atan2(v[:, 3], v[:, 4])   # atan2(sin, cos) = theta

    a = 1.0 / (1.0 + x[:, 2].clamp(min=-1+1e-6))
    b = -a * x[:, 0] * x[:, 1]

    y_prime = torch.stack([1 - a*x[:, 0]**2, b, -x[:, 0]], dim=1)
    z_prime = torch.stack([b, 1 - a*x[:, 1]**2, -x[:, 1]], dim=1)

    # Critical: rotate around x (the first column), not the global [1,0,0] axis
    Rx = _rot_around_axis(x, theta)

    y = torch.bmm(Rx, y_prime.unsqueeze(2)).squeeze(2)
    z = torch.bmm(Rx, z_prime.unsqueeze(2)).squeeze(2)

    Rout = torch.zeros(B, 3, 3, device=dev)
    Rout[:, :, 0] = x
    Rout[:, :, 1] = y
    Rout[:, :, 2] = z
    return Rout


REPS = [
    ("6D",         g_6d,        f_6d,        6),
    ("5D Frisvad",         g_5d,        f_5d,        5),
    ("Quaternion", g_quat,      f_quat,      4),
    ("Axis-angle", g_axisangle, f_axisangle, 3),
    ("Euler",      g_euler,     f_euler,     3),
    ("SVD",        g_svd,       f_svd,       9),
]