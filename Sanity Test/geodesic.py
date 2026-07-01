import torch, torch.nn as nn, numpy as np, math, pickle, time, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}", flush=True)

# ---- Rotation sampling ----
EPS = 1e-6

def sample_unit_sphere_batch(n):
    """
    Muller's method: sample n unit vectors uniformly on S^2.
    Draw from isotropic Gaussian, normalize. Resample any degenerate vectors.
    Returns: (n, 3) tensor of unit vectors.
    """
    v = torch.randn(n, 3)
    norms = v.norm(dim=1, keepdim=True)
    # Resample any nearly-zero vectors (extremely rare)
    bad = (norms < 1e-10).squeeze(1)
    while bad.any():
        v[bad] = torch.randn(bad.sum(), 3)
        norms[bad] = v[bad].norm(dim=1, keepdim=True)
        bad = (norms < 1e-10).squeeze(1)
    return v / norms

def expmap_so3_batch(u):
    """
    Rodrigues exponential map from R^3 -> SO(3).
    u is an axis-angle vector: direction = rotation axis, magnitude beta = rotation angle.

    exp(K) = I + p*K + q*K^2
    where K is the skew-symmetric matrix of u, and:
      p = sin(beta)/beta       (with Taylor series near beta=0)
      q = (1-cos(beta))/beta^2 (with Taylor series near beta=0)
    """
    beta = u.norm(dim=1)  # (n,)
    small = beta < EPS

    # --- p: sin(beta)/beta ---
    p = torch.where(small,
                    1 - beta ** 2 / 6 + beta ** 4 / 120,  # Taylor
                    torch.sin(beta) / beta)  # exact

    # --- q: (1-cos(beta))/beta^2 ---
    q = torch.where(small,
                    0.5 - beta ** 2 / 24 + beta ** 4 / 720,  # Taylor
                    (1 - torch.cos(beta)) / beta ** 2)  # exact

    a, b, c = u[:, 0], u[:, 1], u[:, 2]

    # Skew-symmetric K for each sample
    K = torch.zeros(u.shape[0], 3, 3, device=u.device)
    K[:, 0, 1] = -c;
    K[:, 0, 2] = b
    K[:, 1, 0] = c;
    K[:, 1, 2] = -a
    K[:, 2, 0] = -b;
    K[:, 2, 1] = a

    K2 = K @ K  # (n,3,3)

    I = torch.eye(3, device=u.device).unsqueeze(0).expand(u.shape[0], -1, -1)

    p = p.view(-1, 1, 1)
    q = q.view(-1, 1, 1)

    R = I + p * K + q * K2  # (n,3,3)
    return R

def random_so3_batch(n, device=None):
    """
    Drop-in replacement. Samples n rotation matrices uniformly from SO(3).

    Steps:
      1. Sample unit axis u via Muller's method (isotropic Gaussian + normalize)
      2. Sample angle beta uniformly in [0, pi]
      3. Form axis-angle vector: u * beta
      4. Map to SO(3) via Rodrigues exponential map
    """
    axes = sample_unit_sphere_batch(n)  # (n, 3) unit vectors
    betas = torch.rand(n, 1) * math.pi  # (n, 1) angles in [0, pi]
    u = axes * betas  # (n, 3) axis-angle vectors

    if device is not None:
        u = u.to(device)

    return expmap_so3_batch(u)  # (n, 3, 3)


# ---- Representations ----
def g_6d(M):
    return M[:,:,:2].reshape(-1,6)
def f_6d(r):
    a1,a2=r[:,0:3],r[:,3:6]
    b1=nn.functional.normalize(a1,dim=1)
    b2=nn.functional.normalize(a2-(b1*a2).sum(1,keepdim=True)*b1,dim=1)
    b3=torch.cross(b1,b2,dim=1)
    return torch.stack([b1,b2,b3],dim=2)

def g_quat(M):
    B=M.shape[0]; dev=M.device
    t=M[:,0,0]+M[:,1,1]+M[:,2,2]+1
    q_nz=torch.stack([M[:,2,1]-M[:,1,2],M[:,0,2]-M[:,2,0],M[:,1,0]-M[:,0,1],t],dim=1)
    sgn32=torch.sign(M[:,2,1]); sgn32=torch.where(sgn32==0,torch.ones_like(sgn32),sgn32)
    def ci(i):
        v=M[:,i,0]+M[:,0,i]
        return torch.where(v>0,torch.ones(B,device=dev),
               torch.where(v<0,-torch.ones(B,device=dev), sgn32**(i+1)))
    q_z=torch.stack([
        torch.sqrt(torch.clamp(M[:,0,0]+1,min=1e-8)),
        ci(1)*torch.sqrt(torch.clamp(M[:,1,1]+1,min=1e-8)),
        ci(2)*torch.sqrt(torch.clamp(M[:,2,2]+1,min=1e-8)),
        torch.zeros(B,device=dev)],dim=1)
    return torch.where((t.abs()>1e-7).unsqueeze(1),q_nz,q_z)
def f_quat(q):
    q=nn.functional.normalize(q,dim=1)
    x,y,z,w=q[:,0],q[:,1],q[:,2],q[:,3]
    R=torch.zeros(q.shape[0],3,3,device=q.device)
    R[:,0,0]=1-2*y*y-2*z*z; R[:,0,1]=2*x*y-2*z*w; R[:,0,2]=2*x*z+2*y*w
    R[:,1,0]=2*x*y+2*z*w;   R[:,1,1]=1-2*x*x-2*z*z; R[:,1,2]=2*y*z-2*x*w
    R[:,2,0]=2*x*z-2*y*w;   R[:,2,1]=2*y*z+2*x*w;   R[:,2,2]=1-2*x*x-2*y*y
    return R

def g_axisangle(M):
    trace=M[:,0,0]+M[:,1,1]+M[:,2,2]
    theta=torch.acos(torch.clamp((trace-1)/2,-1+1e-6,1-1e-6))
    ax=torch.stack([M[:,2,1]-M[:,1,2],M[:,0,2]-M[:,2,0],M[:,1,0]-M[:,0,1]],dim=1)
    ax=ax/(2*torch.sin(theta).unsqueeze(1)+1e-8)
    return ax*theta.unsqueeze(1)
def f_axisangle(v):
    theta=v.norm(dim=1,keepdim=True).clamp(min=1e-8)
    ax=v/theta; theta=theta.squeeze(1)
    c,s=torch.cos(theta),torch.sin(theta); t=1-c
    x,y,z=ax[:,0],ax[:,1],ax[:,2]
    R=torch.zeros(v.shape[0],3,3,device=v.device)
    R[:,0,0]=t*x*x+c; R[:,0,1]=t*x*y-s*z; R[:,0,2]=t*x*z+s*y
    R[:,1,0]=t*x*y+s*z; R[:,1,1]=t*y*y+c; R[:,1,2]=t*y*z-s*x
    R[:,2,0]=t*x*z-s*y; R[:,2,1]=t*y*z+s*x; R[:,2,2]=t*z*z+c
    return R

def g_euler(M):
    sy=torch.sqrt(M[:,0,0]**2+M[:,1,0]**2); singular=sy<1e-6
    x=torch.atan2(M[:,2,1],M[:,2,2]); y=torch.atan2(-M[:,2,0],sy); z=torch.atan2(M[:,1,0],M[:,0,0])
    xs=torch.atan2(-M[:,1,2],M[:,1,1]); zs=torch.zeros_like(z)
    return torch.stack([torch.where(singular,xs,x),y,torch.where(singular,zs,z)],dim=1)
def f_euler(e):
    cx,sx=torch.cos(e[:,0]),torch.sin(e[:,0])
    cy,sy=torch.cos(e[:,1]),torch.sin(e[:,1])
    cz,sz=torch.cos(e[:,2]),torch.sin(e[:,2])
    R=torch.zeros(e.shape[0],3,3,device=e.device)
    R[:,0,0]=cy*cz; R[:,0,1]=cz*sx*sy-cx*sz; R[:,0,2]=cx*cz*sy+sx*sz
    R[:,1,0]=cy*sz; R[:,1,1]=cx*cz+sx*sy*sz; R[:,1,2]=cx*sy*sz-cz*sx
    R[:,2,0]=-sy;   R[:,2,1]=cy*sx;           R[:,2,2]=cx*cy
    return R

REPS = [
    ("6D",          g_6d,        f_6d,        6),
 #   ("5D",          g_quat,      f_quat,      4),
    ("Quaternion",  g_quat,      f_quat,      4),
    ("Axis-angle",  g_axisangle, f_axisangle, 3),
    ("Euler",       g_euler,     f_euler,     3),
]

# ---- Network ----
class EncoderMLP(nn.Module):
    def __init__(self, odim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9,128), nn.LeakyReLU(0.2),
            nn.Linear(128,128), nn.LeakyReLU(0.2),
            nn.Linear(128,128), nn.LeakyReLU(0.2),
            nn.Linear(128,odim),
        )
    def forward(self, x): return self.net(x)

def geodesic_deg(M, Mp):
    Mpp = M @ Mp.transpose(-1,-2)  # M @ M'^{-1} = M @ M'^T for orthogonal
    tr = Mpp[:,0,0]+Mpp[:,1,1]+Mpp[:,2,2]
    return torch.acos(torch.clamp((tr-1)/2,-1+1e-6,1-1e-6)) * 180/math.pi

# ---- Training ----
TOTAL = 500_000; BS = 64; LOG_EVERY = 5000

results = {}
for name, g_fn, f_fn, dim in REPS:
    print(f"\n{'='*50}", flush=True)
    print(f"Training {name} ({dim}D output)...", flush=True)
    t0 = time.time()

    net = EncoderMLP(dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-5)

    mean_errors = []
    for it in range(1, TOTAL+1):
        if it == 10_001:
            for pg in opt.param_groups: pg['lr'] = 1e-6

        M = random_so3_batch(BS, device)
        # net transfers 9 so3 values into num of parameters for representation R
        r = net(M.reshape(BS, 9))
        Mp = f_fn(r)
        loss = geodesic_deg(M, Mp).mean()
        opt.zero_grad(); loss.backward(); opt.step()

        if it % LOG_EVERY == 0:
            with torch.no_grad():
                err = geodesic_deg(M, Mp).mean().item()
            mean_errors.append((it, err))
            elapsed = time.time() - t0
            eta = elapsed/it * (TOTAL-it)
            print(f"  [{name}] iter {it:>7d} | err {err:.3f}° | elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m", flush=True)

    # Evaluate on 100k test samples
    net.eval()
    all_errs = []
    with torch.no_grad():
        for _ in range(100):
            M = random_so3_batch(1000, device)
            Mp = f_fn(net(M.reshape(1000, 9)))
            all_errs.append(geodesic_deg(M, Mp).cpu().numpy())
    final_errors = np.concatenate(all_errs)

    results[name] = {"mean_errors": mean_errors, "final_errors": final_errors}
    print(f"  [{name}] FINAL: mean={final_errors.mean():.2f}° max={final_errors.max():.2f}° std={final_errors.std():.2f}°", flush=True)

# Save results
pickle.dump(results, open("./geodesic.pkl", "wb"))
print("\nResults saved.", flush=True)

# ---- Plotting ----
colors = {"6D":"red","Quaternion":"green","Axis-angle":"cyan","Euler":"blue"}
styles = {"6D":"-","Quaternion":"-","Axis-angle":"-","Euler":"-"}
order = ["6D","Quaternion","Axis-angle","Euler"]
#TODO: change back when you implement 5D
#colors = {"6D":"red","5D":"#c8c800","Quaternion":"green","Axis-angle":"cyan","Euler":"blue"}
#styles = {"6D":"-","5D":"--","Quaternion":"-","Axis-angle":"-","Euler":"-"}
#order = ["6D","5D","Quaternion","Axis-angle","Euler"]


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.text(0.01, 0.97, "Sanity Test", fontsize=13, va='top', ha='left')

# (a) Mean errors during training
ax = axes[0]
for name in order:
    iters = [x[0] for x in results[name]["mean_errors"]]
    errs  = [x[1] for x in results[name]["mean_errors"]]
    ax.plot(iters, errs, color=colors[name], linestyle=styles[name], linewidth=1.5, label=name)
ax.set_xlim(0, TOTAL)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{int(x)//1000}k"))
ax.set_xlabel("a. Mean errors during iterations.", fontsize=9)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3)

# (b) Percentile of errors at 500k
ax = axes[1]
pcts = np.linspace(0, 100, 1000)
for name in order:
    vals = np.percentile(results[name]["final_errors"], pcts)
    ax.semilogy(pcts, vals, color=colors[name], linestyle=styles[name], linewidth=1.5, label=name)
ax.set_xlim(0, 100)
ax.set_ylim(0.1, 200)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{int(x)}%"))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y,_: f"{y:g}°"))
ax.set_xlabel("b. Percentile of errors at 500k iteration.", fontsize=9)
ax.legend(fontsize=8, loc='upper left')
ax.tick_params(labelsize=8)
ax.grid(True, alpha=0.3, which='both')

# (c) Table
ax = axes[2]
ax.axis('off')
col_labels = ["", "Mean(°)", "Max(°)", "Std(°)"]
table_data = []
for name in order:
    e = results[name]["final_errors"]
    table_data.append([name, f"{e.mean():.2f}", f"{e.max():.2f}", f"{e.std():.2f}"])
t = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1.1, 1.6)
for (row,col),cell in t.get_celld().items():
    if row==0: cell.set_text_props(fontweight='bold')
ax.set_xlabel("c. Errors at 500k iteration.", fontsize=9)

plt.tight_layout()
out = "./geodesic.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {out}")
