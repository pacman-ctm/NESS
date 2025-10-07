import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms

import time
import argparse

# ---------- Data ----------
def make_split_mnist(root="./data", train_bs=64, test_bs=512, seed=0):
    g = torch.Generator().manual_seed(seed)
    tfm = transforms.ToTensor()
    tr_ds = datasets.MNIST(root, train=True,  transform=tfm, download=True)
    te_ds = datasets.MNIST(root, train=False, transform=tfm, download=True)
    pairs = [(0,1),(2,3),(4,5),(6,7),(8,9)]
    def subset(ds, a, b):
        x, y = ds.data.float()/255.0, ds.targets
        m = (y==a)|(y==b)
        x = x[m].view(-1, 28*28)   # flatten
        y = (y[m]==b).long()
        return TensorDataset(x, y)
    tasks = []
    for i,(a,b) in enumerate(pairs,1):
        tr = subset(tr_ds, a, b)
        te = subset(te_ds, a, b)
        tasks.append({
            "name": f"Task {i}: {a} vs {b}",
            "train": DataLoader(tr, batch_size=train_bs, shuffle=True, generator=g,
                                pin_memory=torch.cuda.is_available(), num_workers=2),
            "test":  DataLoader(te, batch_size=test_bs, shuffle=False,
                                pin_memory=torch.cuda.is_available(), num_workers=2),
        })
    return tasks

# ---------- Model ----------
class SimpleNet(nn.Module):
    """
    y = W2 * act(W1 * x + b1) + b2
    """
    def __init__(self, d=784, h=256, act="relu"):
        super().__init__()
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, 1)
        if act.lower() == "relu":
            self.act = nn.ReLU()
        elif act.lower() == "tanh":
            self.act = nn.Tanh()
        elif act.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act}")

        # (Optional) sensible init for ReLU networks
        # nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        z = self.act(self.fc1(x))
        return self.fc2(z).squeeze(1)  # logits

# ---------- Train / Eval ----------
@torch.no_grad()
def eval_bin(model, loader, device):
    model.eval(); c=n=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = (torch.sigmoid(model(x))>=0.5).long()
        c += (pred==y).sum().item(); n += y.numel()
    return c/max(1,n)

@torch.no_grad()
def eval_bin_with_head(model, head, loader, device):
    model.eval(); head.eval(); c=n=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        h = model.act(model.fc1(x))
        logits = head(h).squeeze(1)
        pred = (torch.sigmoid(logits)>=0.5).long()
        c += (pred==y).sum().item(); n += y.numel()
    return c/max(1,n)

# ---------- Utils --------
def svd_complement_subspace_eps(X, eps1, center=True, relative=False):
    """
    Build U_span from X using an SVD cutoff:
      - If relative=False: pick first j with S[j] <= eps1
      - If relative=True:  pick first j with S[j] <= eps1 * S[0]
    Returns: (U_span, j, S) where U_span has shape (d, r).
    """
    if X.numel() == 0:
        return torch.empty(28*28, 0), 0, torch.empty(0)

    Xc = X - X.mean(0, keepdim=True) if center else X
    U, S, Vt = torch.linalg.svd(Xc, full_matrices=False)  # U:(N,k), S:(k,), Vt:(k,d)

    if relative and S.numel() > 0:
        thr = eps1 * S[0]
        mask = (S <= thr)
    else:
        mask = (S <= eps1)

    j = int(mask.nonzero()[0]) if mask.any() else S.numel()
    V = Vt.transpose(0, 1)             # (d, k)
    U_span = V[:, j:]                  # (d, r)
    return U_span, j, S

# ---------- Run Ours ----------
def train_task_0(epochs_per_task=3, lr=0.1, weight_decay=0.0, seed=0,
                 hidden_size=256, act="relu", log_every=None):
    torch.manual_seed(seed)
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tasks = make_split_mnist(seed=seed)
    model = SimpleNet(d=784, h=hidden_size, act=act).to(device)

    # per-task heads
    heads = nn.ModuleList([nn.Linear(hidden_size, 1).to(device)])

    opt = torch.optim.SGD(
        list(model.fc1.parameters()) + list(heads[0].parameters()),
        lr=lr, weight_decay=weight_decay
    )

    T = len(tasks)
    acc = torch.zeros(T, T)

    print(f"\n=== [OURS] Training {tasks[0]['name']} (task 0) ===")
    model.train(); heads[0].train()
    it_global = 0
    for ep in range(1, epochs_per_task+1):
        for it,(x,y) in enumerate(tasks[0]["train"], 1):
            it_global += 1
            x,y = x.to(device), y.float().to(device)
            opt.zero_grad()
            h = model.act(model.fc1(x))
            logits = heads[0](h).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
            if log_every and it_global % log_every == 0:
                print(f"[OURS] epoch {ep} iter {it} loss={loss.item():.4f}")

    acc[0,0] = eval_bin_with_head(model, heads[0], tasks[0]["test"], device)
    row = " ".join(f"{100*acc[0,u]:6.2f}%" if u == 0 else "  ---- " for u in range(T))
    print(f"[OURS] After task 0: {row}")

    return model, heads, acc, tasks

def train_later_tasks(model, heads, tasks, epochs_per_task=3, lr=0.1, weight_decay=0.0,
                      seed=0, log_every=None, eps1=1e1, center=True, relative=False):
    """
    For t >= 1:
      - Build U_span via SVD cutoff (eps1), using complement directions.
      - Train only {W_tilde, δb1, head[t]} on task t.
      - Commit: W <- W + U_span @ W_tilde, b1 <- b1 + δb1.
      - Evaluate with each task's own head.
    """
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    dtype  = next(model.parameters()).dtype

    T = len(tasks)
    acc = torch.zeros(T, T)

    for t in range(1, T):
        # ----- Concatenate previous inputs -----
        prev_inputs = [tasks[u]["train"].dataset.tensors[0] for u in range(t)]
        X_prev = torch.cat(prev_inputs, dim=0) if prev_inputs else torch.empty(0, 28*28)
        X_prev = X_prev.to(dtype=torch.float32, device="cpu")

        # ----- U_span from SVD complement using eps1 -----
        U_span, j, S = svd_complement_subspace_eps(X_prev, eps1=eps1, center=center, relative=relative)
        Smax = S[0].item() if S.numel() > 0 else float('nan')
        
        d = 28*28
        r = U_span.shape[1]
        U_span_d = U_span.to(device=device, dtype=dtype)

        # ----- Snapshot base W,b1 -----
        with torch.no_grad():
            W_base = model.fc1.weight.data.t().detach().clone().to(device=device, dtype=dtype)  # (d,h)
            b1_base = model.fc1.bias.data.detach().clone().to(device=device, dtype=dtype)       # (h,)


        hdim = W_base.shape[1]

        # ----- Adapter params -----
        W_tilde = nn.Parameter(torch.zeros(r, hdim, device=device, dtype=dtype)) if r > 0 else None
        delta_b1 = nn.Parameter(torch.zeros(hdim, device=device, dtype=dtype))

        # ----- New head for task t -----
        head_t = nn.Linear(hdim, 1).to(device=device, dtype=dtype)
        heads.append(head_t)

        # ----- Optimizer: ONLY subspace coords + current head -----
        params = [p for p in [W_tilde, delta_b1] if p is not None] + list(head_t.parameters())
        opt = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)

        print(f"\n=== [OURS] Training {tasks[t]['name']} (task {t}) ===")
        print(f"[OURS] t={t}: j={j}, U_span.shape={tuple(U_span.shape)}, "
              f"U.shape={(X_prev.shape[0], S.numel()) if S.numel()>0 else (0,0)}, "
              f"S.shape={tuple(S.shape)}")
        print(f"[OURS] t={t}: S_max(X) = {Smax:.6f}")

        model.train(); head_t.train()
        it_global = 0
        for ep in range(1, epochs_per_task + 1):
            for it, (xb, yb) in enumerate(tasks[t]["train"], 1):
                xb = xb.to(device=device, dtype=dtype)
                yb = yb.float().to(device=device, dtype=dtype)

                opt.zero_grad()

                if r > 0:
                    low_rank = U_span_d @ W_tilde         # (d, h)
                    W_eff = W_base + low_rank             # (d, h)
                else:
                    W_eff = W_base

                b1_eff = b1_base + delta_b1               # (h,)
                z1 = F.linear(xb, W_eff.t(), bias=b1_eff) # (B,h)
                h1 = model.act(z1)
                logits = head_t(h1).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, yb)
                loss.backward()
                opt.step()

                if log_every and (it_global := it_global + 1) % log_every == 0:
                    print(f"[OURS] epoch {ep} iter {it} loss={loss.item():.4f}")

        # ----- Commit update to trunk -----
        with torch.no_grad():
            if r > 0:
                W_base.add_(U_span_d @ W_tilde)
            b1_base.add_(delta_b1)
            model.fc1.weight.copy_(W_base.t())
            model.fc1.bias.copy_(b1_base)

        # ----- Evaluate with per-task heads -----
        for u in range(t + 1):
            acc[t, u] = eval_bin_with_head(model, heads[u], tasks[u]["test"], device)
        row = " ".join(f"{100*acc[t,u]:6.2f}%" for u in range(t + 1))
        print(f"[OURS] After task {t}: {row}")

    return model, heads, acc


if __name__ == "__main__":
    start_time = time.time()

    model, heads, acc0, tasks = train_task_0(
        epochs_per_task=5, lr=0.1, weight_decay=0.1, seed=0,
        hidden_size=256, act="relu", log_every=None
    )

    model, heads, acc_later = train_later_tasks(
        model, heads, tasks,
        epochs_per_task=5, lr=0.1, weight_decay=0.1,
        seed=0, log_every=None, eps1=1e-2, center=True, relative=False
    )

    print(f"Training time = {time.time() - start_time}")
    print("\n ------------------- \n")

