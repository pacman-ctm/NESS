import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import time

# ----------------------------
# Data
# ----------------------------
def subset_binary(ds, a, b, flatten=True):
    x = ds.data.float() / 255.0
    y = ds.targets
    m = (y == a) | (y == b)
    x = x[m]
    y = (y[m] == b).long()
    if flatten:
        x = x.view(x.size(0), -1)
    else:
        x = x.unsqueeze(1)
    return TensorDataset(x, y)

def make_split_mnist_tasks(
    root="./data",
    train_batch_size=64,
    test_batch_size=512,
    flatten=True,
    download=True,
    num_workers=2,
    pin_memory=True,
    seed=0
):
    g = torch.Generator().manual_seed(seed)
    tfm = transforms.ToTensor()
    ds_train = datasets.MNIST(root, train=True,  transform=tfm, download=download)
    ds_test  = datasets.MNIST(root, train=False, transform=tfm, download=download)

    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9)]
    stream = []
    for i, (a,b) in enumerate(pairs, 1):
        tr = subset_binary(ds_train, a, b, flatten=flatten)
        te = subset_binary(ds_test,  a, b, flatten=flatten)
        stream.append({
            "name": f"Task {i}: {a} vs {b}",
            "classes": (a,b),
            "train": DataLoader(tr, batch_size=train_batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=pin_memory, generator=g),
            "test":  DataLoader(te, batch_size=test_batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory),
            "train_dataset": tr,  # keep X_i for SVD concat
        })
    return stream

# ----------------------------
# Debug: Print ranks
# ----------------------------
@torch.no_grad()
def _numerical_rank(A: torch.Tensor, eps: float = 1e-6) -> int:
    s = torch.linalg.svdvals(A)
    if s.numel() == 0:
        return 0
    tol = eps * max(A.shape) * s.max()
    return int((s > tol).sum().item())

def _as_2d_weight(m: nn.Module) -> torch.Tensor | None:
    W = getattr(m, "weight", None)
    if W is None:
        return None
    if isinstance(m, nn.Linear):
        return W.detach()                      # [out, in]
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return W.detach().flatten(1)           # [out, in*kH*kW*(kD)]
    if isinstance(m, nn.Embedding):
        return W.detach()                      # [num_embeddings, dim]
    return None

@torch.no_grad()
def print_weight_ranks(model: nn.Module, eps: float = 1e-6) -> None:
    items = []
    for name, module in model.named_modules():
        W2 = _as_2d_weight(module)
        if W2 is not None:
            r = _numerical_rank(W2, eps=eps)
            shape = tuple(W2.shape)
            items.append((name + ".weight", r, shape))
    line = " | ".join(f"{n}: rank={r} (shape={s})" for n, r, s in items)
    print("[RANK] " + (line if line else "no weight matrices found"))

# ----------------------------
# Network
# ----------------------------
class SVDLinearCL(nn.Module):
    def __init__(self, in_features=784, rep_dim=50, hidden_dim=50, num_tasks=5):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(in_features, rep_dim), requires_grad=False)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(rep_dim, hidden_dim),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_tasks)
        ])

    def forward_with_W(self, x, W_eff, task_id):
        z = x @ W_eff        # [N, rep_dim]
        return self.heads[task_id](z).squeeze(1)

    def forward(self, x, task_id):
        return self.forward_with_W(x, self.W, task_id)

    
    # def __init__(self, in_features=784, rep_dim=50, num_tasks=5):
    #     super().__init__()
    #     self.W = nn.Parameter(torch.zeros(in_features, rep_dim), requires_grad=False)
    #     self.heads = nn.ModuleList([nn.Linear(rep_dim, 1) for _ in range(num_tasks)])

    # def forward_with_W(self, x, W_eff, task_id):
    #     z = x @ W_eff
    #     return self.heads[task_id](z).squeeze(1)

    # def forward(self, x, task_id):
    #     return self.forward_with_W(x, self.W, task_id)

@torch.no_grad()
def eval_head(model, loader, device, task_id):
    model.eval()
    c = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).long()
        logits = model(x, task_id)
        pred = (torch.sigmoid(logits) >= 0.5).long()
        c += (pred == y).sum().item()
        n += y.numel()
    return c / max(1, n)

def build_cumulative_X_colwise(task_datasets, device="cpu"):
    xs = []
    for ds in task_datasets:
        Xi = ds.tensors[0]    # [Ni, d]
        xs.append(Xi.t())     # [d, Ni]
    X = torch.cat(xs, dim=1) if xs else None
    print(f"Shape of X = {X.shape}\n")
    return X.to(device)

def first_task_train(model, loader, device, epochs=3, lr_W=0.1, wd_W=5e-4, lr_head=0.05, wd_head=0.0):
    model.W.requires_grad_(True)
    params = [
        {"params": [model.W], "lr": lr_W, "weight_decay": wd_W},
        {"params": model.heads[0].parameters(), "lr": lr_head, "weight_decay": wd_head},
    ]
    opt = torch.optim.SGD(params, momentum=0.9, nesterov=True)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device); y = y.float().to(device)
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(x, task_id=0), y)
            loss.backward()
            opt.step()
    model.W.requires_grad_(False)

def train_task_i_projection(
    model: SVDLinearCL,
    loader,
    head_id: int,
    U_span: torch.Tensor,
    device,
    epochs=3,
    lr_A=0.1,
    wd_A=1e-4,
    lr_head=0.05,
    wd_head=0.0
):
    d, p = model.W.shape
    k = U_span.shape[1]
    if k == 0:
        params = [{"params": model.heads[head_id].parameters(), "lr": lr_head, "weight_decay": wd_head}]
        opt = torch.optim.SGD(params, momentum=0.9, nesterov=True)
        for _ in range(epochs):
            model.train()
            for x, y in loader:
                x = x.to(device); y = y.float().to(device)
                opt.zero_grad()
                loss = F.binary_cross_entropy_with_logits(model(x, head_id), y)
                loss.backward()
                opt.step()
        return

    A = nn.Parameter(torch.zeros(k, p, device=device), requires_grad=True)
    params = [
        {"params": [A], "lr": lr_A, "weight_decay": wd_A},
        {"params": model.heads[head_id].parameters(), "lr": lr_head, "weight_decay": wd_head},
    ]
    opt = torch.optim.SGD(params, momentum=0.9, nesterov=True)

    U_span = U_span.to(device)
    W_fixed = model.W.detach()

    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x = x.to(device); y = y.float().to(device)
            opt.zero_grad()
            W_eff = W_fixed + U_span @ A
            logits = model.forward_with_W(x, W_eff, head_id)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()

    with torch.no_grad():
        model.W.data += U_span @ A.detach()

# ----------------------------
# Main experiment loop
# ----------------------------
def run_training(
    rep_dim=50,
    eps_1=1e-4,
    epochs_per_task=3,
    lr_W=0.1,
    wd_W=5e-4,
    lr_A=0.1,
    wd_A=1e-4,
    lr_head=0.05,
    wd_head=0.0,
    seed=0
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = make_split_mnist_tasks(seed=seed)

    T = len(tasks)
    model = SVDLinearCL(in_features=784, rep_dim=rep_dim, num_tasks=T).to(device)
    acc = torch.zeros((T, T))

    # Task 1: fit W and head_0
    print(f"\n=== [SVD-CL] Training first task: {tasks[0]['name']} ===")
    first_task_train(
        model, tasks[0]["train"], device,
        epochs=epochs_per_task, lr_W=lr_W, wd_W=wd_W, lr_head=lr_head, wd_head=wd_head
    )
    acc[0,0] = eval_head(model, tasks[0]["test"], device, task_id=0)
    print(f"[SVD-CL] After task 0: {100*acc[0,0]:6.2f}%")
    print_weight_ranks(model)

    # Next tasks: projection updates
    seen_train_sets = [tasks[0]["train_dataset"]]
    for t in range(1, T):
        print(f"\n=== [SVD-CL] Preparing subspace for {tasks[t]['name']} (task {t}) ===")
        seen_train_sets.append(tasks[t-1]["train_dataset"])  # ensure cumulative up to t-1
        with torch.no_grad():
            X = build_cumulative_X_colwise(seen_train_sets, device="cpu")
            U, S, Vh = torch.linalg.svd(X, full_matrices=True)
            below = torch.where(S < eps_1)[0]
            if below.numel() == 0:
                U_span = U[:, :0]
                print(f"  No singular values < {eps_1:g}; k=0 (head-only training).")
            else:
                j = int(below[0].item())
                U_span = U[:, j:]
                print(f"  First s_j < {eps_1:g} at j={j}; using k={U_span.shape[1]} columns.")

        train_task_i_projection(
            model, tasks[t]["train"], head_id=t, U_span=U_span, device=device,
            epochs=epochs_per_task, lr_A=lr_A, wd_A=wd_A, lr_head=lr_head, wd_head=wd_head
        )

        for u in range(t + 1):
            acc[t, u] = eval_head(model, tasks[u]["test"], device, task_id=u)
        row = " ".join([f"{100*acc[t,u]:6.2f}%" for u in range(t + 1)])
        print(f"[SVD-CL] After task {t}: {row}")
        print_weight_ranks(model)

    print("\n[SVD-CL] Final accuracy matrix (rows: after task t, cols: u ≤ t):")
    for t in range(T):
        row = " ".join([f"{100*acc[t,u]:6.2f}%" if u <= t else "  ---- " for u in range(T)])
        print(f"t={t}: {row}")
    return model, acc

if __name__ == "__main__":
    start_time = time.time()
    run_training(
        rep_dim=50,
        eps_1=1e-2,
        epochs_per_task=10,
        lr_W=0.1, wd_W=5e-4,
        lr_A=0.1, wd_A=1e-4,
        lr_head=0.05, wd_head=0.0,
        seed=0
    )
    print(f"Training time = {time.time() - start_time}")
    print("\n ------------------- \n")
