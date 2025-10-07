import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
import time

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
        x = x[m].view(-1, 28*28)
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

def train_task(model, loader, opt, device, epochs=3, log_every=None):
    model.train()
    it_global = 0
    for ep in range(1, epochs+1):
        for it,(x,y) in enumerate(loader,1):
            it_global += 1
            x,y = x.to(device), y.float().to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            opt.step()
            if log_every and it_global % log_every == 0:
                print(f"[BL] epoch {ep} iter {it} loss={loss.item():.4f}")

# ---------- Run ----------
def run_baseline(epochs_per_task=3, lr=0.1, weight_decay=0.0, seed=0, hidden_size=256, act="relu"):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    tasks = make_split_mnist(seed=seed)
    model = SimpleNet(d=784, h=hidden_size, act=act).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    T = len(tasks); acc = torch.zeros(T, T)
    for t in range(T):
        print(f"\n=== [BL] Training {tasks[t]['name']} (task {t}) ===")
        train_task(model, tasks[t]["train"], opt, device, epochs=epochs_per_task)
        for u in range(t+1):
            acc[t,u] = eval_bin(model, tasks[u]["test"], device)
        row = " ".join(f"{100*acc[t,u]:6.2f}%" for u in range(t+1))
        print(f"[BL] After task {t}: {row}")

    print("\n[BL] Final accuracy matrix:")
    for t in range(T):
        row = " ".join(f"{100*acc[t,u]:6.2f}%" if u<=t else "  ---- " for u in range(T))
        print(f"t={t}: {row}")
    return model, acc

if __name__ == "__main__":
    start_time = time.time()
    run_baseline(
        epochs_per_task=3,
        lr=0.1,
        weight_decay=0.1,
        seed=0,
        hidden_size=256, 
        act="relu"       
    )
    print(f"Training time = {time.time() - start_time}")
    print("\n ------------------- \n")
