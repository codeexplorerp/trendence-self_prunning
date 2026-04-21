
"""
Self-Pruning Neural Network on CIFAR-10  —  GPU-Optimized Version
Tredence AI Engineering Case Study

GPU optimizations applied:
  • Large batch size (512) to saturate GPU throughput
  • pin_memory=True + non_blocking=True for async CPU→GPU transfers
  • torch.compile() (PyTorch 2.x) for kernel fusion
  • torch.backends.cudnn.benchmark=True for cuDNN auto-tuner
  • Mixed-precision training with torch.amp (FP16 forward, FP32 params)
  • num_workers=4 for parallel data loading
  • Timing printed per epoch so you can see GPU speedup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import sys

# ─────────────────────────────────────────────────────────────────────────────
# GPU Setup & Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def setup_device():
    if not torch.cuda.is_available():
        print("[WARNING] CUDA not found — falling back to CPU.")
        print("          Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/")
        return torch.device("cpu"), False

    device = torch.device("cuda")
    gpu = torch.cuda.get_device_properties(0)

    print("=" * 60)
    print("  GPU Detected")
    print(f"  Name        : {gpu.name}")
    print(f"  VRAM        : {gpu.total_memory / 1024**3:.1f} GB")
    print(f"  CUDA cores  : {gpu.multi_processor_count} SMs")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  CUDA version: {torch.version.cuda}")
    print("=" * 60)

    # cuDNN auto-tuner — finds fastest conv kernels for your GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled   = True

    return device, True


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    Custom linear layer with per-weight learnable sigmoid gates.

    Forward pass:
        gates         = sigmoid(gate_scores)          shape: (out, in)
        pruned_weights = weight * gates               shape: (out, in)
        output         = x @ pruned_weights.T + bias

    When sigmoid(gate_scores_ij) → 0, weight_ij is effectively pruned.
    gate_scores is a full nn.Parameter so gradients flow via autograd.
    Works seamlessly with torch.amp mixed precision.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight + bias (same init as nn.Linear)
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight
        # Init at 0 → initial gates = sigmoid(0) = 0.5 (half-open)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid squashes gate_scores → (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Element-wise gate: zeroed gate_scores → zero effective weight
        pruned_weights = self.weight * gates

        # Standard linear op — gradients flow through weight AND gate_scores
        return F.linear(x, pruned_weights, self.bias)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def get_gates(self) -> torch.Tensor:
        """Gate values detached from the compute graph (for metrics/plots)."""
        return torch.sigmoid(self.gate_scores)

    def sparsity_loss(self) -> torch.Tensor:
        """L1 penalty = sum of all sigmoid gate values in this layer."""
        return torch.sigmoid(self.gate_scores).sum()

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}"


# ─────────────────────────────────────────────────────────────────────────────
# Network Definition
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    4-layer MLP for CIFAR-10.
    All linear layers are PrunableLinear — gates learned end-to-end.

    Input (3×32×32 = 3072) → 512 → 256 → 128 → 10 (logits)

    BatchNorm is placed BEFORE activation for training stability.
    Dropout adds light regularization on top of gate sparsity.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.fc1 = PrunableLinear(3 * 32 * 32, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)

        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)                      # flatten: (B, 3072)

        x = self.drop(F.relu(self.bn1(self.fc1(x))))   # → (B, 512)
        x = self.drop(F.relu(self.bn2(self.fc2(x))))   # → (B, 256)
        x = self.drop(F.relu(self.bn3(self.fc3(x))))   # → (B, 128)
        x = self.fc4(x)                                 # → (B, 10) logits
        return x

    def prunable_layers(self):
        """Yield every PrunableLinear module."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m

    def sparsity_loss(self) -> torch.Tensor:
        """Aggregate L1 sparsity loss across all prunable layers."""
        return sum(layer.sparsity_loss() for layer in self.prunable_layers())

    @torch.no_grad()
    def get_all_gate_values(self) -> np.ndarray:
        """Concatenated flat array of every gate value (CPU numpy)."""
        parts = [layer.get_gates().cpu().float().numpy().ravel()
                 for layer in self.prunable_layers()]
        return np.concatenate(parts)

    @torch.no_grad()
    def compute_sparsity(self, threshold: float = 1e-2) -> float:
        """Fraction (%) of gates below threshold = effectively pruned weights."""
        g = self.get_all_gate_values()
        return float((g < threshold).mean()) * 100.0

    def count_parameters(self) -> dict:
        total   = sum(p.numel() for p in self.parameters())
        weights = sum(p.numel() for n, p in self.named_parameters()
                      if "weight" in n and "gate" not in n)
        gates   = sum(p.numel() for n, p in self.named_parameters()
                      if "gate" in n)
        return {"total": total, "weights": weights, "gates": gates}


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading  —  GPU-aware pipeline
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 512, num_workers: int = 4):
    """
    Returns train/test DataLoaders optimised for GPU:
      - pin_memory=True : keeps tensors in page-locked RAM → faster DMA to GPU
      - persistent_workers=True : worker processes stay alive between epochs
      - prefetch_factor=2 : each worker pre-fetches 2 batches
    """
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=transform_train)
    test_ds  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test)

    loader_kwargs = dict(
        pin_memory=True,          # page-locked memory for fast GPU transfers
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(
        test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    print(f"\n  Dataset   : CIFAR-10")
    print(f"  Train     : {len(train_ds):,} samples  ({len(train_loader)} batches @ bs={batch_size})")
    print(f"  Test      : {len(test_ds):,} samples  ({len(test_loader)} batches)")
    print(f"  Workers   : {num_workers}")

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# Training  —  Mixed-Precision (AMP)
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scaler, lam, device, use_amp):
    """
    One epoch with optional mixed-precision (AMP):
      - autocast : runs forward pass in FP16 (faster on Tensor Core GPUs)
      - GradScaler: scales loss to avoid FP16 underflow during backprop
    """
    model.train()
    total_loss = ce_total = sp_total = correct = total = 0

    for imgs, labels in loader:
        # non_blocking=True overlaps data transfer with GPU compute
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            logits  = model(imgs)
            ce_loss = F.cross_entropy(logits, labels)
            sp_loss = model.sparsity_loss()
            loss    = ce_loss + lam * sp_loss

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = labels.size(0)
        total      += bs
        total_loss += loss.item()    * bs
        ce_total   += ce_loss.item() * bs
        sp_total   += sp_loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()

    n = total
    return total_loss/n, ce_total/n, sp_total/n, correct/n * 100.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        preds  = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Full Experiment for One Lambda
# ─────────────────────────────────────────────────────────────────────────────

def run_experiment(lam, train_loader, test_loader, device,
                   use_amp=True, epochs=25, seed=42):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    model = SelfPruningNet(dropout=0.3).to(device)

    # torch.compile() requires Triton, which is NOT available on Windows.
    # It works on Linux/macOS — skipped automatically here on Windows.
    compiled_model = model
    if hasattr(torch, "compile") and device.type == "cuda" and sys.platform != "win32":
        try:
            compiled_model = torch.compile(model, mode="reduce-overhead")
            print("  torch.compile() : enabled (reduce-overhead mode)")
        except Exception as e:
            print(f"  torch.compile() : skipped ({e})")
    else:
        reason = "Windows — Triton not supported" if sys.platform == "win32" else "CPU mode"
        print(f"  torch.compile() : skipped ({reason})")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    scaler    = GradScaler(enabled=use_amp)

    params = model.count_parameters()
    print(f"\n{'='*60}")
    print(f"  λ = {lam}  |  {epochs} epochs  |  AMP={'on' if use_amp else 'off'}")
    print(f"  Parameters — weights: {params['weights']:,}  "
          f"| gates: {params['gates']:,}  | total: {params['total']:,}")
    print(f"{'='*60}")

    history = {"loss": [], "ce": [], "sp": [], "train_acc": [], "sparsity": []}
    epoch_times = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        loss, ce, sp, tr_acc = train_one_epoch(
            compiled_model, train_loader, optimizer, scaler, lam, device, use_amp)
        scheduler.step()

        sparsity = model.compute_sparsity()   # use non-compiled model for metrics
        dt = time.time() - t0
        epoch_times.append(dt)

        history["loss"].append(loss)
        history["ce"].append(ce)
        history["sp"].append(sp)
        history["train_acc"].append(tr_acc)
        history["sparsity"].append(sparsity)

        if epoch % 5 == 0 or epoch == 1:
            # VRAM usage
            vram_str = ""
            if torch.cuda.is_available():
                used  = torch.cuda.memory_allocated() / 1024**2
                total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                vram_str = f"  VRAM={used:.0f}/{total:.0f}MB"

            print(f"  Epoch {epoch:02d}/{epochs}  "
                  f"Loss={loss:.4f}  CE={ce:.4f}  SP={sp:.1f}  "
                  f"Acc={tr_acc:.1f}%  Sparse={sparsity:.1f}%  "
                  f"({dt:.1f}s){vram_str}")

    test_acc = evaluate(model, test_loader, device)
    sparsity = model.compute_sparsity()
    gates    = model.get_all_gate_values()

    avg_time = np.mean(epoch_times)
    print(f"\n  ✔ Test Accuracy  : {test_acc:.2f}%")
    print(f"  ✔ Sparsity Level : {sparsity:.2f}%  (gates < 0.01)")
    print(f"  ✔ Avg epoch time : {avg_time:.2f}s")

    return model, test_acc, sparsity, gates, history


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(gates_dict: dict, results: dict,
                           save_path: str = "gate_distribution.png"):
    lambdas = list(gates_dict.keys())
    n = len(lambdas)
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, lam, color in zip(axes, lambdas, colors):
        gates = gates_dict[lam]
        acc, sp = results[lam]

        ax.hist(gates, bins=80, color=color, alpha=0.82,
                edgecolor="white", linewidth=0.3)
        ax.axvline(0.01, color="black", linestyle="--",
                   linewidth=1.5, label="Prune threshold (0.01)")

        pruned_pct = (gates < 0.01).mean() * 100
        ax.set_title(f"λ = {lam}\nAcc={acc:.1f}%  |  Pruned={pruned_pct:.1f}%",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Gate value  [sigmoid output]", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(fontsize=9)
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle("Gate Value Distribution — Self-Pruning Network (CIFAR-10)",
                 fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved → {save_path}")


def plot_training_curves(history_dict: dict,
                         save_path: str = "training_curves.png"):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]
    titles = ["Cross-Entropy Loss", "Sparsity (%) over Training",
              "Train Accuracy (%)"]

    keys = ["ce", "sparsity", "train_acc"]

    for ax, key, title in zip(axes, keys, titles):
        for (lam, history), color in zip(history_dict.items(), colors):
            epochs = range(1, len(history[key]) + 1)
            ax.plot(epochs, history[key], color=color,
                    label=f"λ={lam}", linewidth=2.2)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    device, has_gpu = setup_device()

    # Mixed precision only on CUDA (FP16 is slower on CPU)
    use_amp = has_gpu

    # GPU → bigger batch, more workers; CPU → keep it small
    batch_size  = 512 if has_gpu else 128
    num_workers = 4   if has_gpu else 0
    epochs      = 25

    train_loader, test_loader = get_cifar10_loaders(batch_size, num_workers)

    # Three lambda values: low / medium / high sparsity pressure
    lambdas = [1e-5, 1e-4, 1e-3]

    results      = {}   # lam → (test_acc, sparsity)
    gates_dict   = {}   # lam → np.ndarray
    history_dict = {}   # lam → history

    total_start = time.time()

    for lam in lambdas:
        if has_gpu:
            torch.cuda.empty_cache()   # free VRAM between experiments

        model, test_acc, sparsity, gates, history = run_experiment(
            lam, train_loader, test_loader, device,
            use_amp=use_amp, epochs=epochs)

        results[lam]      = (test_acc, sparsity)
        gates_dict[lam]   = gates
        history_dict[lam] = history

    wall_time = time.time() - total_start

    # ── Summary Table ─────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print(f"  {'Lambda':<12}  {'Test Accuracy':>14}  {'Sparsity (%)':>14}")
    print("=" * 58)
    for lam, (acc, sp) in results.items():
        tag = "  ← best" if acc == max(a for a, _ in results.values()) else ""
        print(f"  {lam:<12}  {acc:>13.2f}%  {sp:>13.2f}%{tag}")
    print("=" * 58)
    print(f"\n  Total wall time : {wall_time/60:.1f} min")

    if has_gpu:
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak VRAM used  : {peak_mb:.0f} MB")

    # ── Plots ─────────────────────────────────────────────────────────────
    plot_gate_distribution(gates_dict, results, "gate_distribution.png")
    plot_training_curves(history_dict, "training_curves.png")

    print("\n  Done! Check gate_distribution.png and training_curves.png")


if __name__ == "__main__":
    main()
