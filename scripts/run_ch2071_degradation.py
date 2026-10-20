"""Replicate the degradation experiment of He et al. (2016) on CIFAR-10.

Trains plain and residual CIFAR networks of depth 6n+2 with the protocol of
Section 4.2 of "Deep Residual Learning for Image Recognition":

* three stages of 2n 3x3 convolutions with 16, 32, 64 filters,
* batch normalization after every convolution, He (Kaiming) normal init,
* option A shortcuts (identity, zero padded channels when the width doubles),
* SGD, batch 128 (the last partial batch of each epoch is dropped), momentum 0.9, weight decay 1e-4, learning rate 0.1 divided
  by 10 at 50% and 75% of training (32k and 48k of 64k iterations),
* 4 pixel zero padding, random 32x32 crop and horizontal flip.

The whole dataset lives on the GPU and augmentation runs on the GPU, so an
epoch of a 56 layer network takes a few seconds on a laptop GPU. After each
run the script also records the per layer response standard deviations of
Figure 7, per layer gradient norms at initialization, and a block lesion study
in the style of Veit et al. (2016). Results are appended to a JSON file after
every run, so an interrupted job resumes where it stopped.

Usage:
    python scripts/run_ch2071_degradation.py --epochs 164 --depths 20 32 44 56 110
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def load_cifar10(root: Path, device: torch.device):
    import torchvision

    tr = torchvision.datasets.CIFAR10(root=str(root), train=True, download=True)
    te = torchvision.datasets.CIFAR10(root=str(root), train=False, download=True)

    def to_tensor(ds):
        x = torch.tensor(ds.data).permute(0, 3, 1, 2).float().div_(255.0)
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        x = (x - mean) / std
        y = torch.tensor(ds.targets, dtype=torch.long)
        return x.to(device), y.to(device)

    return to_tensor(tr), to_tensor(te)


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Zero pad by 4, random 32x32 crop, random horizontal flip, on device."""
    b = x.shape[0]
    padded = F.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ox = torch.randint(0, 9, (b,), device=x.device, generator=gen)
    ar = torch.arange(32, device=x.device)
    rows = (oy[:, None] + ar)[:, None, :, None]
    cols = (ox[:, None] + ar)[:, None, None, :]
    bi = torch.arange(b, device=x.device)[:, None, None, None]
    ci = torch.arange(3, device=x.device)[None, :, None, None]
    out = padded[bi, ci, rows, cols]
    flip = torch.rand(b, device=x.device, generator=gen) < 0.5
    out[flip] = out[flip].flip(3)
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


class ShortcutA(nn.Module):
    """Parameter free shortcut: subsample and zero pad the new channels."""

    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.stride = stride
        self.pad = out_ch - in_ch

    def forward(self, x):
        if self.stride > 1:
            x = x[:, :, :: self.stride, :: self.stride]
        if self.pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.pad // 2, self.pad - self.pad // 2))
        return x


class Block(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, residual: bool):
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = ShortcutA(in_ch, out_ch, stride)
        self.lesioned = False

    def forward(self, x):
        if self.lesioned:
            return self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.residual:
            out = out + self.shortcut(x)
        return F.relu(out)


class CifarNet(nn.Module):
    def __init__(self, depth: int, residual: bool, num_classes: int = 10):
        super().__init__()
        assert (depth - 2) % 6 == 0, "depth must be 6n+2"
        n = (depth - 2) // 6
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        blocks = []
        in_ch = 16
        for stage, width in enumerate((16, 32, 64)):
            for i in range(n):
                stride = 2 if (stage > 0 and i == 0) else 1
                blocks.append(Block(in_ch, width, stride, residual))
                in_ch = width
        self.blocks = nn.ModuleList(blocks)
        self.fc = nn.Linear(64, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        for b in self.blocks:
            x = b(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, x, y, batch=1000):
    model.eval()
    wrong, loss = 0, 0.0
    for i in range(0, x.shape[0], batch):
        xb = x[i : i + batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
        loss += F.cross_entropy(logits.float(), y[i : i + batch], reduction="sum").item()
        wrong += (logits.argmax(1) != y[i : i + batch]).sum().item()
    return wrong / x.shape[0], loss / x.shape[0]


@torch.no_grad()
def layer_response_std(model, x, batch=1000):
    """Std of each 3x3 layer output after BN, before ReLU or addition (Fig. 7)."""
    model.eval()
    bns = [model.bn] + [m for b in model.blocks for m in (b.bn1, b.bn2)]
    sums = [0.0] * len(bns)
    sq = [0.0] * len(bns)
    counts = [0] * len(bns)
    hooks = []
    for idx, bn in enumerate(bns):
        def hook(_m, _i, out, idx=idx):
            o = out.float()
            sums[idx] += o.sum().item()
            sq[idx] += (o * o).sum().item()
            counts[idx] += o.numel()
        hooks.append(bn.register_forward_hook(hook))
    for i in range(0, x.shape[0], batch):
        model(x[i : i + batch])
    for h in hooks:
        h.remove()
    out = []
    for s, q, c in zip(sums, sq, counts):
        m = s / c
        out.append(math.sqrt(max(q / c - m * m, 0.0)))
    return out


def grad_norms(model, x, y):
    """Per convolution weight gradient norm on one batch, in fp32."""
    model.train()
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    convs = [model.conv] + [c for b in model.blocks for c in (b.conv1, b.conv2)]
    norms = [c.weight.grad.norm().item() for c in convs]
    model.zero_grad(set_to_none=True)
    return norms


@torch.no_grad()
def lesion_study(model, x, y):
    errs = []
    for b in model.blocks:
        b.lesioned = True
        err, _ = evaluate(model, x, y)
        b.lesioned = False
        errs.append(err)
    return errs


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


class GraphStep:
    """One SGD step (forward, backward, update) captured as a CUDA graph.

    Kernel launch overhead dominates these small networks on Windows, so the
    whole step is recorded once and replayed. The learning rate is baked into
    the graph, so the step is recaptured whenever the schedule changes it.
    """

    def __init__(self, model, opt, bs, device):
        self.model, self.opt = model, opt
        self.x = torch.zeros(bs, 3, 32, 32, device=device)
        self.y = torch.zeros(bs, dtype=torch.long, device=device)
        self.graph, self.lr = None, None

    def _forward(self):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = self.model(self.x)
            loss = F.cross_entropy(logits.float(), self.y)
        return loss, (logits.argmax(1) != self.y).sum()

    def eager(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.zero_grad(set_to_none=True)
        loss, wrong = self._forward()
        loss.backward()
        self.opt.step()
        return loss.detach(), wrong

    def capture(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.graph = torch.cuda.CUDAGraph()
        self.opt.zero_grad(set_to_none=True)
        with torch.cuda.graph(self.graph):
            self.loss, self.wrong = self._forward()
            self.loss.backward()
            self.opt.step()
        self.lr = lr

    def __call__(self, xb, yb, lr):
        self.x.copy_(xb)
        self.y.copy_(yb)
        if self.graph is None:
            # Warm up on a side stream (required before capture), then capture.
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                out = self.eager(lr)
            torch.cuda.current_stream().wait_stream(s)
            self.capture(lr)
            return out
        if lr != self.lr:
            self.capture(lr)
        self.graph.replay()
        return self.loss.detach().clone(), self.wrong.clone()


def train_one(depth, residual, seed, epochs, data, device, log):
    (xtr, ytr), (xte, yte) = data
    torch.manual_seed(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model = CifarNet(depth, residual).to(device)

    probe = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(0))[:256]
    init_grads = grad_norms(model, xtr[probe.to(device)], ytr[probe.to(device)])

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    m1, m2 = int(round(epochs * 0.5)), int(round(epochs * 0.75))
    bs = 128
    n = xtr.shape[0]
    steps_per_epoch = n // bs  # drop the last partial batch: static shapes for the graph
    # He et al. warm up the 110 layer net with lr 0.01 for about 400 iterations.
    warmup_steps = 400 if depth >= 110 else 0
    stepper = GraphStep(model, opt, bs, device)
    history = []
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        base_lr = 0.1 * (0.1 ** ((ep >= m1) + (ep >= m2)))
        model.train()
        perm = torch.randperm(n, device=device, generator=gen)
        xs = torch.empty_like(xtr)
        for i in range(0, n, 5000):
            xs[i : i + 5000] = augment(xtr[perm[i : i + 5000]], gen)
        ys = ytr[perm]
        tot_loss = torch.zeros((), device=device)
        tot_wrong = torch.zeros((), device=device, dtype=torch.long)
        for i in range(steps_per_epoch):
            lr = 0.01 if step < warmup_steps else base_lr
            loss, wrong = stepper(xs[i * bs : (i + 1) * bs], ys[i * bs : (i + 1) * bs], lr)
            tot_loss += loss * bs
            tot_wrong += wrong
            step += 1
        seen = steps_per_epoch * bs
        model.eval()
        test_err, test_loss = evaluate(model, xte, yte)
        rec = {
            "epoch": ep + 1,
            "train_loss": tot_loss.item() / seen,
            "train_err": tot_wrong.item() / seen,
            "test_err": test_err,
            "test_loss": test_loss,
            "lr": base_lr,
            "seconds": time.time() - t0,
        }
        history.append(rec)
        if (ep + 1) % 10 == 0 or ep == 0 or ep + 1 == epochs:
            log(
                f"  {'res' if residual else 'plain'}-{depth} s{seed} ep {ep+1:3d} "
                f"train_err {rec['train_err']:.4f} test_err {test_err:.4f} "
                f"({rec['seconds']:.0f}s)"
            )
    stepper.graph = None
    torch.cuda.synchronize()

    # Clean (non augmented) training error at the end, as an optimization measure.
    clean_train_err, clean_train_loss = evaluate(model, xtr, ytr)
    result = {
        "depth": depth,
        "residual": residual,
        "seed": seed,
        "epochs": epochs,
        "params": sum(p.numel() for p in model.parameters()),
        "history": history,
        "final_test_err": history[-1]["test_err"],
        "final_train_err_clean": clean_train_err,
        "final_train_loss_clean": clean_train_loss,
        "init_grad_norms": init_grads,
        "final_grad_norms": grad_norms(model, xtr[probe.to(device)], ytr[probe.to(device)]),
        "layer_response_std": layer_response_std(model, xte),
        "lesion_test_err": lesion_study(model, xte, yte),
        "wall_seconds": time.time() - t0,
    }
    return result, model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depths", type=int, nargs="+", default=[20, 32, 44, 56, 110])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--epochs", type=int, default=164)
    p.add_argument("--data", type=Path, default=Path("data/cifar10"))
    p.add_argument("--out", type=Path, default=Path("results/ch2071_cifar_degradation.json"))
    p.add_argument("--ckpt", type=Path, default=Path("results/ch2071_ckpt"))
    p.add_argument("--bench", action="store_true", help="time 2 epochs per model and exit")
    args = p.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    data = load_cifar10(args.data, device)

    def log(msg):
        print(msg, flush=True)

    if args.bench:
        for depth in args.depths:
            for residual in (False, True):
                r, _ = train_one(depth, residual, 0, 2, data, device, log)
                log(f"BENCH {'res' if residual else 'plain'}-{depth}: {r['wall_seconds']/2:.1f}s/epoch")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.ckpt.mkdir(parents=True, exist_ok=True)
    store = {"protocol": vars(args) | {"data": str(args.data), "out": str(args.out), "ckpt": str(args.ckpt)}, "runs": []}
    if args.out.exists():
        store = json.loads(args.out.read_text())
    done = {(r["depth"], r["residual"], r["seed"], r["epochs"]) for r in store["runs"]}
    store["device"] = torch.cuda.get_device_name(0)
    store["torch"] = torch.__version__

    for seed in args.seeds:
        for depth in args.depths:
            for residual in (False, True):
                key = (depth, residual, seed, args.epochs)
                if key in done:
                    continue
                log(f"=== {'res' if residual else 'plain'}-{depth} seed {seed}")
                result, model = train_one(depth, residual, seed, args.epochs, data, device, log)
                torch.save(
                    model.state_dict(),
                    args.ckpt / f"{'res' if residual else 'plain'}{depth}_s{seed}_e{args.epochs}.pt",
                )
                store["runs"].append(result)
                args.out.write_text(json.dumps(store))
                log(
                    f"DONE {'res' if residual else 'plain'}-{depth} s{seed}: "
                    f"test_err {result['final_test_err']:.4f} "
                    f"clean_train_err {result['final_train_err_clean']:.4f} "
                    f"in {result['wall_seconds']/60:.1f} min"
                )


if __name__ == "__main__":
    main()
