"""ArogyaNet (own design, no pretrained weights) and the transfer-learning wrappers.

ArogyaNet design decisions (all switchable so each can be ablated):
  D1  residual (skip) connections around every block                       -> use_skip
  D2  squeeze-and-excitation channel attention inside every block          -> use_se
  (fixed) depthwise-separable convolutions to keep params / CPU latency low, global average pooling instead of
          flatten+dense (translation-robust, no resolution-dependent dense layer).
Optional two-branch version: three hand-crafted numbers -> small dense branch concatenated before the classifier.
All models take x in [0,1] of shape (B,1,H,W) (+ meta (B,3) standardised) and return logits (B,)."""
import os
import torch, torch.nn as nn, torch.nn.functional as F


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        h = max(c // r, 4)
        self.fc1, self.fc2 = nn.Linear(c, h), nn.Linear(h, c)

    def forward(self, x):
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean((2, 3))))))
        return x * w[:, :, None, None]


class Block(nn.Module):
    """depthwise 3x3 -> BN -> ReLU -> pointwise 1x1 -> BN -> [SE] -> (+ skip) -> ReLU"""
    def __init__(self, cin, cout, stride, use_skip, use_se):
        super().__init__()
        self.dw = nn.Sequential(nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.ReLU(inplace=True))
        self.pw = nn.Sequential(nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        self.se = SE(cout) if use_se else nn.Identity()
        if use_skip:
            self.skip = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(
                nn.AvgPool2d(stride) if stride > 1 else nn.Identity(), nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout))
        else:
            self.skip = None

    def forward(self, x):
        y = self.se(self.pw(self.dw(x)))
        if self.skip is not None:
            y = y + self.skip(x)
        return F.relu(y)


class ArogyaNet(nn.Module):
    def __init__(self, widths=(24, 48, 96, 192), use_skip=True, use_se=True, meta_dim=0, dropout=0.2, meta_hidden=16):
        super().__init__()
        w0, w1, w2, w3 = widths
        self.stem = nn.Sequential(nn.Conv2d(1, w0, 3, 2, 1, bias=False), nn.BatchNorm2d(w0), nn.ReLU(inplace=True))
        plan = [(w0, w1, 2), (w1, w1, 1), (w1, w2, 2), (w2, w2, 1), (w2, w3, 2), (w3, w3, 1)]
        self.blocks = nn.Sequential(*[Block(a, b, s, use_skip, use_se) for a, b, s in plan])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.meta_dim = meta_dim
        self.meta = nn.Sequential(nn.Linear(meta_dim, meta_hidden), nn.ReLU(inplace=True), nn.Linear(meta_hidden, meta_hidden), nn.ReLU(inplace=True)) if meta_dim else None
        self.embed_dim = w3 + (meta_hidden if meta_dim else 0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.embed_dim, 1)

    def embed(self, x, meta=None):
        f = self.pool(self.blocks(self.stem((x - 0.5) / 0.25))).flatten(1)
        if self.meta is not None:
            f = torch.cat([f, self.meta(meta)], 1)
        return f

    def forward(self, x, meta=None):
        return self.fc(self.drop(self.embed(x, meta))).squeeze(1)


class Transfer(nn.Module):
    """ImageNet-pretrained backbone; the grey image is repeated to 3 channels and ImageNet-normalised."""
    def __init__(self, name="resnet18", pretrained=True, dropout=0.2):
        super().__init__()
        import torchvision.models as tvm
        pretrained = pretrained and os.environ.get("MEDISCAN_NO_PRETRAINED") != "1"   # =1 only for offline smoke tests
        if name == "resnet18":
            m = tvm.resnet18(weights="IMAGENET1K_V1" if pretrained else None); d = m.fc.in_features; m.fc = nn.Identity()
        elif name == "mobilenet_v3_small":
            m = tvm.mobilenet_v3_small(weights="IMAGENET1K_V1" if pretrained else None); d = m.classifier[0].in_features; m.classifier = nn.Identity()
        elif name == "efficientnet_b0":
            m = tvm.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None); d = m.classifier[1].in_features; m.classifier = nn.Identity()
        else:
            raise ValueError(name)
        self.backbone, self.embed_dim, self.meta_dim = m, d, 0
        self.drop, self.fc = nn.Dropout(dropout), nn.Linear(d, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def embed(self, x, meta=None):
        return self.backbone((x.repeat(1, 3, 1, 1) - self.mean) / self.std)

    def forward(self, x, meta=None):
        return self.fc(self.drop(self.embed(x))).squeeze(1)


def build_model(cfg: dict):
    a = cfg["arch"]
    if a == "arogya":
        return ArogyaNet(use_skip=cfg.get("use_skip", True), use_se=cfg.get("use_se", True),
                         meta_dim=3 if cfg.get("meta") else 0, dropout=cfg.get("dropout", 0.2))
    return Transfer(a, pretrained=cfg.get("pretrained", True), dropout=cfg.get("dropout", 0.2))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def arch_table(model, res=224):
    """Layer / output shape / #params for every leaf module, in execution order (forward hooks)."""
    import pandas as pd
    rows, hooks = [], []
    names = {m: n for n, m in model.named_modules()}
    def hook(mod, inp, out):
        shape = tuple(out.shape[1:]) if torch.is_tensor(out) else "-"
        rows.append(dict(layer=f"{names[mod]} ({type(mod).__name__})", output_shape=str(shape), params=sum(p.numel() for p in mod.parameters(recurse=False))))
    for m in model.modules():
        if len(list(m.children())) == 0:
            hooks.append(m.register_forward_hook(hook))
    was = model.training; model.eval()
    with torch.no_grad():
        x = torch.zeros(1, 1, res, res); mt = torch.zeros(1, 3) if getattr(model, "meta_dim", 0) else None
        model(x, mt)
    for h in hooks: h.remove()
    model.train(was)
    df = pd.DataFrame(rows)
    df = df[df.params > 0].reset_index(drop=True)          # parameter-bearing layers (activations/pools omitted)
    df.loc[len(df)] = dict(layer="TOTAL", output_shape="", params=int(df.params.sum()))
    return df
