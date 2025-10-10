# backbones.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import warnings
import torch
import torch.nn as nn
import timm

@dataclass
class EncoderSpec:
    name: str = "resnet50"
    n_channels: int = 3
    out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4)
    pretrained: bool = True
    pretrained_path: Optional[str] = None
    use_input_adapter: bool = False        # map N→3 before backbone
    freeze: bool = False                   # freeze all backbone params


def _adapt_conv1_weight(sd: Dict[str, torch.Tensor], n_channels: int) -> Dict[str, torch.Tensor]:
    """
    If the checkpoint is 3‑channel and our model is n_channels != 3,
    average conv1 weights across input-channel dim and repeat.
    """
    # Try common keys
    conv1_keys = [k for k in sd.keys() if k.endswith("conv1.weight")]
    if not conv1_keys:
        return sd  # nothing to adapt (not all models use 'conv1')
    k = conv1_keys[0]
    w = sd[k]                # [out, in, k, k]
    cin_src = w.shape[1]
    if cin_src == n_channels:
        return sd
    w_avg = w.mean(1, keepdim=True)        # [out,1,k,k]
    sd[k] = w_avg.repeat(1, n_channels, 1, 1) * (cin_src / n_channels)
    return sd


class InputAdapter(nn.Module):
    """
    Optional learnable adapter to map arbitrary N-channel input to 3-channel for
    maximum compatibility with ImageNet-pretrained backbones.
    """
    def __init__(self, in_ch: int, mid_norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, 3, kernel_size=1, bias=False)]
        if mid_norm:
            layers.append(nn.BatchNorm2d(3, affine=True))
        layers.append(nn.ReLU(inplace=True))
        self.proj = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class TimmEncoder(nn.Module):
    """
    Uniform wrapper over timm encoders with features_only=True.

    Exposes:
      - channels: List[int]   (per out_index)
      - strides:  List[int]   (spatial downsample factors per out_index)
      - feature_info:         (timm FeatureInfo)
      - forward(x) -> Tuple[torch.Tensor, ...]
    """
    def __init__(self, spec: EncoderSpec):
        super().__init__()
        self.spec = spec

        # Decide backbone input channels
        target_in = 3 if spec.use_input_adapter else spec.n_channels
        self.input_adapter = (
            InputAdapter(spec.n_channels) if (spec.use_input_adapter and spec.n_channels != 3)
            else nn.Identity()
        )

        # First create an uninitialized features-only network with target_in channels
        net = timm.create_model(
            spec.name,
            pretrained=False,
            features_only=True,
            out_indices=spec.out_indices,
            in_chans=target_in
        )

        # Load weights
        if spec.pretrained_path is not None:
            sd = torch.load(spec.pretrained_path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            # Remove classifier weights (common for ResNets/EffNets etc.)
            sd = {k: v for k, v in sd.items() if not k.startswith(("fc.", "classifier.", "head."))}
            if not spec.use_input_adapter and target_in != 3:
                sd = _adapt_conv1_weight(sd, target_in)
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing or unexpected:
                warnings.warn(f"[TimmEncoder] Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
        elif spec.pretrained:
            # Let timm fetch cached or download pretrained weights
            net = timm.create_model(
                spec.name,
                pretrained=True,
                features_only=True,
                out_indices=spec.out_indices,
                in_chans=target_in
            )

        self.net = net
        self.feature_info = self.net.feature_info
        # Channels per selected stage
        self.channels: List[int] = list(self.feature_info.channels())
        # Strides (downsample factors) per selected stage
        # Some timm models expose .reduction(); else compute from feature_info
        try:
            self.strides: List[int] = list(self.feature_info.reduction())
        except Exception:
            # Fallback: timm usually has 'reduction' field in each item
            self.strides = [fi["reduction"] if "reduction" in fi else 2 ** (i + 1)
                            for i, fi in enumerate(self.feature_info)]

        # Optionally freeze
        if spec.freeze:
            for p in self.net.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.input_adapter(x)
        feats = self.net(x)  # list[tensor], one per out_index
        # Normalize to tuple for downstream code
        return tuple(feats)


def build_encoder(
    name: str = "resnet50",
    n_channels: int = 3,
    out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4),
    pretrained: bool = True,
    pretrained_path: Optional[str] = None,
    use_input_adapter: bool = False,
    freeze: bool = False,
) -> TimmEncoder:
    """
    Convenience factory so callers don’t touch EncoderSpec directly.
    """
    spec = EncoderSpec(
        name=name,
        n_channels=n_channels,
        out_indices=out_indices,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        use_input_adapter=use_input_adapter,
        freeze=freeze,
    )
    return TimmEncoder(spec)
