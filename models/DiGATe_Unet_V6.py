import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .backbones import build_encoder

def LN2d(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(1, c)

class SubPixelUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 1, bias=False)
        self.norm = LN2d(out_ch * 4)
        self.act  = nn.ReLU(inplace=True)
        self.ps   = nn.PixelShuffle(2)
    def forward(self, x):
        return self.ps(self.act(self.norm(self.conv(x))))

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None):
        super().__init__()
        mid_c = out_c if mid_c is None else mid_c
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=False), LN2d(mid_c), nn.ReLU(True),
            nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False), LN2d(out_c), nn.ReLU(True))
    def forward(self,x): return self.block(x)

class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        inter = max(1, inter)
        self.Wg  = nn.Sequential(nn.Conv2d(g_ch, inter, 1, bias=False), LN2d(inter))
        self.Wx  = nn.Sequential(nn.Conv2d(x_ch, inter, 1, bias=False), LN2d(inter))
        # safer 1-ch norm to avoid underflow
        self.psi = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(inter, 1, 1, bias=False),
            nn.BatchNorm2d(1, affine=False),
            nn.Sigmoid()
        )
    def forward(self, g, x):
        a = self.psi(self.Wg(g) + self.Wx(x))
        return a * x

class UpFlex(nn.Module):
    def __init__(self, dec_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = SubPixelUp(dec_ch, dec_ch//2)
        self.attn = AttentionGate(dec_ch//2, skip_ch, inter=min(dec_ch//2, skip_ch)//4)
        self.conv = DoubleConv(dec_ch//2 + skip_ch, out_ch)
    def forward(self, d, s):
        d = self.up(d)
        dy, dx = s.size(2) - d.size(2), s.size(3) - d.size(3)
        d = F.pad(d, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        s = self.attn(d, s)
        return self.conv(torch.cat([s, d], 1))

class XAttn(nn.Module):
    def __init__(self, dim, heads=2, mlp=2.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp)), nn.GELU(),
            nn.Linear(int(dim*mlp), dim))
    def forward(self, q, k, v):
        h,_ = self.attn(self.ln1(q), self.ln1(k), self.ln1(v))
        q = q + h
        return q + self.mlp(self.ln2(q))

class TransUp(nn.Module):
    def __init__(self, dec_ch, skip_ch, out_ch):
        super().__init__()
        self.up = SubPixelUp(dec_ch, dec_ch//2)
        E = out_ch
        self.proj_q = nn.Conv2d(dec_ch//2, E, 1, bias=False)
        self.proj_k = nn.Conv2d(skip_ch,  E, 1, bias=False)
        self.proj_v = nn.Conv2d(skip_ch,  E, 1, bias=False)
        self.xattn  = XAttn(E)
        self.post   = nn.Sequential(
            nn.Conv2d(E, out_ch, 3, 1, 1, bias=False),
            LN2d(out_ch), nn.ReLU(True)
        )
    def forward(self, d, s):
        d = self.up(d)
        dy, dx = s.size(2)-d.size(2), s.size(3)-d.size(3)
        d = F.pad(d, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        B,_,H,W = d.shape
        q = self.proj_q(d).flatten(2).transpose(1,2)
        k = self.proj_k(s).flatten(2).transpose(1,2)
        v = self.proj_v(s).flatten(2).transpose(1,2)
        q = checkpoint(self.xattn, q, k, v) if q.requires_grad else self.xattn(q, k, v)
        q = q.transpose(1,2).reshape(B,-1,H,W)
        return self.post(q)

class OutConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1)
    def forward(self, x): return self.conv(x)

class AdaptiveDecoder(nn.Module):
    def __init__(self, ch_list):
        super().__init__()
        C1,C2,C3,C4,C5 = ch_list
        self.up1  = TransUp(C5,   C4,  C4//2)   # 1/32 → 1/16
        self.up2  = UpFlex(C4//2, C3,  C3//2)   # 1/16 → 1/8
        self.up3  = UpFlex(C3//2, C2,  C2//2)   # 1/8  → 1/4
        self.up4  = UpFlex(C2//2, C1,  C1//2)   # 1/4  → 1/2

        self.ch_x1 = C4 // 2   # 1/16
        self.ch_x2 = C3 // 2   # 1/8
        self.ch_x3 = C2 // 2   # 1/4
        self.ch_x4 = C1 // 2   # 1/2
        self.final_ch = self.ch_x4

    def forward(self, f1,f2,f3,f4,f5):
        x1 = self.up1(f5,f4)  # 1/16
        x2 = self.up2(x1,f3)  # 1/8
        x3 = self.up3(x2,f2)  # 1/4
        x4 = self.up4(x3,f1)  # 1/2
        return x1,x2,x3,x4

class GateFuse(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.g = nn.Sequential(nn.Conv2d(ch * 2, 1, 1), nn.Sigmoid())
    def forward(self, a, b):
        alpha = self.g(torch.cat([a, b], dim=1))
        out = alpha * a + (1 - alpha) * b
        reg = torch.mean(alpha * (1 - alpha))  # encourage confident gates
        return out, reg


# --------------------- model ---------------------
class DiGATe_Unet_V6(nn.Module):
    """
    Args:
        n_classes: output channels
        backbone: timm model name (e.g., 'resnet101', 'tf_efficientnet_b0', ...)
        n_channels: input channels for both streams (if share_backbone=True)
        n_channels_b: optional different channels for stream-B (requires share_backbone=False)
        pretrained: use timm pretrained weights (if pretrained_path is None)
        pretrained_path: local checkpoint path (offline or custom)
        use_input_adapter: if True, map N→3 before backbone to preserve ImageNet conv1
        freeze_backbone: freeze encoder params (handy for Stage-1)
        share_backbone: if True, re-use one encoder for both streams (weight sharing)
        out_indices: backbone stages to extract (default 5 stages)
    """
    def __init__(self,
                 n_classes: int,
                 backbone: str = "resnet50",
                 n_channels: int = 3,
                 n_channels_b: int | None = None,
                 pretrained: bool = True,
                 pretrained_path: str | None = None,
                 use_input_adapter: bool = False,
                 freeze_backbone: bool = False,
                 share_backbone: bool = True,
                 out_indices=(0,1,2,3,4)):
        super().__init__()

        if share_backbone and (n_channels_b is not None) and (n_channels_b != n_channels):
            raise ValueError("When share_backbone=True, n_channels_b must equal n_channels (or be None).")
        n_channels_b = n_channels if n_channels_b is None else n_channels_b

        # Build encoders via factory
        if share_backbone:
            self.encoder = build_encoder(
                name=backbone,
                n_channels=n_channels,
                out_indices=out_indices,
                pretrained=pretrained if pretrained_path is None else False,
                pretrained_path=pretrained_path,
                use_input_adapter=use_input_adapter,
                freeze=freeze_backbone
            )
            ch_list = self.encoder.channels
        else:
            self.encoderA = build_encoder(
                name=backbone,
                n_channels=n_channels,
                out_indices=out_indices,
                pretrained=pretrained if pretrained_path is None else False,
                pretrained_path=pretrained_path,
                use_input_adapter=use_input_adapter,
                freeze=freeze_backbone
            )
            self.encoderB = build_encoder(
                name=backbone,
                n_channels=n_channels_b,
                out_indices=out_indices,
                pretrained=pretrained if pretrained_path is None else False,
                pretrained_path=pretrained_path,
                use_input_adapter=use_input_adapter,
                freeze=freeze_backbone
            )
            # sanity: require equal stage widths for the decoders/gates
            if tuple(self.encoderA.channels) != tuple(self.encoderB.channels):
                raise ValueError(f"EncoderA/B channel lists differ: {self.encoderA.channels} vs {self.encoderB.channels}")
            ch_list = self.encoderA.channels

        # Early-fusion gates at encoder C4 and C3 scales
        C1,C2,C3,C4,C5 = ch_list
        self.efuse_c4 = GateFuse(C4)   # 1/16
        self.efuse_c3 = GateFuse(C3)   # 1/8

        # Two decoders (A: fused path; B: raw path)
        self.decoderA = AdaptiveDecoder(ch_list)
        self.decoderB = AdaptiveDecoder(ch_list)

        # Late fusion gates at 1/4 and 1/2 scales
        self.fuse_x3  = GateFuse(self.decoderA.ch_x3)   # robust across backbones
        self.fuse_x4  = GateFuse(self.decoderA.ch_x4)

        # Heads
        final_ch      = self.decoderA.final_ch
        self.up_final = SubPixelUp(final_ch, final_ch//2)
        self.head     = OutConv(final_ch//2, n_classes)
        self.aux2     = OutConv(self.decoderA.ch_x3, n_classes)  # 1/4
        self.aux3     = OutConv(self.decoderA.ch_x4, n_classes)  # 1/2

        # flags for forward
        self.share_backbone = share_backbone

    def _encode(self, x1, x2):
        if self.share_backbone:
            a1,a2,a3,a4,a5 = self.encoder(x1)
            b1,b2,b3,b4,b5 = self.encoder(x2)
        else:
            a1,a2,a3,a4,a5 = self.encoderA(x1)
            b1,b2,b3,b4,b5 = self.encoderB(x2)
        return (a1,a2,a3,a4,a5), (b1,b2,b3,b4,b5)

    def forward(self, x1, x2):
        # Siamese/dual feature extraction
        (a1,a2,a3,a4,a5), (b1,b2,b3,b4,b5) = self._encode(x1, x2)

        # Early gated fusion at C4 (1/16) and C3 (1/8)
        f4, reg_c4 = self.efuse_c4(a4, b4)
        f3, reg_c3 = self.efuse_c3(a3, b3)

        # Lower/upper levels pass-through for stream A
        f2, f1, f5 = a2, a1, a5

        # Decode: A (uses fused C3/C4), B (raw B-stream)
        _,_,x3A,x4A = self.decoderA(f1,f2,f3,f4,f5)
        _,_,x3B,x4B = self.decoderB(b1,b2,b3,b4,b5)

        # Late fusion at 1/4 & 1/2 scales
        x3, reg_x3 = self.fuse_x3(x3A, x3B)
        x4, reg_x4 = self.fuse_x4(x4A, x4B)

        # Final upsample & heads
        main = self.head(self.up_final(x4))
        aux2 = F.interpolate(self.aux2(x3), size=main.shape[2:], mode="bilinear", align_corners=True)
        aux3 = F.interpolate(self.aux3(x4), size=main.shape[2:], mode="bilinear", align_corners=True)

        reg = (reg_c4, reg_c3, reg_x3, reg_x4)
        return main, aux2, aux3, reg


# --------------------- sanity check ---------------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # Example: shared ResNet101, 3-ch inputs
    m = DiGATe_Unet_V6(
        n_classes=1,
        backbone="mobilenetv3_small_100", #tf_efficientnet_b4
        n_channels=3,
        pretrained=False,          # set True if downloads are allowed
        pretrained_path=None,      # or local .pth file
        use_input_adapter=False,
        freeze_backbone=True,
        share_backbone=False
    ).to(dev)

    x1 = torch.randn(1, 3, 256, 256, device=dev)
    x2 = torch.randn(1, 3, 256, 256, device=dev)
    out = m(x1, x2)
    main, aux2, aux3, reg = out
    print("output shapes:", main.shape, aux2.shape, aux3.shape)
    print("trainable params:",
          sum(p.numel() for p in m.parameters() if p.requires_grad)/1e6, "M")
