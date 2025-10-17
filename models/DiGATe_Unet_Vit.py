import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import timm

# Backbone
class SiameseBackbone(nn.Module):
    def __init__(self, name='vit_base_patch16_224', pretrained=True, pretrained_path=None):
        super().__init__()
        self.net = timm.create_model(name, pretrained=False)  # Always start without pretrained weights

        # If a local pretrained checkpoint path is provided, load weights manually
        if pretrained and pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            self.net.load_state_dict(state_dict)
        elif pretrained:
            # If pretrained=True but no path given, load from timm default
            self.net = timm.create_model(name, pretrained=True)

        self.net.reset_classifier(0)  # Remove classification head

        self.patch_size = self.net.patch_embed.patch_size
        self.embed_dim = self.net.embed_dim
        self.img_size = self.net.patch_embed.img_size

    def forward(self, x):
        B = x.shape[0]
        x = self.net.patch_embed(x)
        cls_token = self.net.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.net.pos_drop(x + self.net.pos_embed)
        for blk in self.net.blocks:
            x = blk(x)
        x = self.net.norm(x)
        x = x[:, 1:, :]  # Remove CLS token
        H = W = int(x.shape[1] ** 0.5)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H, W)

        f1 = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        f2 = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        f3 = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        f4 = x
        f5 = F.adaptive_avg_pool2d(x, output_size=(x.shape[2] // 2, x.shape[3] // 2))

        return f1, f2, f3, f4, f5

    @property
    def channels(self):
        C = self.embed_dim
        return [C, C, C, C, C]


# Normalized Helper
def LN2d(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(1, channels)

# Learnable Up-sampling
class SubPixelUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * 4, 1, bias=False)
        self.norm = LN2d(out_ch * 4)
        self.act  = nn.ReLU(inplace=True)
        self.ps   = nn.PixelShuffle(2)
    def forward(self, x):
        return self.ps(self.act(self.norm(self.conv(x))))

# Double Convolution
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c, mid_c=None):
        super().__init__()
        mid_c = out_c if mid_c is None else mid_c
        self.block = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, 1, bias=False), LN2d(mid_c), nn.ReLU(True),
            nn.Conv2d(mid_c, out_c, 3, 1, 1, bias=False), LN2d(out_c), nn.ReLU(True))
    def forward(self,x): return self.block(x)

# Attention Gate
class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        self.Wg  = nn.Sequential(nn.Conv2d(g_ch, inter, 1, bias=False), LN2d(inter))
        self.Wx  = nn.Sequential(nn.Conv2d(x_ch, inter, 1, bias=False), LN2d(inter))
        self.psi = nn.Sequential(nn.ReLU(True),
                                 nn.Conv2d(inter, 1, 1, bias=False),
                                 LN2d(1), nn.Sigmoid())
    def forward(self, g,x):
        α = self.psi(self.Wg(g) + self.Wx(x))
        return α * x

# Learnable Up Block
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

# Lightweight Transformer Cross-attention Up Block
class XAttn(nn.Module):
    def __init__(self, dim, heads=2, mlp=2.):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp)), nn.GELU(),
            nn.Linear(int(dim*mlp), dim))
    def forward(self,q,k,v):
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
        self.post   = nn.Sequential(nn.Conv2d(E, out_ch, 3, 1, 1, bias=False),
                                    LN2d(out_ch), nn.ReLU(True))
    def forward(self, d, s):
        d = self.up(d)
        dy, dx = s.size(2)-d.size(2), s.size(3)-d.size(3)
        d = F.pad(d,[dx//2, dx-dx//2, dy//2, dy-dy//2])
        B,_,H,W = d.shape
        q = self.proj_q(d).flatten(2).transpose(1,2)
        k = self.proj_k(s).flatten(2).transpose(1,2)
        v = self.proj_v(s).flatten(2).transpose(1,2)
        q = checkpoint(self.xattn, q, k, v)
        q = q.transpose(1,2).reshape(B,-1,H,W)
        return self.post(q)

# Adaptive Decoder
class AdaptiveDecoder(nn.Module):
    def __init__(self, ch):
        super().__init__()
        C1,C2,C3,C4,C5 = ch
        self.up1  = TransUp(C5,   C4,  C4//2)   # 1/32 → 1/16
        self.up2  = UpFlex(C4//2, C3,  C3//2)   # 1/16 → 1/8
        self.up3  = UpFlex(C3//2, C2,  C2//2)   # 1/8  → 1/4
        self.up4  = UpFlex(C2//2, C1,  C1//2)   # 1/4  → 1/2
        self.final_ch = C1//2
    def forward(self,f1,f2,f3,f4,f5):
        x1 = self.up1(f5,f4)  # 1/16
        x2 = self.up2(x1,f3)  # 1/8
        x3 = self.up3(x2,f2)  # 1/4
        x4 = self.up4(x3,f1)  # 1/2
        return x1,x2,x3,x4

# Out heads
class OutConv(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c,out_c,1)
    def forward(self,x): return self.conv(x)

# Gating Fusion
class GateFuse(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.g = nn.Sequential(
            nn.Conv2d(ch * 2, 1, 1),
            nn.Sigmoid()
        )
    # def forward(self, a, b):
    #     print("GateFuse input:", a.shape, b.shape)  # Debug print
    #     α = self.g(torch.cat([a, b], 1))
    #     out = α * a + (1 - α) * b
    #     reg = torch.mean(α * (1 - α))
    #     return out, reg

    def forward(self, a, b):
        α = self.g(torch.cat([a, b], 1))
        out = α * a + (1 - α) * b
        # Compute regularization as mean of α * (1 - α), higher when α is near 0.5
        reg = torch.mean(α * (1 - α))
        return out, reg

# tiny fuse helper
def elem_sum(a, b):                
    return 0.5 * (a + b) 

# Model
class DiGATe_Unet_Vit(nn.Module):
    def __init__(self, n_classes,
                 backbone="resnet50", model_path=None,
                 freeze_backbone=True):
        super().__init__()
        # Backbone
        self.backbone = SiameseBackbone(backbone, pretrained=True, pretrained_path=model_path)
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        C1,C2,C3,C4,C5 = self.backbone.channels          # channel list

        # Early-fusion gates
        self.efuse_c4 = GateFuse(C4)   # 1/16-res (layer-3 output)
        self.efuse_c3 = GateFuse(C3)   # 1/8-res  (layer-2 output)

        # Two independent decoders
        self.decoderA = AdaptiveDecoder(self.backbone.channels) 
        self.decoderB = AdaptiveDecoder(self.backbone.channels)  

        # Late fusion gates (¼-res and ½-res)
        final_ch        = self.decoderA.final_ch         # C1//2
        self.fuse_x3    = GateFuse(final_ch)             # 1/4-res
        self.fuse_x4    = GateFuse(final_ch)             # 1/2-res

        # Up-sample to full resolution + heads
        self.up_final   = SubPixelUp(final_ch, final_ch//2)
        self.head       = OutConv(final_ch//2, n_classes)
        # Deep-supervision heads (¼ and ½ resolution)
        self.aux2       = OutConv(final_ch, n_classes)
        self.aux3       = OutConv(final_ch,   n_classes)

    # ────────────────────────────────────────────────────────────────────────────
    def forward(self, x1, x2):
        # Siamese feature extraction
        a1,a2,a3,a4,a5 = self.backbone(x1) # Stream 1
        b1,b2,b3,b4,b5 = self.backbone(x2) # Stream 2

        # EARLY gated fusion
        f4, reg_c4 = self.efuse_c4(a4, b4)      # 1/16-res   (C4)
        f3, reg_c3 = self.efuse_c3(a3, b3)      # 1/8-res    (C3)
        # lower-level features are left untouched
        f2, f1, f5 = a2, a1, a5

        #  Stream 1 branch receives fused (a⊕b) features
        _,_,x3A,x4A = self.decoderA(f1,f2,f3,f4,f5)
        #  Stream 2 branch stays the same
        _,_,x3B,x4B = self.decoderB(b1,b2,b3,b4,b5)

        # LATE gated fusion (¼ & ½-res)
        x3, reg_x3 = self.fuse_x3(x3A, x3B)     # 1/4
        x4, reg_x4 = self.fuse_x4(x4A, x4B)     # 1/2

        # Final up-sampling & prediction
        main = self.head(self.up_final(x4))
        aux2 = F.interpolate(self.aux2(x3), size=main.shape[2:],
                             mode="bilinear", align_corners=True)
        aux3 = F.interpolate(self.aux3(x4), size=main.shape[2:],
                             mode="bilinear", align_corners=True)

        # Collect all regularisation terms
        reg = reg_c4, reg_c3, reg_x3, reg_x4 
        return main, aux2, aux3, reg        

# ───────────────────────── sanity check ────────────────────────────
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DiGATe_Unet_Vit(1, backbone="vit_base_patch16_224", model_path='/home/user1/.cache/torch/hub/checkpoints/vit_base_patch16_224.pth').to(dev)
    x1 = torch.randn(1,3,224,224, device=dev)
    x2 = torch.randn(1,3,224,224, device=dev)

    main, aux2, aux3, reg = model(x1,x2)

    print("output shapes:", main.shape, aux2.shape, aux3.shape)
    print("trainable params:",
          sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M")
