import torch
from torch import nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59d273b4a62BA70a0d63453c94
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# The original unet class is preserved for reference if needed.
class unet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

#  MODIFIED TWO-STREAM MODEL                          
class TwoStreamUNet(nn.Module):
    """
    A modified U-Net architecture that accepts two separate input streams (x1, x2).
    Each stream is processed by a full U-Net-like encoder-decoder path.
    The final feature maps from both streams are concatenated and passed through
    a final output convolution layer.
    """
    def __init__(self, n_classes, n_channels=3, bilinear=True):
        super(TwoStreamUNet, self).__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Stream 1
        self.inc1 = DoubleConv(n_channels, 64)
        self.down1_1 = Down(64, 128)
        self.down2_1 = Down(128, 256)
        self.down3_1 = Down(256, 512)
        self.down4_1 = Down(512, 1024 // factor)
        self.up1_1 = Up(1024, 512 // factor, bilinear)
        self.up2_1 = Up(512, 256 // factor, bilinear)
        self.up3_1 = Up(256, 128 // factor, bilinear)
        self.up4_1 = Up(128, 64, bilinear)

        # Stream 2
        self.inc2 = DoubleConv(n_channels, 64)
        self.down1_2 = Down(64, 128)
        self.down2_2 = Down(128, 256)
        self.down3_2 = Down(256, 512)
        self.down4_2 = Down(512, 1024 // factor)
        self.up1_2 = Up(1024, 512 // factor, bilinear)
        self.up2_2 = Up(512, 256 // factor, bilinear)
        self.up3_2 = Up(256, 128 // factor, bilinear)
        self.up4_2 = Up(128, 64, bilinear)

        # Final Output Layer
        # It takes the concatenated features from both streams (64 + 64 channels)
        self.outc = OutConv(128, n_classes)

    def forward(self, x1, x2):
        # Stream 1 
        s1_d1 = self.inc1(x1)
        s1_d2 = self.down1_1(s1_d1)
        s1_d3 = self.down2_1(s1_d2)
        s1_d4 = self.down3_1(s1_d3)
        s1_d5 = self.down4_1(s1_d4)
        s1_u1 = self.up1_1(s1_d5, s1_d4)
        s1_u2 = self.up2_1(s1_u1, s1_d3)
        s1_u3 = self.up3_1(s1_u2, s1_d2)
        s1_features = self.up4_1(s1_u3, s1_d1) # Final features for stream 1

        # Stream 2
        s2_d1 = self.inc2(x2)
        s2_d2 = self.down1_2(s2_d1)
        s2_d3 = self.down2_2(s2_d2)
        s2_d4 = self.down3_2(s2_d3)
        s2_d5 = self.down4_2(s2_d4)
        s2_u1 = self.up1_2(s2_d5, s2_d4)
        s2_u2 = self.up2_2(s2_u1, s2_d3)
        s2_u3 = self.up3_2(s2_u2, s2_d2)
        s2_features = self.up4_2(s2_u3, s2_d1) # Final features for stream 2

        # Concatenate Features and Final Output
        combined_features = torch.cat([s1_features, s2_features], dim=1)
        logits = self.outc(combined_features)
        return logits

if __name__ == '__main__':
    n_output_classes = 1
    model = TwoStreamUNet(n_classes=n_output_classes, n_channels=3)
    input_stream1 = torch.randn(32, 3, 256, 256)
    input_stream2 = torch.randn(32, 3, 256, 256)
    output = model(input_stream1, input_stream2)

    print(f"Input shape (x1): {input_stream1.shape}")
    print(f"Input shape (x2): {input_stream2.shape}")
    print(f"Output shape: {output.shape}")
    print("trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M")