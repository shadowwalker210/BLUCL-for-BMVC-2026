import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms as T
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset

# DSConv (Depthwise Separable Convolution)
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()

        # Depthwise Convolution:
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=bias)
        # Pointwise Convolution:
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class BNReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.bn = nn.BatchNorm2d(features)

    def forward(self, x):
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class DSConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__(
            DSConv(in_channels, out_channels, kernel_size=kernel_size,
                   stride=stride, padding=padding, dilation=dilation, bias=bias),
            BNReLU(out_channels)
        )

# Residual Separable Convolution Block

class ResidualDSConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = DSConvBNReLU(in_channels=channels, out_channels=channels,
                                  kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.conv2 = DSConvBNReLU(in_channels=channels, out_channels=channels,
                                  kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
    def forward(self, x):
        x_conv = self.conv1(x)
        x_conv = self.conv2(x_conv)
        x_add = x + x_conv
        return F.relu(x_add)

# Upsampling Separable Convolutional Block

class UpConvBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dsconv = DSConvBNReLU(ch_in, ch_out)

    def forward(self,x):
        x = self.up(x)
        x = self.dsconv(x)
        return x

# Inverted Residual Block (IRB)

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion_ratio=1):
        super().__init__()
        self.stride = stride
        mid_channels = in_channels * expansion_ratio
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # Pointwise expansion
        if expansion_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SiLU(inplace=True)
            ])
        else:
            mid_channels = in_channels  # no expansion

        # Depthwise convolution
        layers.extend([
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1,
                      groups=mid_channels, bias=False),  # Depthwise
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True)
        ])

        # Pointwise projection
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual and x.shape == out.shape:
            return x + out
        else:
            return out

# Scale-aware Context Aggregation Block (SCAB)

class ScaleAwareConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Padding is set to match the dilation to keep the output size the same as input (for kernel size 3).
        self.conv_d1 = DSConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = DSConvBNReLU(in_channels, out_channels, kernel_size=3, padding=3, dilation=3)
        self.conv_d3 = DSConvBNReLU(in_channels, out_channels, kernel_size=3, padding=5, dilation=5)
        self.conv_d4 = DSConvBNReLU(in_channels, out_channels, kernel_size=3, padding=7, dilation=7)
        
        # Pointwise Convolution after Fusion
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        
        x_d1 = self.conv_d1(x)
        x_d2 = self.conv_d2(x)
        x_d3 = self.conv_d3(x)
        x_d4 = self.conv_d4(x)
        
        x_concat = torch.cat([x_d1,x_d2,x_d3, x_d4], dim=1)
        
        out = self.fuse(x_concat)
        
        return out

# Attention Mechanisms

# ----- Squeeze and Excitation Block -----
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output: (B, C, 1, 1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(b, c)          # (B, C)
        y = self.relu(self.fc1(y))                      # (B, C//r)
        y = self.sigmoid(self.fc2(y))                   # (B, C)
        y = y.view(b, c, 1, 1)                          # reshape to (B, C, 1, 1)
        return x * y.expand_as(x)                       # channel-wise scaling
        
# ----- Channel Attention Module (CAM) -----
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, ratio=8):
        """
        Translation of the TensorFlow Channel Attention Module.
        Based on: https://arxiv.org/abs/1807.06521
        """
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # output: (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # output: (B, C, 1, 1)

        # Shared MLP
        self.conv1 = nn.Conv2d(in_channels, in_channels // ratio, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // ratio, in_channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global Average Pooling path
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        # Global Max Pooling path
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        # Combine and apply sigmoid
        out = self.sigmoid(avg_out + max_out)
        return out
    
# ----- Spatial Attention Module (SAM) -----
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=3):
        """
        Spatial Attention Module (translated from Keras)
        Based on: https://arxiv.org/abs/1807.06521
        """
        super(SpatialAttentionModule, self).__init__()
        
        padding = (kernel_size - 1) // 2  # To preserve spatial dimensions
        
        self.conv1 = nn.Conv2d(2, 64, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=kernel_size, padding=padding, bias=False)
        self.conv4 = nn.Conv2d(16, 1, kernel_size=kernel_size, padding=padding, bias=False)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average and max
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Stack along channel dimension → shape: (B, 2, H, W)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # Pass through conv layers
        x = self.relu(self.conv1(x_cat))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        
        return x

# Boundary-guided Contextual Attention (BCA) Module
## Extracts boundary-sensitive structural information from feature maps using morphological operations.
class BoundaryExtractor(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        
        # Size of local neighborhood used for morphology
        self.kernel_size = kernel_size
        #  Padding to preserve spatial dimensions
        self.padding = kernel_size // 2

    def forward(self, x):
        dilation = F.max_pool2d(x, self.kernel_size, 1, self.padding) # Morphological Dilation
        erosion = -F.max_pool2d(-x, self.kernel_size, 1, self.padding) # Morphological Erosion
        boundary = dilation - erosion # Morphological Gradient
        return erosion, boundary
        
class BGAttention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.boundary_ext = BoundaryExtractor(kernel_size=kernel_size)
        self.channel_att = ChannelAttentionModule(channels)
        self.spatial_att = nn.Sequential(
                    nn.Conv2d(4, 64, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False),
                    nn.Sigmoid())

    def forward(self, x):
        
        # Channel Attention
        channel_att = self.channel_att(x)
        feat_refined = torch.mul(x,channel_att) # feat_sum * channel_att
        
        # Morphological Boundary Extraction
        interior_feat, boundary_feat = self.boundary_ext(feat_refined) # returns erosion, boundary
        # Spatial Descriptors
        s_avg = torch.mean(feat_refined, dim=1, keepdim=True) # spatial
        s_max, _ = torch.max(feat_refined, dim=1, keepdim=True) # spatial
        s_interior = torch.mean(interior_feat, dim=1, keepdim=True)
        s_boundary = torch.mean(boundary_feat, dim=1, keepdim=True)
    
        # -----------------------------
        # Boundary-Guided Spatial Attention
        # -----------------------------
        spatial_input = torch.cat([s_avg, s_max, s_interior, s_boundary], dim=1)
        spatial_att = self.spatial_att(spatial_input)
        
        return feat_refined * spatial_att


# A. Contrastive Learning Branch
## CL Encoder (Shared)
class CLEncoder(nn.Module):
    """
    Encoder with progressive downsampling.
    """
    def __init__(self, in_channels=3):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        # --- ENCODER PATH ---
        # Stage 1: Input 224x224
        self.dsconv1 = DSConvBNReLU(in_channels=in_channels, out_channels=32)
        self.bga1 = BGAttention(32)
        self.irb1 = InvertedResidualBlock(32,32)
        self.resdsc1 = ResidualDSConvBlock(32)
        
        # Stage 2: Input 112x112
        self.dsconv2 = DSConvBNReLU(32, 64)
        self.bga2 = BGAttention(64)
        self.irb2 = InvertedResidualBlock(64,64)
        self.resdsc2 = ResidualDSConvBlock(64)
        
        # Stage 3: Input 56x56
        self.dsconv3 = DSConvBNReLU(64, 128)
        self.bga3 = BGAttention(128)
        self.irb3 = InvertedResidualBlock(128,128)
        self.resdsc3 = ResidualDSConvBlock(128)
        
        # Stage 4: Input 28x28
        self.dsconv4 = DSConvBNReLU(128, 256)
        self.bga4 = BGAttention(256)
        self.irb4 = InvertedResidualBlock(256,256)
        self.resdsc4 = ResidualDSConvBlock(256)
        
        # Stage 5: Input 14x14
        self.dsconv5 = DSConvBNReLU(256, 512)
        self.resdsc5 = ResidualDSConvBlock(512)
        

    def forward(self, x):
        # ---------------------
        # 224 -> 112
        e1 = self.dsconv1(x)
        e1 = self.bga1(e1)
        e1 = self.irb1(e1)
        e1 = self.resdsc1(e1)
        p1 = self.pool(e1)

        # ---------------------
        # 112 -> 56
        e2 = self.dsconv2(p1)
        e2 = self.bga2(e2)
        e2 = self.irb2(e2)
        e2 = self.resdsc2(e2)
        p2 = self.pool(e2)

        # ---------------------
        # 56 -> 28
        e3 = self.dsconv3(p2)
        e3 = self.bga3(e3)
        e3 = self.irb3(e3)
        e3 = self.resdsc3(e3)
        p3 = self.pool(e3)
        
        # ---------------------
        # 28
        e4 = self.dsconv4(p3)
        e4 = self.bga4(e4)
        e4 = self.irb4(e4)
        e4 = self.resdsc4(e4)
        p4 = self.pool(e4)
        
        # ---------------------
        # 14
        e5 = self.dsconv5(p4)
        e5 = self.resdsc5(e5)

        return e1, e2, e3, e4, e5


# Projection Head
class ConvProjectionHead(nn.Module):
    """
    Deep Projection Head funneling from 512 down to a n-channel embedding.
    Uses 1x1 convolutions and maintains the spatial topology.
    """
    def __init__(self, in_dim=512, out_dim=256):
        super().__init__()
        
        # Layer 1: -> 256
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_dim, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.bga = BGAttention(256)
        # Layer 2: -> 32
        # No ReLU on the final embedding space to allow negative vector values
        self.layer2 = nn.Conv2d(256, out_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bga(x)
        x = self.layer2(x)
        return x

# CL Encoder-Projection Head Model
class CLModel(nn.Module):

    def __init__(self, in_channels=3, proj_dim=128):
        super().__init__()

        self.encoder = CLEncoder(in_channels=in_channels)
        self.projection_head = ConvProjectionHead(in_dim=512, out_dim=proj_dim)

    def forward(self, x):

        # Encoder
        e1, e2, e3, e4, e5 = self.encoder(x)

        # Dense embeddings
        z = self.projection_head(e5)

        return z

# B. Supervised Segmentation Branch
## Segmentation Decoder

class SegDecoder(nn.Module):
    """
    Supervised Learning Segmentation Decoder
    """

    def __init__(self):

        super().__init__()
        
        # Multi-Scale Features Fused with Skip Connections
        self.skip4 = ScaleAwareConvBlock(256,256)
        self.skip3 = ScaleAwareConvBlock(128,128)
        self.skip2 = ScaleAwareConvBlock(64,64)
        self.skip1 = ScaleAwareConvBlock(32,32)
        
        # 14 -> 28
        self.upconv4 = UpConvBlock(512, 256)
        self.dsconv5 = DSConvBNReLU(512, 256)
        self.bga5 = BGAttention(256)
        self.irb5 = InvertedResidualBlock(256, 256, expansion_ratio=2)
        self.resdsc5 = ResidualDSConvBlock(256)
        
        # 28 -> 56
        self.upconv3 = UpConvBlock(256, 128)
        self.dsconv6 = DSConvBNReLU(256, 128)
        self.bga6 = BGAttention(128)
        self.irb6 = InvertedResidualBlock(128, 128, expansion_ratio=2)
        self.resdsc6 = ResidualDSConvBlock(128)

        # 56 -> 112
        self.upconv2 = UpConvBlock(128, 64)
        self.dsconv7 = DSConvBNReLU(128, 64)
        self.bga7 = BGAttention(64)
        self.irb7 = InvertedResidualBlock(64,64, expansion_ratio=2)
        self.resdsc7 = ResidualDSConvBlock(64)

        # 112 -> 224
        self.upconv1 = UpConvBlock(64, 32)
        self.dsconv8 = DSConvBNReLU(64, 32)
        self.bga8 = BGAttention(32)
        self.irb8 = InvertedResidualBlock(32,32, expansion_ratio=2)
        self.resdsc8 = ResidualDSConvBlock(32)
        
        # ------- Output layer ------- #
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        
    
    def forward(self, e1, e2, e3, e4, e5):
        
        # Multi-Scale Features Integrated Skip Connections originating from Shared CL-Encoder Features
        e4 = self.skip4(e4)  #256
        e3 = self.skip3(e3)  #128
        e2 = self.skip2(e2)  #64
        e1 = self.skip1(e1)  #32
        

        # 14 -> 28
        d4 = self.upconv4(e5)                    # -> 256
        d4 = self.dsconv5(torch.cat((e4, d4), dim=1))  # 256 + 256 = 512 -> DSConvBNReLU(512,256)
        d4 = self.bga5(d4)
        d4 = self.irb5(d4)
        d4 = self.resdsc5(d4)
        
        # 28 -> 56
        d3 = self.upconv3(d4)                    # -> 128
        d3 = self.dsconv6(torch.cat((e3, d3), dim=1))  # 128 + 128 = 256 -> DSConvBNReLU(256,128)
        d3 = self.bga6(d3)
        d3 = self.irb6(d3)
        d3 = self.resdsc6(d3)

        # 56 -> 112
        d2 = self.upconv2(d3)                    # -> 64
        d2 = self.dsconv7(torch.cat((e2, d2), dim=1))  # 64 + 64 = 128 -> DSConvBNReLU(128,64)
        d2 = self.bga7(d2)
        d2 = self.irb7(d2)
        d2 = self.resdsc7(d2)

        # 112 -> 224
        d1 = self.upconv1(d2)                    # -> 32
        d1 = self.dsconv8(torch.cat((e1, d1), dim=1))  # 32 + 32 = 64 -> DSConvBNReLU(64,32)
        d1 = self.bga8(d1)
        d1 = self.irb8(d1)
        d1 = self.resdsc8(d1)
        
        out = torch.sigmoid(self.final_conv(d1))  # -> 1

        return out

# Supervised Segmentation Encoder-Decoder Network
class SegUNET(nn.Module):

    def __init__(self, in_channels=3, proj_dim=128):
        super().__init__()

        self.encoder = CLEncoder(in_channels=in_channels)
        self.decoder = SegDecoder()

    def forward(self, x):

        # Encoder
        e1, e2, e3, e4, e5 = self.encoder(x)
        
        # Decoder
        d = self.decoder(e1, e2, e3, e4, e5)

        return d

# BLUCL: Unified CL Segmentation Model
class BLUCL(nn.Module):
    def __init__(self, encoder, projection_head, prototype_layer, seg_decoder):
        super().__init__()

        # Shared Encoder
        self.encoder = encoder #CLEncoder(in_channels=in_channels)

        # Contrastive Learning Branch
        self.projection_head = projection_head 

        # Segmentation Branch
        self.seg_decoder = seg_decoder #SegDecoder()
        
        # Prototype Layer
        self.prototype_layer = prototype_layer
        
    def forward(self, x, mode="seg"):

        # Shared Encoder
        e1, e2, e3, e4, e5 = self.encoder(x)

        # =========================
        # CL Branch (only)
        # =========================
        if mode == "cl":

            # CL Decoder + Projection Head
            cl_features =  e5
            z = self.projection_head(cl_features)
            
            # Prototype Assignment
            B, D, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).reshape(-1, D) # Flatten embeddings
            prototype_logits = self.prototype_layer(z_flat)

            return z, prototype_logits

        # ============================
        # Segmentation Branch (only)
        # ============================
        seg_out = self.seg_decoder(e1, e2, e3, e4, e5)

        return seg_out


# Prototype-learning modules are omitted from the current public release.
