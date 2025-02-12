import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.deformable import DeformableConv2d



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)

        return self.sigmoid(x)


class DeformableSpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.deformable_conv = DeformableConv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.deformable_conv(attention)
        attention = self.sigmoid(attention)

        x = x * attention

        return x


class EnhancedSpatialAttention(nn.Module):
    """original code: https://github.com/njulj/RFANet/blob/master/ESA.py"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        out_channels = in_channels // reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.stride_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=3)
        self.relu = nn.ReLU(inplace=True)

        self.conv_groups = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c0 = self.conv1(x)

        c1 = self.stride_conv(c0)
        c1 = self.max_pool(c1)

        c1 = self.conv_groups(c1)
        c1 = F.interpolate(c1, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        c2 = self.conv2(self.branch_conv(c0)+c1)
        c2 = self.sigmoid(c2)

        return x * c2


class DeformableEnhancedSpatialAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        out_channels = in_channels // reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.stride_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=3)

        self.deformable_conv_groups = nn.Sequential(
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c0 = self.conv1(x)

        c1 = self.stride_conv(c0)
        c1 = self.max_pool(c1)

        c1 = self.deformable_conv_groups(c1)
        c1 = F.interpolate(c1, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        c2 = self.conv2(self.branch_conv(c0)+c1)
        c2 = self.sigmoid(c2)

        return x * c2


class FrequencyDeformableAttention(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        out_channels = in_channels // reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.stride_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=3)

        self.deformable_conv_groups = nn.Sequential(
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c0 = self.conv1(x)

        c1 = self.stride_conv(c0)
        c1 = self.max_pool(c1)

        #-----------------------------------------
        fft_map = torch.fft.fft2(c1, dim=(-2, -1))

        c1_magnitude_map = torch.abs(fft_map)
        c1_phase_map = torch.angle(fft_map)

        c1_attn = self.deformable_conv_groups(c1_magnitude_map)
        c1_real_part = c1_attn * torch.cos(c1_phase_map)
        c1_imag_part = c1_attn * torch.sin(c1_phase_map)
        c1_fft_map = torch.complex(c1_real_part, c1_imag_part)

        reconstructed_c1 = torch.real(torch.fft.ifft2(c1_fft_map, dim=(-2, -1)))
        #-----------------------------------------

        c1 = F.interpolate(reconstructed_c1, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        c2 = self.conv2(self.branch_conv(c0)+c1)
        c2 = self.sigmoid(c2)

        return x * c2


# Frequency Deformable Enhanced Spatial Attention applied by Phase and Amplitude information
class FrequencyDeformableAttentionWithPhase(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        out_channels = in_channels // reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.stride_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=3)

        self.deformable_conv_groups_amp = nn.Sequential(
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.deformable_conv_groups_phase = nn.Sequential(
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c0 = self.conv1(x)

        c1 = self.stride_conv(c0)
        c1 = self.max_pool(c1)

        #-----------------------------------------
        fft_map = torch.fft.fft2(c1, dim=(-2, -1))

        c1_magnitude_map = torch.abs(fft_map)
        c1_phase_map = torch.angle(fft_map)

        c1_attn_amp = self.deformable_conv_groups_amp(c1_magnitude_map)
        c1_attn_pahse = self.deformable_conv_groups_phase(c1_phase_map)

        c1_real_part = c1_attn_amp * torch.cos(c1_attn_pahse)
        c1_imag_part = c1_attn_amp * torch.sin(c1_attn_pahse)
        c1_fft_map = torch.complex(c1_real_part, c1_imag_part)

        reconstructed_c1 = torch.real(torch.fft.ifft2(c1_fft_map, dim=(-2, -1)))
        #-----------------------------------------

        c1 = F.interpolate(reconstructed_c1, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        c2 = self.conv2(self.branch_conv(c0)+c1)
        c2 = self.sigmoid(c2)

        return x * c2


class FrequencyDeformableAttentionBandwidth(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        out_channels = in_channels // reduction
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.stride_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=3)

        self.high_band_freq_deformable_conv_groups = nn.Sequential(
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.conv2 = nn.Conv2d(out_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def apply_frequency_filter(self, fft_map, filter_type):
        # Get dimensions
        _, _, h, w = fft_map.shape
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        center_y, center_x = h // 2, w // 2

        # Calculate distance from the center
        distance = torch.sqrt((y - center_y) ** 2 + (x - center_x) ** 2).to(fft_map.device)

        if filter_type == 'low':
            # Low-pass filter mask
            mask = (distance <= (h // 4)).float()
        elif filter_type == 'mid':
            # Band-pass filter mask
            mask = ((distance > (h // 4)) & (distance < (h // 2))).float()
        elif filter_type == 'high':
            # High-pass filter mask
            mask = (distance >= (h // 2)).float()
        elif filter_type == 'band_high':
            # Combined band and high-pass filter mask
            mask = (distance > (h // 16)).float()
        else:
            raise ValueError("Unknown filter type")

        # Apply mask to fft_map
        return fft_map * mask

    def forward(self, x):
        c0 = self.conv1(x)

        c1 = self.stride_conv(c0)
        c1 = self.max_pool(c1)

        #-----------------------------------------
        fft_map = torch.fft.fft2(c1, dim=(-2, -1))
        fft_map = torch.fft.fftshift(fft_map, dim=(-2, -1))

        c1_magnitude_map = torch.abs(fft_map)
        c1_phase_map = torch.angle(fft_map)

        high_band_freq = self.apply_frequency_filter(c1_magnitude_map, 'band_high')

        high_band_freq_attn = self.high_band_freq_deformable_conv_groups(high_band_freq)

        # Combine the frequency components
        # combined_attn = high_freq_attn + band_freq_attn

        c1_real_part = high_band_freq_attn * torch.cos(c1_phase_map)
        c1_imag_part = high_band_freq_attn * torch.sin(c1_phase_map)
        c1_fft_map = torch.complex(c1_real_part, c1_imag_part)

        c1_fft_map = torch.fft.ifftshift(c1_fft_map, dim=(-2, -1))
        reconstructed_c1 = torch.real(torch.fft.ifft2(c1_fft_map, dim=(-2, -1)))
        #-----------------------------------------

        c1 = F.interpolate(reconstructed_c1, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)

        c2 = self.conv2(self.branch_conv(c0)+c1)
        c2 = self.sigmoid(c2)

        return x * c2

