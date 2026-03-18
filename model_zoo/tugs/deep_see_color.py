import torch
from sympy.physics.optics import medium
from torch import nn
import math

class DeattenuateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5

    def forward(self, direct, J):
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        channel_spatial = torch.std(J, dim=[2, 3])
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        return intensity_loss + spatial_variation_loss

class SeaThruNet(nn.Module):
    def __init__(self, cost_ratio=1000., depth_clip=5.):
        super().__init__()
        self.backscatter_conv = nn.Conv2d(1, 3, 1, padding=0, bias=False)
        # nn.init.uniform_(self.backscatter_conv.weight, 0, 5)
        nn.init.constant_(self.backscatter_conv.weight, 0.9)
        self.B_inf = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]), requires_grad=True)

        self.residual_conv = nn.Conv2d(1, 3, 1, padding=0, bias=False)
        nn.init.uniform_(self.residual_conv.weight, 0, 1)
        self.J_prime = nn.Parameter(torch.tensor([0., 0., 0.]), requires_grad=True)
        

        self.l1 = nn.L1Loss(reduction='mean')
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2, reduction='mean')
        self.mse = nn.MSELoss()
        self.deattenuate = DeattenuateLoss()
        self.relu = nn.ReLU()

        self.cost_ratio = cost_ratio
        self.depth_clip = depth_clip

    def attenuation(self, J, medium, depth1):
        f = torch.exp(
            -torch.clamp(
                medium * depth1, 0, float(torch.log(torch.tensor([3.])))
            )
        )
        return f, J * f

    def backscatter(self, medium, depth2):
        # medium (1 H W C), depth (H W 1)
        dep = depth2.unsqueeze(0).permute(0, 3, 1, 2)
        beta_b_conv = self.relu(self.backscatter_conv(dep)).permute(0, 2, 3, 1)
        beta_d_conv = self.relu(self.residual_conv(dep)).permute(0, 2, 3, 1)
        backscatter = self.B_inf * (1 - torch.exp(-beta_b_conv)) + self.J_prime * torch.exp(-beta_d_conv)
        return backscatter

    def forward(self, J, medium, depth1, depth2):
        f, rgb_direct = self.attenuation(J, medium, depth2)
        rgb_backscatter = self.backscatter(medium, depth2.detach())
        return f, rgb_direct, rgb_backscatter

    def loss(self, J, backscatter, gt, depth1, depth2, alpha, step):
        # backscatter (H W 3), depth (H W 1), gt (H W 3)
        backscatter_masked = backscatter * (depth1 > 0.).repeat(1, 1, 3)
        direct = gt - backscatter_masked
        # mask = self.relu(1 - torch.exp(self.depth_clip - depth))
        pos = self.l1(self.relu(direct), torch.zeros_like(direct))
        neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))
        all_loss = self.cost_ratio * neg + pos
        # if step >= 1000:
        #     all_loss += self.deattenuate(direct=direct.unsqueeze(0).permute(0, 3, 1, 2), J=J.unsqueeze(0).permute(0, 3, 1, 2))
        return all_loss


# class SeaThruNet(nn.Module):
#     def __init__(self, cost_ratio=1000., depth_clip=5.):
#         super().__init__()
#         self.backscatter_conv = nn.Conv2d(3, 3, 3, padding=1, bias=False)
#         nn.init.uniform_(self.backscatter_conv.weight, 0, 5)
#         self.B_inf = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3]), requires_grad=True)
#         self.relu = nn.ReLU()

#         self.l1 = nn.L1Loss(reduction='mean')
#         self.smooth_l1 = nn.SmoothL1Loss(beta=0.2, reduction='mean')
#         self.mse = nn.MSELoss()
#         self.relu = nn.ReLU()

#         self.cost_ratio = cost_ratio
#         self.depth_clip = depth_clip

#     def attenuation(self, J, medium, depth):
#         f = torch.exp(-torch.clamp(medium * depth, 0, float(torch.log(torch.tensor([3.])))))
#         return J * f

#     def backscatter(self, medium, depth):
#         # medium (1 H W C), depth (H W 1)
#         beta_b_conv = self.relu(self.backscatter_conv(medium.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
#         backscatter = self.B_inf * (1 - torch.exp(-beta_b_conv * depth))
#         return backscatter

#     def forward(self, J, medium, depth):
#         rgb_direct = self.attenuation(J, medium, depth)
#         rgb_backscatter = self.backscatter(medium, depth)
#         return rgb_direct, rgb_backscatter

#     def loss(self, J, backscatter, gt, depth, alpht, step):
#         # backscatter (H W 3), depth (H W 1), gt (H W 3)
#         backscatter_masked = backscatter * (depth > 0.).repeat(1, 1, 3)
#         direct = gt - backscatter_masked
#         mask = self.relu(1 - torch.exp(self.depth_clip - depth))
#         pos = self.l1(self.relu(direct) * mask, torch.zeros_like(direct))
#         neg = self.smooth_l1(self.relu(-direct), torch.zeros_like(direct))

#         return self.cost_ratio * neg + pos
