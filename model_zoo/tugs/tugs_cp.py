# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from tensorly import cp_to_tensor
from tensorly.decomposition import CPPower
from tensorly.metrics import MSE
from torch import nn

try:
    import tensorly
except ImportError:
    print("Please install tensorly")

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from pytorch_msssim import SSIM
from torch.nn import Parameter

from nerfstudio.cameras.camera_optimizers import CameraOptimizer, CameraOptimizerConfig
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers
from nerfstudio.model_components.lib_bilagrid import BilateralGrid, color_correct, slice, total_variation_loss
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.math import k_nearest_sklearn, random_quat_tensor
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.spherical_harmonics import RGB2SH, SH2RGB, num_sh_bases

from .tugs_cp_strategy import CPStrategy
from .deep_see_color import SeaThruNet


def resize_image(image: torch.Tensor, d: int):
    """
    Downscale images using the same 'area' method in opencv

    :param image shape [H, W, C]
    :param d downscale factor (must be 2, 4, 8, etc.)

    return downscaled image in shape [H//d, W//d, C]
    """
    import torch.nn.functional as tf

    image = image.to(torch.float32)
    weight = (1.0 / (d * d)) * torch.ones((1, 1, d, d), dtype=torch.float32, device=image.device)
    return tf.conv2d(image.permute(2, 0, 1)[:, None, ...], weight, stride=d).squeeze(1).permute(1, 2, 0)


@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


@dataclass
class CPGSModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: CPGSModel)

    NUM_STEP: int = 50000
    rank: int = 100
    medium_num: int = 1

    cost_ratio = 1000.
    depth_clip = 4

    use_dep_loss: bool = True
    dep_loss_weight: float = 1

    recog_loss: Literal["l1", "dep_l1", "reg_l1", "reg_l2"] = "dep_l1"
    """recog loss to use"""
    ssim_loss: Literal["reg_ssim", "ssim"] = "ssim"
    """ssim loss to use"""
    cull_alpha_thresh: float = 0.5
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    continue_cull_post_densification: bool = True
    """If True, continue to cull gaussians post refinement"""
    cull_alpha_thresh_post: float = 0.1
    """threshold of opacity for post culling gaussians"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    densify_grad_thresh: float = 0.0008
    """threshold of positional gradient norm for densifying gaussians"""

    reset_alpha_every: int = 10
    """Every this many refinement steps, reset the alpha"""
    stop_screen_size_at: int = 0
    """stop culling/splitting at this step WRT screen size of gaussians"""

    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""

    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 3000
    """training starts at 1/d resolution, every n steps this is doubled"""
    num_downscales: int = 2
    """at the beginning, resolution is 1/2^d, where d is this number"""
    use_absgrad: bool = True
    """Whether to use absgrad to densify gaussians, if False, will use grad rather than absgrad"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    random_init: bool = False
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 50000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 10.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    output_depth_during_training: bool = False
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """
    camera_optimizer: CameraOptimizerConfig = field(default_factory=lambda: CameraOptimizerConfig(mode="off"))
    """Config of the camera optimizer to use"""
    use_bilateral_grid: bool = False
    """If True, use bilateral grid to handle the ISP changes in the image space. This technique was introduced in the paper 'Bilateral Guided Radiance Field Processing' (https://bilarfpro.github.io/)."""
    grid_shape: Tuple[int, int, int] = (16, 16, 8)
    """Shape of the bilateral grid (X, Y, W)"""
    color_corrected_metrics: bool = False
    """If True, apply color correction to the rendered images before computing the metrics."""


class CPGSModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: CPGSModelConfig

    def __init__(
            self,
            *args,
            seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)
        tensorly.set_backend('pytorch')

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            # means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
            means = self.seed_points[0]
        else:
            # means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)
            means = (torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale
        distances, _ = k_nearest_sklearn(means.data, 3)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        # scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        scales = torch.log(avg_dist.repeat(1, 3))
        num_points = means.shape[0]
        # quats = torch.nn.Parameter(random_quat_tensor(num_points))
        quats = random_quat_tensor(num_points)
        dim_sh = num_sh_bases(self.config.sh_degree)
        self.decomposition = CPPower(rank=self.config.rank, n_repeat=10, n_iteration=30)

        if (
                self.seed_points is not None
                and not self.config.random_init
                # We can have colors without points.
                and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            # features_dc = torch.nn.Parameter(shs[:, 0, :])
            # features_rest = torch.nn.Parameter(shs[:, 1:, :])
            features_dc = shs[:, 0, :]
            features_rest = shs[:, 1:, :]
        else:
            # features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            # features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))
            features_dc = torch.rand(num_points, 3)
            features_rest = torch.zeros((num_points, dim_sh - 1, 3))

        # opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        opacities = torch.logit(0.1 * torch.ones(num_points, 1))

        t_device = features_dc.device
        features_rest_flatten = torch.flatten(features_rest, 1).to(t_device)
        tesnor_gauss_params = torch.cat((
            means.to(t_device), scales.to(t_device), quats.to(t_device),
            features_dc.to(t_device), features_rest_flatten.to(t_device), opacities.to(t_device)
        ), dim=1).to(t_device)

        eigenvalue, factors = self.decomposition.fit_transform(tesnor_gauss_params.unsqueeze(0))
        # eigenvalue = eigenvalue.repeat(self.config.medium_num, 1)
        #  eigenvalue: [rank], factors: List[[size, rank], ...] -> [N Point, rank], [59, rank]
        CONSOLE.log(f"gauss_params: eigenvalue.shape and factors.shape: {eigenvalue.shape}",
                    *[f", {i.shape}" for i in factors])

        # eigenvalue = torch.nn.Parameter(eigenvalue, requires_grad=True)
        medium_factor = torch.nn.Parameter(factors[0].repeat(2, 1), requires_grad=True)
        num_factor = torch.nn.Parameter(factors[1] * eigenvalue, requires_grad=True)
        gs_factor = torch.nn.Parameter(factors[2], requires_grad=True)

        CONSOLE.log("any NaN or INF in cp tensor: ",
                    any([torch.isnan(i).any() & torch.isinf(i).any() for i in [eigenvalue] + factors]))
        CONSOLE.log("CP decomposition error MSE:", MSE(tesnor_gauss_params, cp_to_tensor(
            (eigenvalue, factors)
        )))

        self.seathru = SeaThruNet(cost_ratio=self.config.cost_ratio, depth_clip=self.config.depth_clip)

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": gs_factor[0:3, ...],
                "scales": gs_factor[3:6, ...],
                "quats": gs_factor[6:10, ...],
                "features_dc": gs_factor[10:13, ...],
                "features_rest": gs_factor[13:-1, ...],
                "opacities": gs_factor[-1:, ],
                "medium_factor": medium_factor,
                "num_factor": num_factor,
            }
        )

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0
        self.relu = torch.nn.ReLU()
        self.l1 = torch.nn.L1Loss()
        self.smooth_l1 = torch.nn.SmoothL1Loss(beta=0.2)

        # try:
        #     import pyiqa
        # except ImportError:
        #     print("Please install pyiqa==0.1.11")
        # assert self.config.iqa in pyiqa.list_models(), f"{self.config.iqa} not in pyiqa({pyiqa.__version__}) models list"
        # self.iqa = pyiqa.create_metric(self.config.iqa, device=self.device, as_loss=True)

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        # Strategy for GS densification
        self.strategy = CPStrategy(
            prune_opa=self.config.cull_alpha_thresh,
            continue_cull_post_densification=self.config.continue_cull_post_densification,
            prune_opa_post=self.config.cull_alpha_thresh_post,
            grow_grad2d=self.config.densify_grad_thresh,
            grow_scale3d=self.config.densify_size_thresh,
            grow_scale2d=self.config.split_screen_size,
            prune_scale3d=self.config.cull_scale_thresh,
            prune_scale2d=self.config.cull_screen_size,
            refine_scale2d_stop_iter=self.config.stop_screen_size_at,
            refine_start_iter=self.config.warmup_length,
            refine_stop_iter=self.config.stop_split_at,
            reset_every=self.config.reset_alpha_every * self.config.refine_every,
            refine_every=self.config.refine_every,
            pause_refine_after_reset=self.num_train_data + self.config.refine_every,
            absgrad=self.config.use_absgrad,
            revised_opacity=False,
            verbose=True,
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

    # @property
    # def colors(self):
    #     if self.config.sh_degree > 0:
    #         return SH2RGB(self.features_dc)
    #     else:
    #         return torch.sigmoid(self.features_dc)
    #
    # @property
    # def shs_0(self):
    #     if self.config.sh_degree > 0:
    #         return self.features_dc
    #     else:
    #         return RGB2SH(torch.sigmoid(self.features_dc))
    #
    # @property
    # def shs_rest(self):
    #     return self.features_rest
    #
    @property
    def num_points(self):
        # return self.means.shape[0]
        return self.gauss_params["num_factor"].shape[0]

    #
    # @property
    # def means(self):
    #     return self.gauss_params["means"]
    #
    @property
    def scales(self):
        # return self.gauss_params["scales"]
        return cp_to_tensor((self.eigenvalue, [self.medium_factor, self.num_factor, self.gauss_params['scales']]))

    # @property
    # def quats(self):
    #     return self.gauss_params["quats"]
    #
    # @property
    # def features_dc(self):
    #     return self.gauss_params["features_dc"]
    #
    # @property
    # def features_rest(self):
    #     return self.gauss_params["features_rest"]
    #
    # @property
    # def opacities(self):
    #     return self.gauss_params["opacities"]

    @property
    def eigenvalue(self):
        # return self.gauss_params["eigenvalue"]
        return torch.ones(self.config.rank).to(self.device)

    @property
    def medium_factor(self):
        return self.gauss_params["medium_factor"]

    @property
    def num_factor(self):
        return self.gauss_params["num_factor"]

    @property
    def gs_factor(self):
        return torch.cat([
            self.gauss_params["means"], self.gauss_params["scales"], self.gauss_params["quats"],
            self.gauss_params["features_dc"], self.gauss_params["features_rest"], self.gauss_params["opacities"]
        ], dim=0)

    @property
    def b_inf(self):
        return self.seathru.B_inf

    @property
    def j_prime(self):
        return self.seathru.J_prime

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        # self.step = 30000
        # if "means" in dict:
        #     # For backwards compatibility, we remap the names of parameters from
        #     # means->gauss_params.means since old checkpoints have that format
        #     for p in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]:
        #         dict[f"gauss_params.{p}"] = dict[p]
        # newp = dict["gauss_params.means"].shape[0]
        # CONSOLE.log(f"We have {newp} points in this 3DGS")
        # for name, param in self.gauss_params.items():
        #     old_shape = param.shape
        #     new_shape = (newp,) + old_shape[1:]
        #     self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        # super().load_state_dict(dict, **kwargs)

        self.step = self.config.NUM_STEP

        if "gauss_params.num_factor" in dict:
            new_n = dict["gauss_params.num_factor"].shape[0]
            old_shape = self.gauss_params["num_factor"].shape
            self.gauss_params["num_factor"] = torch.nn.Parameter(
                torch.zeros((new_n,) + old_shape[1:], device=self.device)
            )

        # if "gauss_params.eigenvalue" in dict:
        #     new_n = dict["gauss_params.eigenvalue"].shape[0]
        #     old_shape = self.gauss_params["eigenvalue"].shape
        #     self.gauss_params["eigenvalue"] = torch.nn.Parameter(
        #         torch.zeros((new_n,) + old_shape[1:], device=self.device)
        #     )

        super().load_state_dict(dict, **kwargs)
        CONSOLE.log(f"After load state_dict, we now have {self.gauss_params['num_factor'].shape[0]} GSs ")
        CONSOLE.log(f"b_inf : {self.b_inf}, j_prime : {self.j_prime}")

    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def step_post_backward(self, step):
        assert step == self.step
        self.strategy.step_post_backward(
            params=self.gauss_params,
            optimizers=self.optimizers,
            state=self.strategy_state,
            step=self.step,
            info=self.info,
            last_size=self.last_size,
            packed=False,
        )
        if self.step % self.config.refine_every == 0:
            CONSOLE.log(f"b_inf : {self.b_inf}, j_prime : {self.j_prime}")

    def get_training_callbacks(
            self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                self.step_cb,
                args=[training_callback_attributes.optimizers],
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.step_post_backward,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN],
                self.show_memory_allocated,
            )
        )
        return cbs

    def show_memory_allocated(self, step):
        CONSOLE.log(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 * 1024):.2f} MB")

    def step_cb(self, optimizers: Optimizers, step):
        self.step = step
        self.optimizers = optimizers.optimizers

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]] for name in self.gauss_params.keys()
            # ["means", "scales", "quats", "features_dc", "features_rest", "opacities", "num_factor", "medium_factor"]
            # ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        gps["bs"] = list(self.seathru.parameters())
        # gps["da"] = list(self.direct.parameters())
        if self.config.use_bilateral_grid:
            gps["bilateral_grid"] = list(self.bil_grids.parameters())
        self.camera_optimizer.get_param_groups(param_groups=gps)
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, image):
        d = self._get_downscale_factor()
        if d > 1:
            return resize_image(image, d)
        return image

    @staticmethod
    def get_empty_outputs(width: int, height: int, background: torch.Tensor) -> Dict[str, Union[torch.Tensor, List]]:
        rgb = background.repeat(height, width, 1)
        depth = background.new_ones(*rgb.shape[:2], 1) * 10
        accumulation = background.new_zeros(*rgb.shape[:2], 1)
        return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}

    def _get_background_color(self):
        if self.config.background_color == "random":
            if self.training:
                background = torch.rand(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        elif self.config.background_color == "white":
            background = torch.ones(3, device=self.device)
        elif self.config.background_color == "black":
            background = torch.zeros(3, device=self.device)
        else:
            raise ValueError(f"Unknown background color {self.config.background_color}")
        return background

    def _apply_bilateral_grid(self, rgb: torch.Tensor, cam_idx: int, H: int, W: int) -> torch.Tensor:
        # make xy grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, 1.0, H, device=self.device),
            torch.linspace(0, 1.0, W, device=self.device),
            indexing="ij",
        )
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

        out = slice(
            bil_grids=self.bil_grids,
            rgb=rgb,
            xy=grid_xy,
            grid_idx=torch.tensor(cam_idx, device=self.device, dtype=torch.long),
        )
        return out["rgb"]

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        CONSOLE.log(1)
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        multi_gs_tensor = cp_to_tensor(
            (self.eigenvalue, [self.medium_factor, self.num_factor, self.gs_factor])
        )
        render_list_to_gs = [multi_gs_tensor[0], multi_gs_tensor[1]]
        rgb_list, alpha_list, depth_list = [], [], []
        # for gs_tensor in [multi_gs_tensor[0], multi_gs_tensor.reshape(-1, multi_gs_tensor.shape[2])]:
        for index, gs_tensor in enumerate(render_list_to_gs):
            # cropping
            if self.crop_box is not None and not self.training:
                # crop_ids = self.crop_box.within(self.means).squeeze()
                crop_ids = self.crop_box.within(gs_tensor[:, 0:3]).squeeze()
                if crop_ids.sum() == 0:
                    return self.get_empty_outputs(
                        int(camera.width.item()), int(camera.height.item()), self.background_color
                    )
            else:
                crop_ids = None

            if crop_ids is not None:
                means_crop = gs_tensor[crop_ids, 0:3]
                scales_crop = gs_tensor[crop_ids, 3:6]
                quats_crop = gs_tensor[crop_ids, 6:10]
                features_dc_crop = gs_tensor[crop_ids, 10:13]
                features_rest_crop = gs_tensor[crop_ids, 13:-1]
                opacities_crop = gs_tensor[crop_ids, -1]
            else:
                means_crop = gs_tensor[..., 0:3]
                scales_crop = gs_tensor[..., 3:6]
                quats_crop = gs_tensor[..., 6:10]
                features_dc_crop = gs_tensor[..., 10:13]
                features_rest_crop = gs_tensor[..., 13:-1]
                opacities_crop = gs_tensor[..., -1]

            features_rest_crop = features_rest_crop.reshape(-1, features_rest_crop.shape[-1] // 3, 3)
            colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

            if self.config.sh_degree > 0:
                sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
            else:
                colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
                sh_degree_to_use = None

            render, alpha, info = rasterization(
                means=means_crop,
                quats=quats_crop,  # rasterization does normalization internally
                scales=torch.exp(scales_crop),
                opacities=torch.sigmoid(opacities_crop).squeeze(-1),
                colors=colors_crop,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sh_degree=sh_degree_to_use,
                sparse_grad=False,
                # absgrad=self.strategy.absgrad,
                absgrad=self.config.use_absgrad,
                rasterize_mode=self.config.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=0.0 if index == 0 else 3.0,
            )
            if index == 0:
                self.info = info
                if self.training:
                    self.strategy.step_pre_backward(
                        self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
                    )
            alpha = alpha[:, ...]

            rgb = torch.clamp(render[:, ..., :3], 0.0, 1.0)

            # apply bilateral grid
            if self.config.use_bilateral_grid and self.training:
                if camera.metadata is not None and "cam_idx" in camera.metadata:
                    rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, 0.01).squeeze(0)

            rgb_list.append(rgb)
            alpha_list.append(alpha)
            depth_list.append(depth_im)

        rgb_object, medium = rgb_list
        depth_objrct, depth_mediun = depth_list

        f, rgb_direct, rgb_backscatter = self.seathru(J=rgb_object, medium=medium, depth1=depth_objrct, depth2=depth_mediun)
        rgb = torch.clamp(rgb_direct + rgb_backscatter, 0.0, 1.0)

        background = self._get_background_color()
        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        return {
            "rgb": rgb.squeeze(0),
            "depth": depth_objrct,
            "depth1": depth_mediun,
            "accumulation": alpha_list[0].squeeze(0),
            "a": rgb_direct.squeeze(0),
            "b": rgb_backscatter.squeeze(0),
            "f": f.squeeze(0),
            "rgb_object": rgb_object.squeeze(0),
            "medium": medium.squeeze(0),
            "background": background
        }

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]

        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)
        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            metrics_dict["cc_psnr"] = self.psnr(cc_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        # Ll1 = torch.abs(gt_img - pred_img).mean()
        # simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])

        if self.config.recog_loss == "l1":
            recon_loss = torch.abs(gt_img - pred_img).mean()
        elif self.config.recog_loss == "dep_l1":
            depth = torch.clamp(outputs["depth"].detach(), min=1.0)
            recon_loss = (depth * torch.abs(gt_img - pred_img)).mean()
        elif self.config.recog_loss == "reg_l1":
            recon_loss = torch.abs((gt_img - pred_img) / (pred_img.detach() + 1e-3)).mean()
        else:
            recon_loss = (((pred_img - gt_img) / (pred_img.detach() + 1e-3)) ** 2).mean()

        if self.config.ssim_loss != "ssim":
            simloss = 1 - self.ssim((gt_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...],
                                    (pred_img / (pred_img.detach() + 1e-3)).permute(2, 0, 1)[None, ...])
        else:
            simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])

        seathru_loss = self.seathru.loss(
            J=outputs["rgb_object"], backscatter=outputs["b"], gt=gt_img,
            depth1=outputs["depth"].detach(), depth2=outputs["depth1"].detach(), alpha=outputs["accumulation"], step=self.step,
        )

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * recon_loss + self.config.ssim_lambda * simloss,
            "seathru_loss": seathru_loss
        }

        if self.config.use_dep_loss:
            loss_dict["dep_tv_loss"] = self.config.dep_loss_weight * total_variation_loss(
                torch.moveaxis(torch.cat([outputs["depth"], outputs["depth1"]], dim=0), -1, 1)
            )

        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (torch.maximum(
                scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1), torch.tensor(self.config.max_gauss_ratio),
            ) - self.config.max_gauss_ratio)
            loss_dict["scale_reg"] = 0.1 * scale_reg.mean()

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
            self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        predicted_rgb = outputs["rgb"]
        cc_rgb = None

        # b = outputs["b"]
        # mean, std = b.mean(dim=(0, 1)), b.std(dim=(0, 1)),
        # b = (b - mean) / std

        combined_rgb = torch.cat([
            gt_rgb, predicted_rgb, outputs["rgb_object"], outputs["a"], outputs["b"],
        ], dim=1)
        combined_depth = torch.cat([outputs["depth"], outputs["depth1"]], dim=1)
        images_dict = {"gt_pre_obj_a_b_j": combined_rgb, "d_d1": combined_depth}
        # images_dict = {"gt_ou": gt_rgb, "predicted_rgb_ou": predicted_rgb,
        #                "rgb_object_ou": outputs["rgb_object"], "a_ou": outputs["a"], "b_ou": outputs["b"],
        #                "depth_ou": outputs["depth"], "depth1_ou": outputs["depth1"],
        #                "medius_ou": outputs["medium"]}
        #
        # torch.save(images_dict, f"images_dict_{batch["image"]}_{self.step}.pt")

        # images_dict = {"gt": gt_rgb, "predicted_rgb": predicted_rgb,
        #                "rgb_object": outputs["rgb_object"], "a": outputs["a"], "b": outputs["b"]}

        if self.config.color_corrected_metrics:
            cc_rgb = color_correct(predicted_rgb, gt_rgb)
            cc_rgb = torch.moveaxis(cc_rgb, -1, 0)[None, ...]

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]
        # predicted_clear = torch.moveaxis(predicted_clear, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # psnr_clear = self.psnr(gt_rgb, predicted_clear)
        # ssim_clear = self.ssim(gt_rgb, predicted_clear)
        # lpips_clear = self.lpips(gt_rgb, predicted_clear)

        # all of these metrics will be logged as scalars
        metrics_dict = {
            "psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)
        }  # type: ignore

        if self.config.color_corrected_metrics:
            assert cc_rgb is not None
            cc_psnr = self.psnr(gt_rgb, cc_rgb)
            cc_ssim = self.ssim(gt_rgb, cc_rgb)
            cc_lpips = self.lpips(gt_rgb, cc_rgb)
            metrics_dict["cc_psnr"] = float(cc_psnr.item())
            metrics_dict["cc_ssim"] = float(cc_ssim)
            metrics_dict["cc_lpips"] = float(cc_lpips)

        # images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
