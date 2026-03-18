from dataclasses import dataclass
from typing import Any, Dict, Tuple, Union

import torch
from tensorly import cp_to_tensor
from tensorly.decomposition import parafac
from torch import Tensor
import torch.nn.functional as F

from gsplat import Strategy
from gsplat.utils import normalized_quat_to_rotmat
from gsplat.strategy.ops import _update_param_with_optimizer


@dataclass
class CPStrategy(Strategy):
    """A default strategy that follows the original 3DGS paper:

    `3D Gaussian Splatting for Real-Time Radiance Field Rendering <https://arxiv.org/abs/2308.04079>`_

    The strategy will:

    - Periodically duplicate GSs with high image plane gradients and small scales.
    - Periodically split GSs with high image plane gradients and large scales.
    - Periodically prune GSs with low opacity.
    - Periodically reset GSs to a lower opacity.

    If `absgrad=True`, it will use the absolute gradients instead of average gradients
    for GS duplicating & splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`_

    Which typically leads to better results but requires to set the `grow_grad2d` to a
    higher value, e.g., 0.0008. Also, the :func:`rasterization` function should be called
    with `absgrad=True` as well so that the absolute gradients are computed.

    Args:
        prune_opa (float): GSs with opacity below this value will be pruned. Default is 0.005.
        grow_grad2d (float): GSs with image plane gradient above this value will be
          split/duplicated. Default is 0.0002.
        grow_scale3d (float): GSs with 3d scale (normalized by scene_scale) below this
          value will be duplicated. Above will be split. Default is 0.01.
        grow_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be split. Default is 0.05.
        prune_scale3d (float): GSs with 3d scale (normalized by scene_scale) above this
          value will be pruned. Default is 0.1.
        prune_scale2d (float): GSs with 2d scale (normalized by image resolution) above
          this value will be pruned. Default is 0.15.
        refine_scale2d_stop_iter (int): Stop refining GSs based on 2d scale after this
          iteration. Default is 0. Set to a positive value to enable this feature.
        refine_start_iter (int): Start refining GSs after this iteration. Default is 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default is 15_000.
        reset_every (int): Reset opacities every this steps. Default is 3000.
        refine_every (int): Refine GSs every this steps. Default is 100.
        pause_refine_after_reset (int): Pause refining GSs until this number of steps after
          reset, Default is 0 (no pause at all) and one might want to set this number to the
          number of images in training set.
        absgrad (bool): Use absolute gradients for GS splitting. Default is False.
        revised_opacity (bool): Whether to use revised opacity heuristic from
          arXiv:2404.06109 (experimental). Default is False.
        verbose (bool): Whether to print verbose information. Default is False.

    Examples:

        >>> from gsplat import DefaultStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = DefaultStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)

    """

    prune_opa: float = 0.005
    continue_cull_post_densification: bool = False
    prune_opa_post: float = 0

    grow_grad2d: float = 0.0002
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.5
    prune_scale3d_post: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = False
    revised_opacity: bool = False
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            info: Dict[str, Any],
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
                "means2d" in info
        ), "The 2D means of the Gaussians is required but missing."
        info["means2d"].retain_grad()

    def step_post_backward(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            info: Dict[str, Any],
            last_size: Tuple[int, int],
            packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            if self.continue_cull_post_densification and step % self.refine_every == 0:
                n_prune = self._prune_gs_post(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_prune} GSs deletes after refine_stop_iter "
                        f"because under a certain opacity threshold {self.prune_opa_post}. "
                        f"Now having {params['num_factor'].shape[0]} GSs."
                    )
                # del state["opacities"]
                torch.cuda.empty_cache()
                return
            else:
                return

        self._update_state(params, state, info, packed=packed)

        if (
                step > self.refine_start_iter
                and step % self.refine_every == 0
                and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            # opacities = params["opacities"]
            # eigenvalue = torch.ones(opacities.shape[-1]).to(opacities.device)
            # so = cp_to_tensor((
            #     eigenvalue, [
            #         params["medium_factor"][0], params["num_factor"], torch.cat((params["scales"], opacities), dim=0)
            #     ]
            # ))[0]
            # state["scales"] = so[:, 0:-1]
            # state["opacities"] = so[:, -1:]
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step, last_size)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {params['num_factor'].shape[0]} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {params['num_factor'].shape[0]} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            # del state["scales"]
            # del state["opacities"]
            torch.cuda.empty_cache()

            if step % self.reset_every == 0:
                reset_opa(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    value=0,
                )
                print(f"Step {step}: reset opacities factor for GSs.")

    def _update_state(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            state: Dict[str, Any],
            info: Dict[str, Any],
            packed: bool = False,
    ):
        for key in ["means2d", "width", "height", "n_cameras", "radii", "gaussian_ids"]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info["means2d"].absgrad.clone()
        else:
            grads = info["means2d"].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        # n_gaussian = len(list(params.values())[0])
        n_gaussian = params['num_factor'].shape[0]
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = info["radii"] > 0.0  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel]  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
            last_size: Tuple[int, int]
    ) -> Tuple[int, int]:
        count = state["count"]
        # grads = (state["grad2d"] / count.clamp_min(1)) * 0.5 * max(last_size[0], last_size[1])
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d
        # is_small = (
        #         torch.exp(params["scales"]).max(dim=-1).values
        #         <= self.grow_scale3d * state["scene_scale"]
        # )
        # is_small = (torch.exp(state["scales"]).max(dim=-1).values <= self.grow_scale3d * state["scene_scale"])
        # is_dupli = is_grad_high & is_small
        # n_dupli = is_dupli.sum().item()
        #
        # is_large = ~is_small
        # is_split = is_grad_high & is_large
        # if step < self.refine_scale2d_stop_iter:
        #     is_split |= state["radii"] > self.grow_scale2d
        # n_split = is_split.sum().item()
        if step < self.refine_scale2d_stop_iter:
            is_grad_high |= state["radii"] > self.grow_scale2d
        n_dupli = is_grad_high.sum().item()

        # first duplicate
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_grad_high)

        # # new GSs added by duplication will not be split
        # is_split = torch.cat(
        #     [
        #         is_split,
        #         torch.zeros(n_dupli, dtype=torch.bool, device=device),
        #     ]
        # )
        #
        # # then split
        # if n_split > 0:
        #     split(
        #         params=params,
        #         optimizers=optimizers,
        #         state=state,
        #         mask=is_split,
        #         revised_opacity=self.revised_opacity,
        #     )
        # return n_dupli, n_split
        return n_dupli, 0

    @torch.no_grad()
    def _prune_gs(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
    ) -> int:
        # is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        opacities = params["opacities"].detach()
        opacities = cp_to_tensor(
            (torch.ones(opacities.shape[-1]).to(opacities.device),
             [params["medium_factor"][0:1, :], params["num_factor"], opacities])
        )[0]
        is_prune = torch.sigmoid(opacities.flatten()) < self.prune_opa
        # is_prune |= torch.sigmoid(state["opacities2"].flatten()) < self.prune_opa2
        # if step > self.reset_every:
        #     # is_too_big = (torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale3d * state["scene_scale"])
        #     is_too_big = (torch.exp(state["scales"]).max(dim=-1).values > self.prune_scale3d * state["scene_scale"])
        #     # The official code also implements sreen-size pruning but
        #     # it's actually not being used due to a bug:
        #     # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
        #     # We implement it here for completeness but set `refine_scale2d_stop_iter`
        #     # to 0 by default to disable it.
        #     if step < self.refine_scale2d_stop_iter:
        #         is_too_big |= state["radii"] > self.prune_scale2d
        #
        #     is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        del opacities

        return n_prune

    @torch.no_grad()
    def _prune_gs_post(
            self,
            params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
            optimizers: Dict[str, torch.optim.Optimizer],
            state: Dict[str, Any],
            step: int,
    ) -> int:
        # is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa_post
        # is_prune = torch.sigmoid(state["opacities"].flatten()) < self.prune_opa_post
        opacities = params["opacities"].detach()
        opacities = cp_to_tensor(
            (torch.ones(opacities.shape[-1]).to(opacities.device),
             [params["medium_factor"][0:1, :], params["num_factor"], opacities])
        )[0]
        is_prune = torch.sigmoid(opacities.flatten()) < self.prune_opa

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        del opacities

        return n_prune


@torch.no_grad()
def duplicate(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        mask: Tensor,
):
    """Inplace duplicate the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
    """
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "num_factor":
            return torch.nn.Parameter(torch.cat([p, p[sel]]))
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=["num_factor"])
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))


@torch.no_grad()
def split(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        mask: Tensor,
        revised_opacity: bool = False,
):
    """Inplace split the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to split the Gaussians.
        revised_opacity: Whether to use revised opacity formulation
          from arXiv:2404.06109. Default: False.
    """
    raise NotImplementedError
    # device = mask.device
    # sel = torch.where(mask)[0]
    # rest = torch.where(~mask)[0]
    #
    # # scales = torch.exp(params["scales"][sel])
    # # quats = F.normalize(params["quats"][sel], dim=-1)
    # # rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    #
    # add_gs = state["need_tensor"][sel]
    # scales = torch.exp(add_gs[:, 3:6])
    # quats = F.normalize(add_gs[:, 6:10], dim=-1)
    # rotmats = normalized_quat_to_rotmat(quats)  # [N, 3, 3]
    # samples = torch.einsum(
    #     "nij,nj,bnj->bni",
    #     rotmats,
    #     scales,
    #     torch.randn(2, len(scales), 3, device=device),
    # )  # [2, N, 3]
    # repeats = [2] + [1] * (add_gs.dim() - 1)
    # means = (add_gs + samples).reshape(-1, 3)
    # scales = torch.log(scales / 1.6).repeat(2, 1)
    # new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(add_gs))
    # opacities = torch.logit(new_opacities).repeat(repeats)
    #
    # gs_factor = params["gs_factor"]
    # medium_factor = params["medium_factor"]
    # num_factor = params["num_factor"][sel].repeat(2, 1)
    #
    # add_gs = torch.cat([
    #     means, scales, add_gs[6:-1].repeat(repeats), opacities
    # ], dim=1).T.unsqueeze(1).repeat(1, medium_factor.shape[0], 1)  # (59, medium_num, 2N)
    #
    # rank = gs_factor.shape[1]
    # weights, factors = parafac(add_gs, rank=rank, init=[
    #     torch.ones(rank).to(device), [gs_factor, medium_factor, num_factor]
    # ], fixed_modes=[0, 1], n_iter_max=100)
    #
    # def param_fn(name: str, p: Tensor) -> Tensor:
    #     # repeats = [2] + [1] * (p.dim() - 1)
    #     # if name == "means":
    #     #     p_split = (p[sel] + samples).reshape(-1, 3)  # [2N, 3]
    #     # elif name == "scales":
    #     #     p_split = torch.log(scales / 1.6).repeat(2, 1)  # [2N, 3]
    #     # elif name == "opacities" and revised_opacity:
    #     #     new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
    #     #     p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
    #     # else:
    #     #     p_split = p[sel].repeat(repeats)
    #     # p_new = torch.cat([p[rest], p_split])
    #     # p_new = torch.nn.Parameter(p_new)
    #     # return p_new
    #
    #     if name == "num_factor":
    #         return torch.nn.Parameter(torch.cat([p[rest], factors[-1]]))
    #     else:
    #         raise ValueError(f"Unexpected parameter name: {name}")
    #
    # def optimizer_fn(key: str, v: Tensor) -> Tensor:
    #     v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
    #     return torch.cat([v[rest], v_split])
    #
    # # update the parameters and the state in the optimizers
    # _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=["num_factor"])
    # # update the extra running state
    # for k, v in state.items():
    #     if isinstance(v, torch.Tensor) and k != "need_tensor":
    #         repeats = [2] + [1] * (v.dim() - 1)
    #         v_new = v[sel].repeat(repeats)
    #         state[k] = torch.cat((v[rest], v_new))


@torch.no_grad()
def remove(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        mask: Tensor,
):
    """Inplace remove the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to remove the Gaussians.
    """
    sel = torch.where(~mask)[0]

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "num_factor":
            return torch.nn.Parameter(p[sel])
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return v[sel]

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers, names=["num_factor"])
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v[sel]


@torch.no_grad()
def reset_opa(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Tensor],
        value: float,
):
    """Inplace reset the opacities to the given post-sigmoid value.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        value: The value to reset the opacities
    """

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            opacities = torch.clamp(p, max=(torch.tensor(value)).item())
            return torch.nn.Parameter(opacities)
        else:
            raise ValueError(f"Unexpected parameter name: {name}")

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        return torch.zeros_like(v)

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(
        param_fn, optimizer_fn, params, optimizers, names=["opacities"]
    )
