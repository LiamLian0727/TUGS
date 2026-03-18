from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig

from .full_images_datamanager import FullImageDatamanagerConfig

from .tugs_cp import CPGSModelConfig as TUGSCPModelConfig

NUM_STEP = 20000
max_norm = 10
tugs_cp_method  = MethodSpecification(
    config=TrainerConfig(
        method_name="cpgs_v4",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=2000,
        steps_per_eval_all_images=1000,
        max_num_iterations=NUM_STEP,
        mixed_precision=False,
        pipeline=VanillaPipelineConfig(
            datamanager=FullImageDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=TUGSCPModelConfig(
                NUM_STEP=NUM_STEP,
                sh_degree=3,
                stop_split_at=10000,
                rank=20,
                dep_loss_weight=1,
                warmup_length=1000,
                # densify_size_thresh=0.01,
                densify_grad_thresh=0.001,
                output_depth_during_training=True,
                cull_alpha_thresh=0.1,
                continue_cull_post_densification=True,
                cull_alpha_thresh_post=0.005,
                reset_alpha_every=5,
                # use_scale_regularization=True,
                sh_degree_interval=1000
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15, max_norm=max_norm),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=NUM_STEP,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15, max_norm=max_norm),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15, max_norm=max_norm),
                # "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15, max_norm=10),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15, max_norm=max_norm),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15, max_norm=max_norm),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15, max_norm=max_norm),
                "scheduler": None
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15, max_norm=max_norm),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=NUM_STEP, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15, max_norm=max_norm),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=NUM_STEP, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "num_factor": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15, max_norm=max_norm),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=NUM_STEP,
                ),
            },
            # "eigenvalue": {
            #     "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            #     "scheduler": ExponentialDecaySchedulerConfig(
            #         lr_final=5e-5,
            #         max_steps=NUM_STEP,
            #     ),
            # },
            "medium_factor": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15, max_norm=max_norm),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5,
                    max_steps=NUM_STEP,
                ),
            },
            "bs": {
                "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=max_norm),
                "scheduler": ExponentialDecaySchedulerConfig(
                    # warmup_steps=2000,
                    lr_final=1e-4,
                    max_steps=NUM_STEP,
                ),
            },
            # "da": {
            #     "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15, max_norm=max_norm),
            #     "scheduler": ExponentialDecaySchedulerConfig(
            #         # warmup_steps=2000,
            #         lr_final=1.e-4,
            #         max_steps=NUM_STEP,
            #     ),
            # }
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Test 3DGS.",
)
