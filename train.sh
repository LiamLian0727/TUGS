# ns-train 的示例
ns-train cpgs_v4 --vis viewer+wandb colmap --downscale-factor 1 --colmap-path sparse/0 --data /root/data/SeathruNeRF_dataset/Panama --images-path images_wb
ns-train cpgs_v4 --vis viewer+wandb colmap --downscale-factor 1 --colmap-path sparse/0 --data /root/data/SeathruNeRF_dataset/JapaneseGradens-RedSea --images-path images_wb
ns-train cpgs_v4 --vis viewer+wandb colmap --downscale-factor 1 --colmap-path sparse/0 --data /root/data/SeathruNeRF_dataset/IUI3-RedSea --images-path Images_wb
ns-train cpgs_v4 --vis viewer+wandb colmap --downscale-factor 1 --colmap-path sparse/0 --data /root/data/SeathruNeRF_dataset/Curasao --images-path images_wb

# 评估示例  
ns-eval --load-config outputs/unnamed/cpgs_v4/2025-02-27_131439/config.yml --render-output-path renders/eval
