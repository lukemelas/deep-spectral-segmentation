# # Examples
# export TOKENIZERS_PARALLELISM=true
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py +limit_train_batches=30 +limit_val_batches=30  # debug
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py  # run, single gpu
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --nproc_per_node=4 --master_port=5432 train.py  # standard

# # Eval
# semsegdir="${PWD}/../preprocess/extract/data/VOC2012/semantic_segmentations/patches/laplacian_dino_vitb8_fixed_15"
# semsegfiles=${semsegdir}/*
# for d in $semsegfiles
# do
#     python eval.py segments_dir=${d}
# done

################################ 

# Debug
bs=144; backbone=resnet50; lamc=0; lr=1e-3; HYDRA_FULL_ERROR=1 WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=0 python train.py \
lambda_contrastive=$lamc lr=$lr data.loader.batch_size=${bs} \
model.name=${backbone} name=debug \
segments_dir="${PWD}/../preprocess/extract/data/VOC2012/semantic_segmentations/patches/laplacian_dino_vitb8_fixed_15/segmaps_e_d12_pca_0_s10" \
matching='"[(0, 0), (1, 16), (2, 11), (3, 19), (4, 2), (5, 1), (6, 6), (7, 10), (8, 15), (9, 5), (10, 8), (11, 13), (12, 14), (13, 4), (14, 3), (15, 7), (16, 17), (17, 18), (18, 12), (19, 20), (20, 9)]"' \


# ResNet example
bs=144; backbone=resnet50; lamc=0; lr=2e-4; CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --nproc_per_node=4 --master_port=5432 train.py  \
lambda_contrastive=$lamc lr=$lr data.loader.batch_size=${bs} \
model.name=${backbone} name=${backbone}_lr_${lr}_lamcon_${lambda_contrastive} \
segments_dir="${PWD}/../preprocess/extract/data/VOC2012/semantic_segmentations/patches/laplacian_dino_vitb8_fixed_15/segmaps_e_d12_pca_0_s10" \
matching='"[(0, 0), (1, 16), (2, 11), (3, 19), (4, 2), (5, 1), (6, 6), (7, 10), (8, 15), (9, 5), (10, 8), (11, 13), (12, 14), (13, 4), (14, 3), (15, 7), (16, 17), (17, 18), (18, 12), (19, 20), (20, 9)]"' \


# DINO example
bs=24; backbone=vitb8; lamc=0; lr=2e-4; CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --use_env --nproc_per_node=4 --master_port=6432 train.py  \
lambda_contrastive=$lamc lr=$lr data.loader.batch_size=${bs} \
model.name=${backbone} name=${backbone}_lr_${lr}_lamcon_${lambda_contrastive} \
segments_dir="${PWD}/../preprocess/extract/data/VOC2012/semantic_segmentations/patches/laplacian_dino_vitb8_fixed_15/segmaps_e_d12_pca_0_s10" \
matching='"[(0, 0), (1, 16), (2, 11), (3, 19), (4, 2), (5, 1), (6, 6), (7, 10), (8, 15), (9, 5), (10, 8), (11, 13), (12, 14), (13, 4), (14, 3), (15, 7), (16, 17), (17, 18), (18, 12), (19, 20), (20, 9)]"' \



# 

bs=144; backbone=resnet50; lamc=0; lr=1e-3; HYDRA_FULL_ERROR=1 WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch --use_env --nproc_per_node=2 --master_port=6499 train.py \
lambda_contrastive=$lamc lr=$lr data.loader.batch_size=${bs} \
+eval_masks_before_training=False \
model.name=${backbone} name=debug \
segments_dir="${PWD}/../preprocess/extract/data/VOC2012/semantic_segmentations/patches/laplacian_dino_vitb8_fixed_15/segmaps_e_d12_pca_0_s10" \
matching='"[(0, 0), (1, 16), (2, 11), (3, 19), (4, 2), (5, 1), (6, 6), (7, 10), (8, 15), (9, 5), (10, 8), (11, 13), (12, 14), (13, 4), (14, 3), (15, 7), (16, 17), (17, 18), (18, 12), (19, 20), (20, 9)]"' \
