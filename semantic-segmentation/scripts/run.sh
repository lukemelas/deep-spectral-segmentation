# # Examples
# export TOKENIZERS_PARALLELISM=true
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py +limit_train_batches=30 +limit_val_batches=30  # debug
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py  # run, single gpu
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --nproc_per_node=4 --master_port=5432 train.py  # standard

# Debug
# WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=3 python train.py +limit_train_batches=40 data.loader.batch_size=8 eval_every=1000 +eval_masks_before_training=False model.name=vits16

for d in segmaps_e_d5_pca_0_s10 segmaps_e_d5_pca_0_s11 segmaps_e_d5_pca_0_s12 segmaps_e_d5_pca_0_s13 segmaps_e_d5_pca_0_s14
do
    python eval.py segments_dir="${PWD}/../preprocess/extract/data/VOC2012/semantic_segmentations/patches/laplacian_dino_vitb16_fixed_15/${d}"
done

