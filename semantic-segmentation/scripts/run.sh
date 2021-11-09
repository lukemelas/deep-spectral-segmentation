# Examples
export TOKENIZERS_PARALLELISM=true
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py +limit_train_batches=30 +limit_val_batches=30  # debug
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py  # run, single gpu
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --nproc_per_node=4 --master_port=5432 train.py  # standard

# Debug
WANDB_MODE=dryrun CUDA_VISIBLE_DEVICES=3 python train.py +limit_train_batches=40 data.loader.batch_size=8 eval_every=1000 +eval_masks_before_training=False