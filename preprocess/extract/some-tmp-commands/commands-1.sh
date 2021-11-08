# # Examples
# export TOKENIZERS_PARALLELISM=true
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py +limit_train_batches=30 +limit_val_batches=30  # debug
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py  # run, single gpu
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --use_env --nproc_per_node=4 --master_port=5432 main.py  # standard

python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features" \
--which_matrix "laplacian" \
--output_dir "./data/VOC2012/eigs/laplacian" \

#####################################################################

python extract.py extract_multi_region_segmentations \
--non_adaptive_num_segments 15 \
--features_dir "./data/VOC2012/features" \
--eigs_dir "./data/VOC2012/eigs" \
--output_dir "./data/VOC2012/multi_region_segmentation/fixed_15"

python extract.py extract_single_region_segmentations \
--features_dir "./data/VOC2012/features" \
--eigs_dir "./data/VOC2012/eigs" \
--output_dir "./data/VOC2012/single_region_segmentation/patches"

python extract.py extract_bboxes \
--features_dir "./data/VOC2012/features" \
--segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed_15" \
--num_erode 2 --num_dilate 5 \
--output_file "./data/VOC2012/multi_region_bboxes/fixed_15/bboxes_e2_d5.pth"

python extract.py extract_bbox_features \
--model_name dino_vits16 \
--images_root "./data/VOC2012/images" \
--bbox_file "./data/VOC2012/multi_region_bboxes/fixed_15/bboxes_e2_d5.pth" \
--output_file "./data/VOC2012/features" \
--output_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_features_e2_d5.pth" \

python extract.py extract_bbox_clusters \
--bbox_features_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_features_e2_d5.pth" \
--pca_dim 32 --num_clusters 20 --seed 0 \
--output_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_32_s0.pth" \

python extract.py extract_semantic_segmentations \
--segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed_15" \
--bbox_clusters_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s0.pth" \
--output_dir "./data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s0" \

################################### GAN

# scp -r luke@192.168.101.18:/home/luke/machine-learning-datasets/semantic-segmentation/VOCSegmentation/saliency_gan_model ./data/VOC2012/single_region_segmentation/

python extract.py extract_bboxes \
--features_dir "./data/VOC2012/features" \
--segmentations_dir "./data/VOC2012/single_region_segmentation/saliency_gan_model" \
--num_erode 0 --num_dilate 75 \
--output_file "./data/VOC2012/single_region_bboxes/saliency_gan_model/bboxes_e0_d75.pth"

# # This requires modification:
# from PIL import Image
# import torch
# path = "./data/VOC2012/single_region_bboxes/saliency_gan_model/bboxes_e0_d75.pth"
# bboxes_list = torch.load(path)
# for bbox_dict in bboxes_list:
#     if len(bbox_dict['bboxes']) == 0:
#         id = bbox_dict['id']
#         w, h = Image.open(f"./data/VOC2012/images/{id}.jpg").size
#         bbox_dict['bboxes'] = [[0, 0, w, h]]
#     bbox_dict['bboxes_original_resolution'] = bbox_dict['bboxes']
#     bbox_dict['segment_indices'] = [1]
# torch.save(bboxes_list, path)
# print('Done')

python extract.py extract_bbox_features \
--model_name dino_vits16 \
--images_root "./data/VOC2012/images" \
--bbox_file "./data/VOC2012/single_region_bboxes/saliency_gan_model/bboxes_e0_d75.pth" \
--output_file "./data/VOC2012/single_region_bboxes/saliency_gan_model/bbox_features_e0_d75.pth" \

python extract.py extract_bbox_clusters \
--bbox_features_file "./data/VOC2012/single_region_bboxes/saliency_gan_model/bbox_features_e0_d75.pth" \
--pca_dim 0 --num_clusters 20 \
--output_file "./data/VOC2012/single_region_bboxes/saliency_gan_model/bbox_clusters_e0_d75_pca_0.pth" \

python extract.py extract_semantic_segmentations \
--segmentations_dir "./data/VOC2012/single_region_segmentation/saliency_gan_model" \
--bbox_clusters_file "./data/VOC2012/single_region_bboxes/saliency_gan_model/bbox_clusters_e0_d75_pca_0.pth" \
--output_dir "./data/VOC2012/semantic_segmentations/patches/saliency_gan_model/segmaps_e0_d75_pca_0" \

python main.py segments_dir=/data_q1_d/extra-storage/found_new/data/VOC2012/semantic_segmentations/patches/saliency_gan_model/segmaps_e0_d75_pca_0


python main.py segments_dir=/data_q1_d/extra-storage/found_new/data/VOC2012/semantic_segmentations/patches/saliency_gan_model/segmaps_e0_d75_pca_0_s0