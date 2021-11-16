# dino_vits16_8x_lambda_10_fixed_10
# dino_vits16_baseline_fixed_15
# dino_vits16_8x_lambda_50_fixed_15 

####### 

# python extract.py extract_bboxes \
# --features_dir "./data/VOC2012/features/dino_vits16" \
# --segmentations_dir "./data/VOC2012/multi_region_segmentation/dino_vits16_8x_lambda_10_fixed_10" \
# --num_erode 2 --num_dilate 5 --downsample_factor 8 \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_10/bboxes_e2_d5.pth"

# python extract.py extract_bboxes \
# --features_dir "./data/VOC2012/features/dino_vits16" \
# --segmentations_dir "./data/VOC2012/multi_region_segmentation/dino_vits16_baseline_fixed_15" \
# --num_erode 2 --num_dilate 5 --downsample_factor 8 \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_baseline_fixed_15/bboxes_e2_d5.pth"

# python extract.py extract_bboxes \
# --features_dir "./data/VOC2012/features/dino_vits16" \
# --segmentations_dir "./data/VOC2012/multi_region_segmentation/dino_vits16_8x_lambda_50_fixed_15" \
# --num_erode 2 --num_dilate 5 --downsample_factor 8 \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_50_fixed_15/bboxes_e2_d5.pth"

####### 

# CUDA_VISIBLE_DEVICES=1 python extract.py extract_bbox_features \
# --model_name dino_vits16 \
# --images_root "./data/VOC2012/images" \
# --bbox_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_10/bboxes_e2_d5.pth" \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_10/bbox_features_e2_d5.pth" \

# CUDA_VISIBLE_DEVICES=2 python extract.py extract_bbox_features \
# --model_name dino_vits16 \
# --images_root "./data/VOC2012/images" \
# --bbox_file "./data/VOC2012/multi_region_bboxes/dino_vits16_baseline_fixed_15/bboxes_e2_d5.pth" \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_baseline_fixed_15/bbox_features_e2_d5.pth" \

# CUDA_VISIBLE_DEVICES=3 python extract.py extract_bbox_features \
# --model_name dino_vits16 \
# --images_root "./data/VOC2012/images" \
# --bbox_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_50_fixed_15/bboxes_e2_d5.pth" \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_50_fixed_15/bbox_features_e2_d5.pth" \

####### 

for i in 10 11 12 13 14
do

    python extract.py extract_bbox_clusters \
    --bbox_features_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_10/bbox_features_e2_d5.pth" \
    --pca_dim 0 --num_clusters 20 --seed ${i} \
    --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_10/bbox_clusters_e2_d5_pca_0_s${i}.pth" \

    python extract.py extract_semantic_segmentations \
    --segmentations_dir "./data/VOC2012/multi_region_segmentation/dino_vits16_8x_lambda_10_fixed_10" \
    --bbox_clusters_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_10/bbox_clusters_e2_d5_pca_0_s${i}.pth" \
    --output_dir "./data/VOC2012/semantic_segmentations/patches/dino_vits16_8x_lambda_10_fixed_10/segmaps_e2_d5_pca_0_s${i}" \

done


for i in 10 11 12 13 14
do

    python extract.py extract_bbox_clusters \
    --bbox_features_file "./data/VOC2012/multi_region_bboxes/dino_vits16_baseline_fixed_15/bbox_features_e2_d5.pth" \
    --pca_dim 0 --num_clusters 20 --seed ${i} \
    --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_baseline_fixed_15/bbox_clusters_e2_d5_pca_0_s${i}.pth" \

    python extract.py extract_semantic_segmentations \
    --segmentations_dir "./data/VOC2012/multi_region_segmentation/dino_vits16_baseline_fixed_15" \
    --bbox_clusters_file "./data/VOC2012/multi_region_bboxes/dino_vits16_baseline_fixed_15/bbox_clusters_e2_d5_pca_0_s${i}.pth" \
    --output_dir "./data/VOC2012/semantic_segmentations/patches/dino_vits16_baseline_fixed_15/segmaps_e2_d5_pca_0_s${i}" \

done


for i in 10 11 12 13 14
do

    python extract.py extract_bbox_clusters \
    --bbox_features_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_50_fixed_15/bbox_features_e2_d5.pth" \
    --pca_dim 0 --num_clusters 20 --seed ${i} \
    --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_50_fixed_15/bbox_clusters_e2_d5_pca_0_s${i}.pth" \

    python extract.py extract_semantic_segmentations \
    --segmentations_dir "./data/VOC2012/multi_region_segmentation/dino_vits16_8x_lambda_50_fixed_15" \
    --bbox_clusters_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_50_fixed_15/bbox_clusters_e2_d5_pca_0_s${i}.pth" \
    --output_dir "./data/VOC2012/semantic_segmentations/patches/dino_vits16_8x_lambda_50_fixed_15/segmaps_e2_d5_pca_0_s${i}" \

done
