# multiple seeds

# python extract.py extract_bbox_clusters \
# --bbox_features_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_features_e2_d5.pth" \
# --pca_dim 0 --num_clusters 20 --seed 10 \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_clusters_e2_d5_pca_0_s10.pth" \

# python extract.py extract_bbox_clusters \
# --bbox_features_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_features_e2_d5.pth" \
# --pca_dim 0 --num_clusters 20 --seed 11 \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_clusters_e2_d5_pca_0_s11.pth" \

# python extract.py extract_bbox_clusters \
# --bbox_features_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_features_e2_d5.pth" \
# --pca_dim 0 --num_clusters 20 --seed 12 \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_clusters_e2_d5_pca_0_s12.pth" \

# python extract.py extract_bbox_clusters \
# --bbox_features_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_features_e2_d5.pth" \
# --pca_dim 0 --num_clusters 20 --seed 13 \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_clusters_e2_d5_pca_0_s13.pth" \

# python extract.py extract_bbox_clusters \
# --bbox_features_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_features_e2_d5.pth" \
# --pca_dim 0 --num_clusters 20 --seed 14 \
# --output_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_clusters_e2_d5_pca_0_s14.pth" \


for i in 10 11 12 13 14
do
    python extract.py extract_semantic_segmentations \
    --segmentations_dir "./data/VOC2012/multi_region_segmentation/dino_vits16_8x_lambda_10_fixed_15" \
    --bbox_clusters_file "./data/VOC2012/multi_region_bboxes/dino_vits16_8x_lambda_10_fixed_15/bbox_clusters_e2_d5_pca_0_s${i}.pth" \
    --output_dir "./data/VOC2012/semantic_segmentations/patches/dino_vits16_8x_lambda_10_fixed_15/segmaps_e2_d5_pca_0_s${i}"
done


for i in 10 11 12 13 14
do
    echo """python eval.py segments_dir=/data_q1_d/extra-storage/found_new/./data/VOC2012/semantic_segmentations/patches/dino_vits16_8x_lambda_10_fixed_15/segmaps_e2_d5_pca_0_s${i}"""
done


python eval.py segments_dir=/data_q1_d/extra-storage/found_new/./data/VOC2012/semantic_segmentations/patches/dino_vits16_8x_lambda_10_fixed_15/segmaps_e2_d5_pca_0_s10
python eval.py segments_dir=/data_q1_d/extra-storage/found_new/./data/VOC2012/semantic_segmentations/patches/dino_vits16_8x_lambda_10_fixed_15/segmaps_e2_d5_pca_0_s11
python eval.py segments_dir=/data_q1_d/extra-storage/found_new/./data/VOC2012/semantic_segmentations/patches/dino_vits16_8x_lambda_10_fixed_15/segmaps_e2_d5_pca_0_s12
python eval.py segments_dir=/data_q1_d/extra-storage/found_new/./data/VOC2012/semantic_segmentations/patches/dino_vits16_8x_lambda_10_fixed_15/segmaps_e2_d5_pca_0_s13
python eval.py segments_dir=/data_q1_d/extra-storage/found_new/./data/VOC2012/semantic_segmentations/patches/dino_vits16_8x_lambda_10_fixed_15/segmaps_e2_d5_pca_0_s14