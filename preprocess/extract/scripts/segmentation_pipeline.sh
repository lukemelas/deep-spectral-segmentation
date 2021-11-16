# # (1) Example set of parameters
# DATASET="VOC2012"
# MODEL="dino_vitb8"
# MATRIX="laplacian_dino_vitb8"
# DOWNSAMPLE=8
# N_SEG=15
# N_ERODE=6
# N_DILATE=12

# (2) Example set of parameters
DATASET="VOC2012"
MODEL="dino_vitb16"
MATRIX="laplacian_dino_vitb16"
DOWNSAMPLE=16
N_SEG=15
N_ERODE=2
N_DILATE=5

# Message 
echo "STARTING PIPELINE -- $(date)"
echo """
MODEL:       ${MODEL}
MATRIX:      ${MATRIX}
DOWNSAMPLE:  ${DOWNSAMPLE}
DATASET:     ${DATASET}
N_SEG:       ${N_SEG}
N_ERODE:     ${N_ERODE}
N_DILATE:    ${N_DILATE}
"""

# Extract segments
python extract.py extract_multi_region_segmentations \
--non_adaptive_num_segments ${N_SEG} \
--features_dir "./data/${DATASET}/features/${MODEL}" \
--eigs_dir "./data/${DATASET}/eigs/${MATRIX}" \
--output_dir "./data/${DATASET}/multi_region_segmentation/${MATRIX}_fixed_${N_SEG}"

# # Extract bounding boxes
# python extract.py extract_bboxes \
# --features_dir "./data/${DATASET}/features/${MODEL}" \
# --segmentations_dir "./data/${DATASET}/multi_region_segmentation/${MATRIX}_fixed_${N_SEG}" \
# --num_erode ${N_EDODE} --num_dilate ${N_DILATE} --downsample_factor ${DOWNSAMPLE} \
# --output_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}_fixed_${N_SEG}/bboxes_e${N_EDODE}_d${N_DILATE}.pth"

# python extract.py extract_bbox_features \
# --model_name ${MODEL} \
# --images_root "./data/${DATASET}/images" \
# --bbox_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}_fixed_${N_SEG}/bboxes_e${N_EDODE}_d${N_DILATE}.pth" \
# --output_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}_fixed_${N_SEG}/bbox_features_e${N_EDODE}_d${N_DILATE}.pth" \

# # Extract clusters
# for SEED in 10 11 12 13 14
# do
#     python extract.py extract_bbox_clusters \
#     --bbox_features_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}_fixed_${N_SEG}/bbox_features_e${N_EDODE}_d${N_DILATE}.pth" \
#     --pca_dim 0 --num_clusters 20 --seed 0 \
#     --output_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}_fixed_${N_SEG}/bbox_clusters_e${N_EDODE}_d${N_DILATE}_pca_0_s${SEED}.pth" \
# done

# # Create semantic segmentations
# for SEED in 10 11 12 13 14
# do
#     python extract.py extract_semantic_segmentations \
#     --segmentations_dir "./data/${DATASET}/multi_region_segmentation/${MATRIX}_fixed_${N_SEG}" \
#     --bbox_clusters_file "./data/${DATASET}/multi_region_bboxes/${MATRIX}_fixed_${N_SEG}/bbox_clusters_e${N_EDODE}_d${N_DILATE}_pca_0_s${SEED}.pth" \
#     --output_dir "./data/${DATASET}/semantic_segmentations/patches/${MATRIX}_fixed_${N_SEG}/segmaps_e${N_EDODE}_d${N_DILATE}_pca_0_s${SEED}" \
# done

# # Message
# echo "FINISHED PIPELINE -- $(date)"
