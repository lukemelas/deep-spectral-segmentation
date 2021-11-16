######################################################


python extract.py extract_bbox_clusters \
--bbox_features_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_features_e2_d5.pth" \
--pca_dim 0 --num_clusters 20 --seed 10 \
--output_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s10.pth" \

python extract.py extract_bbox_clusters \
--bbox_features_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_features_e2_d5.pth" \
--pca_dim 0 --num_clusters 20 --seed 11 \
--output_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s11.pth" \

python extract.py extract_bbox_clusters \
--bbox_features_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_features_e2_d5.pth" \
--pca_dim 0 --num_clusters 20 --seed 12 \
--output_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s12.pth" \

python extract.py extract_bbox_clusters \
--bbox_features_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_features_e2_d5.pth" \
--pca_dim 0 --num_clusters 20 --seed 13 \
--output_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s13.pth" \

python extract.py extract_bbox_clusters \
--bbox_features_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_features_e2_d5.pth" \
--pca_dim 0 --num_clusters 20 --seed 14 \
--output_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s14.pth" \

python extract.py extract_bbox_clusters \
--bbox_features_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_features_e2_d5.pth" \
--pca_dim 0 --num_clusters 20 --seed 15 \
--output_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s15.pth" \

######################################################

python extract.py extract_semantic_segmentations \
--segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed_15" \
--bbox_clusters_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s10.pth" \
--output_dir "./data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s10" \

python extract.py extract_semantic_segmentations \
--segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed_15" \
--bbox_clusters_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s11.pth" \
--output_dir "./data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s11" \

python extract.py extract_semantic_segmentations \
--segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed_15" \
--bbox_clusters_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s12.pth" \
--output_dir "./data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s12" \

python extract.py extract_semantic_segmentations \
--segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed_15" \
--bbox_clusters_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s13.pth" \
--output_dir "./data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s13" \

python extract.py extract_semantic_segmentations \
--segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed_15" \
--bbox_clusters_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s14.pth" \
--output_dir "./data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s14" \

python extract.py extract_semantic_segmentations \
--segmentations_dir "./data/VOC2012/multi_region_segmentation/fixed_15" \
--bbox_clusters_file "./data/VOC2012/multi_region_bboxes/fixed_15/bbox_clusters_e2_d5_pca_0_s15.pth" \
--output_dir "./data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s15" \

######################################################

python main.py segments_dir=/data_q1_d/extra-storage/found_new/data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s0
python main.py segments_dir=/data_q1_d/extra-storage/found_new/data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s1
python main.py segments_dir=/data_q1_d/extra-storage/found_new/data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s2
python main.py segments_dir=/data_q1_d/extra-storage/found_new/data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s3
python main.py segments_dir=/data_q1_d/extra-storage/found_new/data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s4
python main.py segments_dir=/data_q1_d/extra-storage/found_new/data/VOC2012/semantic_segmentations/patches/fixed_15/segmaps_e2_d5_pca_0_s5
