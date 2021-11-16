python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/dino_vitb16" \
--which_matrix "matting_laplacian" \
--output_dir "./data/VOC2012/eigs/matting_laplacian_dino_vitb16_16x_lambda_0.2" --image_downsample_factor 16 --image_color_lambda 0.2

# python extract.py extract_eigs \
# --images_root "./data/VOC2012/images" \
# --features_dir "./data/VOC2012/features/dino_vitb16" \
# --which_matrix "matting_laplacian" \
# --output_dir "./data/VOC2012/eigs/matting_laplacian_dino_vitb16_16x_lambda_0.5" --image_downsample_factor 16 --image_color_lambda 1.0

python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/dino_vitb16" \
--which_matrix "matting_laplacian" \
--output_dir "./data/VOC2012/eigs/matting_laplacian_dino_vitb16_16x_lambda_1.0" --image_downsample_factor 16 --image_color_lambda 1.0

python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/dino_vitb16" \
--which_matrix "matting_laplacian" \
--output_dir "./data/VOC2012/eigs/matting_laplacian_dino_vitb16_16x_lambda_2.0" --image_downsample_factor 16 --image_color_lambda 2.0

python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/dino_vitb16" \
--which_matrix "matting_laplacian" \
--output_dir "./data/VOC2012/eigs/matting_laplacian_dino_vitb16_16x_lambda_5.0" --image_downsample_factor 16 --image_color_lambda 5.0

