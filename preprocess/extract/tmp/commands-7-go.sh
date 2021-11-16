python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/mocov3_vits16" \
--which_matrix "laplacian" \
--output_dir "./data/VOC2012/eigs/laplacian_mocov3_vits16"

python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/mocov3_vitb16" \
--which_matrix "laplacian" \
--output_dir "./data/VOC2012/eigs/laplacian_mocov3_vitb16"

python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/dino_vits16" \
--which_matrix "laplacian" \
--output_dir "./data/VOC2012/eigs/laplacian_dino_vits16"

python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/dino_vitb16" \
--which_matrix "laplacian" \
--output_dir "./data/VOC2012/eigs/laplacian_dino_vitb16"

python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/dino_vits8" \
--which_matrix "laplacian" \
--output_dir "./data/VOC2012/eigs/laplacian_dino_vits8"

python extract.py extract_eigs \
--images_root "./data/VOC2012/images" \
--features_dir "./data/VOC2012/features/dino_vitb8" \
--which_matrix "laplacian" \
--output_dir "./data/VOC2012/eigs/laplacian_dino_vitb8"

############################### 

for lam in 0.0 0.10 1.0 10.0 100.0 
do
    python extract.py extract_eigs \
    --images_root "./data/VOC2012/images" \
    --features_dir "./data/VOC2012/features/dino_vits16" \
    --which_matrix "matting_laplacian" \
    --output_dir "./data/VOC2012/eigs/matting_laplacian_dino_vits16_16x_lambda_${lam}" --image_downsample_factor 16 --image_color_lambda ${lam}
done

for lam in 0.0 0.10 1.0 100.0  # 10.0 
do
    python extract.py extract_eigs \
    --images_root "./data/VOC2012/images" \
    --features_dir "./data/VOC2012/features/dino_vitb16" \
    --which_matrix "matting_laplacian" \
    --output_dir "./data/VOC2012/eigs/matting_laplacian_dino_vitb16_8x_lambda_${lam}" --image_downsample_factor 8 --image_color_lambda ${lam}
done
