eigs_dir="../preprocess/extract/data/VOC2012/eigs"
for d in $(ls ${eigs_dir} | grep laplacian)
do
    python main.py --eigenseg --precomputed_eigs_dir ${eigs_dir}/${d} --which_matrix laplacian --dataset VOC12 --name $d --skip_if_exists
done

for d in $(ls ${eigs_dir} | grep affinity)
do
    python main.py --eigenseg --precomputed_eigs_dir ${eigs_dir}/${d} --which_matrix affinity --dataset VOC12 --name $d --skip_if_exists
done

# # Example
# eigs_dir="../preprocess/extract/data/VOC2012/eigs"; d="laplacian_mocov3_vitb16"; python main.py --eigenseg --precomputed_eigs_dir ${eigs_dir}/${d} --which_matrix laplacian --dataset VOC12 --name $d --skip_if_exists
