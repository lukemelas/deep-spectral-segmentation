# Examples
# ./scripts/evaluate.sh VOC2012 VOC12
# ./scripts/evaluate.sh VOC2007 VOC07

# Params
dataset=${1:-VOC2012}
split=${2:-VOC12}

# Print to user
echo "dataset: $dataset"
echo "split: $split"

# Vars
eigs_dir="../preprocess/extract/data/${dataset}/eigs"

# Stuff
for d in $(ls ${eigs_dir})
do
    echo ""
    outdir="./outputs/${split}_train/$d"
    if [ -d ${outdir} ] && [ ! "$(echo "${outdir}/"*)" = "${outdir}/*" ]
    then
        echo "Output directory already exists and is not empty: $outdir"
    else
        echo "Running with output directory: $outdir"
        # echo "python main.py --eigenseg --precomputed_eigs_dir ${eigs_dir}/${d} --which_matrix laplacian --dataset $split --name $d --skip_if_exists"
        python main.py \
        --eigenseg \
        --precomputed_eigs_dir ${eigs_dir}/${d} \
        --dataset $split \
        --name $d \
        --skip_if_exists
    fi
done


# # And here is an example of a single run:
# d=affinity_mocov3_vitb16_16x_lambda_0.0_nonormalize_threshold_at_zero; dataset=VOC2007; split=VOC07; eigs_dir="../preprocess/extract/data/${dataset}/eigs"; python main.py \
# --eigenseg \
# --precomputed_eigs_dir ${eigs_dir}/${d} \
# --dataset $split \
# --name debug \
# --skip_if_exists