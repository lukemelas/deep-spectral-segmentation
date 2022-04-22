# Examples
# ./scripts/print-results.sh VOC12
# ./scripts/print-results.sh VOC07

# Params
split=${1:-VOC12}

# Print to user
echo "split: $split"

res_dir="outputs/${split}_trainval/"
for d in $(ls ${res_dir})
do
    echo $(cat ${res_dir}/${d}/results.txt) -- ${d}
done

