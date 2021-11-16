
res_dir="outputs/VOC12_train/"
for d in $(ls ${res_dir})
do
    echo $(cat ${res_dir}/${d}/results.txt) -- ${d}
done
