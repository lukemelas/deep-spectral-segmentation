
# [Note] From the runs, it seems like `normalize` is better, `threshold_at_zero` is absolutely 
# critical, `lapnorm` is better, and `lam` > 2 is better. I should look into why thresholding 
# at zero is essential for `laplacian` and terrible for `affinity`.


# Laplacians
for ds in 16  # 8 16
do 
    for model in dino_vits16 # mocov3_vits16 mocov3_vitb16 dino_vits16 dino_vitb16 dino_vits8 dino_vitb8
    do 
        for norm in normalize  # nonormalize 
        do 
            for thresh in threshold_at_zero  # nothreshold_at_zero 
            do 
                for lapnorm in lapnorm  # nolapnorm 
                do 
                    for lam in 0.0 # 1.0 2.0 10.0 20.0 50.0 100.0 500.0 
                    do
                        echo "******************"
                        echo "model=${model}"
                        echo "ds=${ds}"
                        echo "lam=${lam}"
                        echo "******************"

                        dataset=VOC2007; python extract.py extract_eigs \
                        --images_root "./data/${dataset}/images" \
                        --features_dir "./data/${dataset}/features/${model}/" \
                        --which_matrix "laplacian" \
                        --output_dir "./data/${dataset}/eigs/laplacian_${model}_${ds}x_lambda_${lam}_${norm}_${thresh}_${lapnorm}" \
                        --image_downsample_factor ${ds} \
                        --image_color_lambda ${lam} \
                        --K 2 \
                        --${norm} --${thresh} --${lapnorm}
                    done
                done
            done
        done
    done
done


# Affinities (not normalized)
for ds in 16  # 8
do 
    for model in mocov3_vits16 mocov3_vitb16 # dino_vits16 dino_vitb16  # dino_vits8
    do 
        for norm in normalize nonormalize 
        do 
            for thresh in threshold_at_zero nothreshold_at_zero 
            do 
                for lam in 0.0
                do 
    
                echo "******************"
                echo "model=${model}"
                echo "ds=${ds}"
                echo "lam=${lam}"
                echo "norm=${norm}"
                echo "thresh=${thresh}"
                echo "******************"

                dataset=VOC2007; python extract.py extract_eigs \
                --images_root "./data/${dataset}/images" \
                --features_dir "./data/${dataset}/features/${model}/" \
                --which_matrix "affinity" \
                --output_dir "./data/${dataset}/eigs/affinity_${model}_${ds}x_lambda_${lam}_${norm}_${thresh}" \
                --image_downsample_factor ${ds} \
                --image_color_lambda ${lam} \
                --K 2 \
                --${norm} --${thresh}

                done
            done
        done
    done
done

