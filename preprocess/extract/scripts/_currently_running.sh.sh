# Laplacians (continued)
for ds in 8  # 16  # 8 16
do 
    for model in mocov3_vitb16 mocov3_vits16 dino_vitb16 dino_vits16  # mocov3_vits16 mocov3_vitb16 dino_vits16 dino_vitb16 dino_vits8 dino_vitb8
    do 
        for norm in normalize  # nonormalize 
        do 
            for thresh in threshold_at_zero  # nothreshold_at_zero 
            do 
                for lapnorm in lapnorm  # nolapnorm 
                do 
                    for lam in 8.0        #  20.0 50.0 100.0 0.0 1.0 2.0 10.0
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


# Laplacians on COCO
for dataset in COCO  # VOC2012
do
    for ds in 8  # 16  # 8 16
    do 
        for model in dino_vits8  # mocov3_vits16 mocov3_vitb16 dino_vits16 dino_vitb16 dino_vits8 dino_vitb8
        do 
            for norm in normalize  # nonormalize 
            do 
                for thresh in threshold_at_zero  # nothreshold_at_zero 
                do 
                    for lapnorm in lapnorm  # nolapnorm 
                    do 
                        for lam in 10.0    #  20.0 50.0 100.0 0.0 1.0 2.0 10.0
                        do
                            echo "******************"
                            echo "model=${model}"
                            echo "ds=${ds}"
                            echo "lam=${lam}"
                            echo "******************"

                            python extract.py extract_eigs \
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
done
