## Session name
NAME=launch_lost

## Create tmux session
tmux new-session -d -s ${NAME}

## Create the windows on which each node or .launch file is going to run
tmux send-keys -t ${NAME} 'tmux new-window -n WIN0 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN1 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN2 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN3 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN4 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN5 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN6 ' ENTER
tmux send-keys -t ${NAME} 'tmux new-window -n WIN7 ' ENTER

## Send commands to each window
tmux send-keys -t ${NAME} "tmux send-keys -t WIN0 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=./dino:$PYTHONPATH python new_lost.py --dataset VOC07 --set trainval --eigenseg  # VIT-S/16\n' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN1 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./dino:$PYTHONPATH python new_lost.py --dataset VOC07 --set trainval --eigenseg  --patch_size 8  # VIT-S/8' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN2 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 PYTHONPATH=./dino:$PYTHONPATH python new_lost.py --dataset VOC07 --set trainval --eigenseg  --arch vit_base  # VIT-B/8\n' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN3 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 PYTHONPATH=./dino:$PYTHONPATH python new_lost.py --dataset VOC07 --set trainval --eigenseg  --arch resnet50  # ResNet50-DINO\n' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN4 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 PYTHONPATH=./dino:$PYTHONPATH python new_lost.py --dataset VOC12 --set trainval --eigenseg  # VIT-S/16\n' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN5 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5 PYTHONPATH=./dino:$PYTHONPATH python new_lost.py --dataset VOC12 --set trainval --eigenseg  --patch_size 8  # VIT-S/8\n' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN6 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=6 PYTHONPATH=./dino:$PYTHONPATH python new_lost.py --dataset VOC12 --set trainval --eigenseg  --arch vit_base  # VIT-B/8\n' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN7 'OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 PYTHONPATH=./dino:$PYTHONPATH python new_lost.py --dataset VOC12 --set trainval --eigenseg  --arch resnet50  # ResNet50-DINO\n' ENTER" ENTER

## Start a new line on window 0
tmux send-keys -t ${NAME} ENTER

## Attach to session
# tmux send-keys -t ${NAME} "tmux select-window -t 1" ENTER
# tmux send-keys -t ${NAME} "tmux send-keys 'nvidia-smi -l 60' ENTER" ENTER
# tmux attach -t ${NAME}
echo "Launched ${NAME}"
