### Usage:
### >>> wandb sweep sweeps/sweep.yaml
### >>> ./sweeps/launch.sh

## Session name
NAME=sweep6
COMMAND="wandb agent lukemelas2/found/${1}"

## Print
echo "Launching sweep ${1} on tmux session ${NAME}"

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
tmux send-keys -t ${NAME} "tmux send-keys -t WIN0 'CUDA_VISIBLE_DEVICES=0 $COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN1 'CUDA_VISIBLE_DEVICES=1 $COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN2 'CUDA_VISIBLE_DEVICES=2 $COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN3 'CUDA_VISIBLE_DEVICES=3 $COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN4 'CUDA_VISIBLE_DEVICES=4 $COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN5 'CUDA_VISIBLE_DEVICES=5 $COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN6 'CUDA_VISIBLE_DEVICES=6 $COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN7 'CUDA_VISIBLE_DEVICES=7 $COMMAND' ENTER" ENTER

## Start a new line on window 0
tmux send-keys -t ${NAME} ENTER

echo "Launched ${NAME}"
