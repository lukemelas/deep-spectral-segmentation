### Usage:
### >>> wandb sweep sweeps/sweep.yaml
### >>> ./sweeps/launch.sh

## Session name
NAME=sweep
COMMAND="wandb agent lukemelas2/sequential_interpret/0h05dtsd"

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
tmux send-keys -t ${NAME} "tmux send-keys -t WIN0 '$COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN1 '$COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN2 '$COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN3 '$COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN4 '$COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN5 '$COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN6 '$COMMAND' ENTER" ENTER
tmux send-keys -t ${NAME} "tmux send-keys -t WIN7 '$COMMAND' ENTER" ENTER

## Start a new line on window 0
tmux send-keys -t ${NAME} ENTER

## Attach to session
# tmux send-keys -t ${NAME} "tmux select-window -t 1" ENTER
# tmux send-keys -t ${NAME} "tmux send-keys 'nvidia-smi -l 60' ENTER" ENTER
# tmux attach -t ${NAME}
tmux switch -t ${NAME}
