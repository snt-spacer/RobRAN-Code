#!/bin/bash

ROBOTS=(
    # "Jetbot"
    # "Leatherback"
    # "FloatingPlatform"
    "Turtlebot2"
    # "Kingfisher"
)

for seed in {0..9}
do
    for robot in "${ROBOTS[@]}"
    do
        echo "Training with robot: $robot and seed: $seed"

        # Set algorithm based on robot type
        if [ "$robot" = "FloatingPlatform" ]; then
            algorithm="ppo-discrete"
        else
            algorithm="ppo"
        fi

        ./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
        --task Isaac-RANS-Single-v0 \
        env.task_name=GoToPositionWithObstacles \
        env.robot_name=$robot \
        --num_envs 4096 \
        --headless \
        --algorithm=$algorithm \
        --seed $seed
    done
done
