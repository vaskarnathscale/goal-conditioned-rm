set -x

python ./examples/get_representation.py \
    --dataset-path /mnt/efs/vaskarnath/workspace/research_code/examples/data/preference_ranking_dataset \
    --reward-model-path ./ckpt/contrastive_reward_model \
    --save-dir ./examples/data/representation