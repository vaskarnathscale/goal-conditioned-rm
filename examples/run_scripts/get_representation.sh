set -x

python ./examples/get_representation.py \
    --dataset-path /mnt/efs/vaskarnath/workspace/research_code/examples/data/paired_openmathinstruct-1-masked-extra-drop \
    --reward-model-path ./examples/reward_model \
    --save-dir ./examples/data/representation