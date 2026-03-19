set -e

# Set up Conda
source /fsx/home/yutong/miniconda3/bin/activate  
conda activate verl|| { echo "Failed to activate Conda environment"; exit 1; }

CKPT_DIR=/fsx/home/yutong/Github/verl/checkpoints/verl-tutorial/GSM8K-GRPO-03-16-20-16/global_step_105/actor
OUTPUT_DIR=/fsx/home/yutong/Github/verl/checkpoints/merged_ckpts/verl-tutorial/GSM8K-GRPO-03-16-20-16_global_step_105

cd /fsx/home/yutong/Github/verl

echo "Merging FSDP checkpoint from $CKPT_DIR"
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $CKPT_DIR \
    --target_dir $OUTPUT_DIR

echo "Merged checkpoint saved to $OUTPUT_DIR"