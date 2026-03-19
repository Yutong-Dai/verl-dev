# Enable error handling
set -e

PROJECT_DIR=/fsx/home/yutong/Github/verl
echo "PROJECT_DIR: $PROJECT_DIR"

# Set up Conda
source /fsx/home/yutong/miniconda3/bin/activate  
conda activate verl|| { echo "Failed to activate Conda environment"; exit 1; }


# Print environment information for debugging
echo "Python path: $(which python)"
echo "Conda environment: $CONDA_PREFIX"

cd $PROJECT_DIR

source $PROJECT_DIR/.env
if [ "$USE_SF_WANDB" = "True" ]; then
    wandb login --relogin --host=https://salesforceairesearch.wandb.io $WANDB_API_KEY_SF || { echo "Failed to login to wandb"; exit 1; }
else
    export WANDB_API_KEY="$WANDB_API_KEY_PERSONAL"
    wandb login --relogin $WANDB_API_KEY_PERSONAL || { echo "Failed to login to wandb"; exit 1; }
fi
export EXPERIMENT_NAME=GSM8K-GRPO-$(date +%Y-%m-%d-%H-%M-%S)
export WANDB_PROJECT=verl-tutorial
export LOG_FILENAME=logs/$EXPERIMENT_NAME.log
export LOG_MODE="[wandb,console]"
# export LOG_MODE="[console]" 
export SAVE_CONTENTS="[model,optimizer,extra]"
# export SAVE_CONTENTS=['model']

MODEL_PATH=/fsx/sfr/data/yutong/ckpt/Qwen2.5-0.5B-Instruct
DATA_PATH=/fsx/home/yutong

# VLLM save memory related settings
# if want to save memory, set ENFORCE_EAGER=True and FREE_CACHE_ENGINE=True
ENFORCE_EAGER=False
FREE_CACHE_ENGINE=True

python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_PATH/data/gsm8k/train.parquet \
    data.val_files=$DATA_PATH/data/gsm8k/test.parquet \
    data.train_batch_size=1024 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=512 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents=$SAVE_CONTENTS \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=48 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=48 \
    actor_rollout_ref.rollout.enforce_eager=$ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.n=8 \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    trainer.total_epochs=15 \
    trainer.logger=$LOG_MODE \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.val_before_train=True
