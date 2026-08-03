source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASETS=(
    CancerMyth/CancerMyth
)
MODELS=(
    Qwen/Qwen2.5-7B-Instruct
)

for DATASET in "${DATASETS[@]}"; do
    IFS='/' read -r -a DATASET_PARTS <<< "${DATASET}"
    DATASET_NAME=${DATASET_PARTS[0]}
    FPQ_NAME=${DATASET_PARTS[1]}
    FPQ_PATH=${SCRATCH_DIR}/datasets/${FPQ_NAME}/train.jsonl
    for MODEL in "${MODELS[@]}"; do
        echo "Training ${MODEL} on ${DATASET}..."
        torchrun \
            --standalone \
            --nproc_per_node=6 \
            FalseQA/train_arc_da.py \
                --model_name_or_path "${MODEL}" \
                --fpq_path "${FPQ_PATH}" \
                --output_dir "${SCRATCH_DIR}/FP_Hallucination/train/${DATASET_NAME}/${MODEL}/output_adapter" \
                --use_chat_template \
                --bf16 \
                --per_device_train_batch_size 1 \
                --gradient_accumulation_steps 4 \
                --gradient_checkpointing
        echo "Finished training ${MODEL} on ${DATASET}"
    done
done