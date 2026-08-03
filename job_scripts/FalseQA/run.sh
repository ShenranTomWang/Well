source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    CancerMyth/CancerMyth
    CancerMyth/CancerMythNFP
    CREPE/CREPEFPQ
    CREPE/CREPETPQ
)
RAG_CONDITIONS=(0 all 4)

for RAG in "${RAG_CONDITIONS[@]}"; do
    for DATASET_PAIR in "${DATASETS[@]}"; do
        IFS='/' read -r -a DATASET_ARRAY <<< "${DATASET_PAIR}"
        DATASET="${DATASET_ARRAY[0]}"
        DATA_SPLIT="${DATASET_ARRAY[1]}"
        MODEL_PATH=${SCRATCH_DIR}/FP_Hallucination/train/${DATASET}/Qwen/Qwen2.5-7B-Instruct/output_adapter
        echo "Running FalseQA Qwen2.5-7B-Instruct on ${DATA_SPLIT} with RAG=${RAG}"
        python -m FalseQA.run \
            --model_path "${MODEL_PATH}" \
            --dataset_path "${SCRATCH_DIR}/datasets/${DATA_SPLIT}/test.jsonl" \
            --output_dir "${SCRATCH_DIR}/FP_Hallucination/out/${DATA_SPLIT}/FalseQA/Qwen2.5-7B-Instruct" \
            --RAG "${RAG}" \
            --batch_size 4
    done
done
