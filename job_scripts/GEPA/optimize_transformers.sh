source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    CancerMyth
    QA2FPQ
    SynQA2FPQ
    CREPEFPQ
)
MODELS=(
    gemma-4-E4B-it
)
THINKING_MODELS=(
    gemma-4-E4B-it
)
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        echo "Optimizing direct QA for ${MODEL} on ${DATASET}..."
        python -m GEPA.optimize_direct_qa \
            transformers \
                --model_name ${HF_HOME}/${MODEL} \
                --dataset ${DATASET} \
                --train_path ${SCRATCH_DIR}/datasets/${DATASET}/train.jsonl \
                --val_path ${SCRATCH_DIR}/datasets/${DATASET}/dev.jsonl \
                --out_dir ${SCRATCH_DIR}/FP_Hallucination/GEPA/${DATASET}/${MODEL} \
                --val_limit 50 \
                no_passages
        echo "Finished optimizing direct QA for ${MODEL} on ${DATASET}"
    done

    for MODEL in "${THINKING_MODELS[@]}"; do
        echo "Optimizing direct QA with thinking for ${MODEL} on ${DATASET}..."
        python -m GEPA.optimize_direct_qa \
            transformers \
                --model_name ${HF_HOME}/${MODEL} \
                --dataset ${DATASET} \
                --train_path ${SCRATCH_DIR}/datasets/${DATASET}/train.jsonl \
                --val_path ${SCRATCH_DIR}/datasets/${DATASET}/dev.jsonl \
                --out_dir ${SCRATCH_DIR}/FP_Hallucination/GEPA/${DATASET}/${MODEL}_thinking \
                --enable_thinking \
                --val_limit 50 \
                no_passages
        echo "Finished optimizing direct QA with thinking for ${MODEL} on ${DATASET}"
    done
done
