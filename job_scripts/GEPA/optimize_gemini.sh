source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

MODEL=gemini-3-flash-preview
DATASETS=(
    CancerMyth
    QA2FPQ
    SynQA2FPQ
    CREPEFPQ
)
for DATASET in "${DATASETS[@]}"; do
    echo "Optimizing direct QA for ${MODEL} on ${DATASET}..."
    python -m GEPA.optimize_direct_qa \
        gemini \
            --model_name ${MODEL} \
            --dataset ${DATASET} \
            --train_path ${SCRATCH_DIR}/datasets/${DATASET}/train.jsonl \
            --val_path ${SCRATCH_DIR}/datasets/${DATASET}/dev.jsonl \
            --out_dir ${SCRATCH_DIR}/FP_Hallucination/GEPA/${DATASET}/${MODEL} \
            --val_limit 50 \
            no_passages
    echo "Finished optimizing direct QA for ${MODEL} on ${DATASET}"
done