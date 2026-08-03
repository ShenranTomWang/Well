source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    CancerMyth/CancerMyth/CancerMythNFP
    QA2/QA2FPQ/QA2TPQ
    SynQA2/SynQA2FPQ/SynQA2TPQ
    CREPE/CREPEFPQ/CREPETPQ
)
MODELS=(
    gemma-4-E4B-it
)
THINKING_MODELS=(
    gemma-4-E4B-it
)
for DATASET in "${DATASETS[@]}"; do
    IFS='/' read -r DATASET_FAMILY FPQ_DATASET TPQ_DATASET <<< "${DATASET}"
    for MODEL in "${MODELS[@]}"; do
        echo "Optimizing direct QA for ${MODEL} on ${DATASET}..."
        python -m GEPA.optimize_direct_qa_balanced \
            transformers \
                --model_name ${HF_HOME}/${MODEL} \
                --dataset_family ${DATASET_FAMILY} \
                --fpq_train_path ${SCRATCH_DIR}/datasets/${FPQ_DATASET}/train.jsonl \
                --tpq_train_path ${SCRATCH_DIR}/datasets/${TPQ_DATASET}/train.jsonl \
                --fpq_val_path ${SCRATCH_DIR}/datasets/${FPQ_DATASET}/dev.jsonl \
                --tpq_val_path ${SCRATCH_DIR}/datasets/${TPQ_DATASET}/dev.jsonl \
                --fpq_val_limit 50 \
                --tpq_val_limit 50 \
                --out_dir ${SCRATCH_DIR}/FP_Hallucination/GEPA_balanced/${DATASET_FAMILY}/${MODEL} \
                no_passages
        echo "Finished optimizing direct QA for ${MODEL} on ${DATASET}"
    done

    for MODEL in "${THINKING_MODELS[@]}"; do
        echo "Optimizing direct QA with thinking for ${MODEL} on ${DATASET}..."
        python -m GEPA.optimize_direct_qa_balanced \
            transformers \
                --model_name ${HF_HOME}/${MODEL} \
                --dataset_family ${DATASET_FAMILY} \
                --fpq_train_path ${SCRATCH_DIR}/datasets/${FPQ_DATASET}/train.jsonl \
                --tpq_train_path ${SCRATCH_DIR}/datasets/${TPQ_DATASET}/train.jsonl \
                --fpq_val_path ${SCRATCH_DIR}/datasets/${FPQ_DATASET}/dev.jsonl \
                --tpq_val_path ${SCRATCH_DIR}/datasets/${TPQ_DATASET}/dev.jsonl \
                --fpq_val_limit 50 \
                --tpq_val_limit 50 \
                --out_dir ${SCRATCH_DIR}/FP_Hallucination/GEPA_balanced/${DATASET_FAMILY}/${MODEL}_thinking \
                --enable_thinking \
                no_passages
        echo "Finished optimizing direct QA with thinking for ${MODEL} on ${DATASET}"
    done
done
