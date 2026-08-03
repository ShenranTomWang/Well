source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

HEADS_ROOT=${HEADS_ROOT:-${SCRATCH_DIR}/FP_Hallucination/FAITH/heads}
OUT_ROOT=${OUT_ROOT:-${SCRATCH_DIR}/FP_Hallucination/out}
SOURCE_COMMAND=${SOURCE_COMMAND:-no_passages}

DATASETS=(
    CancerMyth
    CancerMythNFP
    QA2FPQ
    QA2TPQ
    SynQA2FPQ
    SynQA2TPQ
    CREPEFPQ
    CREPETPQ
)
MODELS=(
    Meta-Llama-3-8B-Instruct
    Qwen2.5-7B-Instruct
    gemma-4-E4B-it
    Olmo-3-7B-Instruct
)

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "Running ${MODEL} on ${DATASET} test set with movie heads knocked out..."
        python -m FAITH.knock_out_direct_qa_top_heads \
            --model-name-or-path ${HF_HOME}/${MODEL} \
            --dataset-path ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --heads-dir ${HEADS_ROOT}/${MODEL} \
            --out-file ${OUT_ROOT}/${DATASET}/FAITH/${MODEL}/RAG=0.jsonl \
            --disable-few-shot \
            --batch_size 12 \
            ${SOURCE_COMMAND}
        echo "Finished..."
    done
done
