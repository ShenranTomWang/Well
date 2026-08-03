source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

HEADS_ROOT=${HEADS_ROOT:-${SCRATCH_DIR}/FP_Hallucination/FAITH/heads}
OUT_ROOT=${OUT_ROOT:-${SCRATCH_DIR}/FP_Hallucination/out}

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
        echo "Evaluating ${MODEL} on ${DATASET} test set with movie heads knocked out..."
        python run_response_level_score.py \
            response_level_score_submit \
                --file ${OUT_ROOT}/${DATASET}/FAITH/${MODEL}/RAG=0.jsonl \
                --dataset ${DATASET}
        
        python run_response_level_score.py \
            print_results \
                --file ${OUT_ROOT}/${DATASET}/FAITH/${MODEL}/RAG=0_response_level_score_evaluated.jsonl
        echo "Finished..."
    done
done
