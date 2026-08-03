source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    CREPEFPQ
    CREPETPQ
    CancerMyth
    CancerMythNFP
)
MODEL=Qwen2.5-7B-Instruct
RAG_CONDITIONS=(0 all 4)

for DATASET in "${DATASETS[@]}"; do
    for RAG in "${RAG_CONDITIONS[@]}"; do
        FILE=${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FalseQA/${MODEL}/RAG=${RAG}.jsonl
        echo "Submitting batched response-level evaluation for ${DATASET}, ${MODEL}, RAG=${RAG}"
        python run_response_level_score.py \
            response_level_score_submit \
                --file "${FILE}" \
                --dataset "${DATASET}"
    done
done
