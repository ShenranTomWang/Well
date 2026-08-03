source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    SynQA2FPQ/0
    SynQA2TPQ/1
    CancerMyth/0
    CancerMythNFP/1
)

for COMBINATION in "${DATASETS[@]}"; do
    IFS='/' read -r -a parts <<< "$COMBINATION"
    DATASET=${parts[0]}
    EXPECTED_RESULT=${parts[1]}
    echo "Checking ${DATASET} with top-4 RAG passages (mini check)"
    python -m prompting.run_fact_check \
        minicheck \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/minicheck/RAG=4.jsonl \
            --check_gold \
            use_RAG

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/minicheck/RAG=4.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with top-4 RAG passages (mini check)"

    echo "Checking ${DATASET} with all passages (mini check)"
    python -m prompting.run_fact_check \
        minicheck \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/minicheck/RAG=all.jsonl \
            --check_gold \
            use_passages

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/minicheck/RAG=all.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with all passages (mini check)"
done