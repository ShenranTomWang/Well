source ${HOME_DIR}/.bashrc
source "${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate"
cd "${PROJECT_DIR}/FP_Hallucination"

DATASETS=(
    SynQA2FPQ
    SynQA2TPQ
    QA2FPQ
    QA2TPQ
    CancerMyth
    CancerMythNFP
    CREPEFPQ
    CREPETPQ
)
MODEL=gemini-3-flash-preview
RAG_CONDITIONS=(0 all 4 web)
THINKING=false
BATCHING=true

for RAG in "${RAG_CONDITIONS[@]}"; do
    echo "Processing direct QA with ${MODEL} (RAG=${RAG})"
    for DATASET in "${DATASETS[@]}"; do
        python -m prompting.run_direct_qa_pipeline \
            --backend gemini \
            --model_name "${MODEL}" \
            --dataset_path "${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl" \
            --output_dir "${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA/${MODEL}" \
            --RAG "${RAG}" \
            --thinking "${THINKING}" \
            --batching "${BATCHING}"
    done
done
