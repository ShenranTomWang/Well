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
    echo "Processing FP identification with ${MODEL} (RAG=${RAG})"
    for DATASET in "${DATASETS[@]}"; do
        IDENTIFICATION_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FP_Identification/${MODEL}"
        FINAL_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FP_Identification_Final_Response/${MODEL}"

        python -m prompting.run_fp_identification_pipeline \
            --backend gemini \
            --model_name "${MODEL}" \
            --dataset_path "${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl" \
            --identification_output_dir "${IDENTIFICATION_OUTPUT_DIR}" \
            --output_dir "${FINAL_OUTPUT_DIR}" \
            --RAG "${RAG}" \
            --thinking "${THINKING}" \
            --batching "${BATCHING}"
    done
done
