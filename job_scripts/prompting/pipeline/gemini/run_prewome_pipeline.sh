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
    echo "Processing PreWoMe with ${MODEL} (RAG=${RAG})"
    for DATASET in "${DATASETS[@]}"; do
        PRESUPPOSITION_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Presupposition_Extraction/${MODEL}"
        FEEDBACK_ACTION_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Feedback_Action/${MODEL}"
        FINAL_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Feedback_Action_Final_Response/${MODEL}"
        PRESUPPOSITION_FILE="${PRESUPPOSITION_OUTPUT_DIR}/presuppositions.jsonl"
        FEEDBACK_ACTION_FILE="${FEEDBACK_ACTION_OUTPUT_DIR}/RAG=${RAG}_gemini_checked.jsonl"
        FINAL_RESPONSE_FILE="${FINAL_OUTPUT_DIR}/RAG=${RAG}_gemini_checked.jsonl"

        if [[ -f "${FINAL_RESPONSE_FILE}" ]]; then
            echo "Skipping ${DATASET}: final response exists at ${FINAL_RESPONSE_FILE}"
            continue
        fi

        python -m prompting.run_prewome \
            --backend gemini \
            --model_name "${MODEL}" \
            --dataset_path "${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl" \
            --presupposition_file "${PRESUPPOSITION_FILE}" \
            --feedback_action_file "${FEEDBACK_ACTION_FILE}" \
            --presupposition_output_dir "${PRESUPPOSITION_OUTPUT_DIR}" \
            --feedback_action_output_dir "${FEEDBACK_ACTION_OUTPUT_DIR}" \
            --output_dir "${FINAL_OUTPUT_DIR}" \
            --RAG "${RAG}" \
            --thinking "${THINKING}" \
            --batching "${BATCHING}"
    done
done
