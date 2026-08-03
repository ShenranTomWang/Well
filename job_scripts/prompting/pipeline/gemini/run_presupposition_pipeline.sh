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
    FACTCHECKERS=(llm)
    if [[ "${RAG}" == "all" || "${RAG}" == "4" ]]; then
        FACTCHECKERS+=(minicheck)
    fi

    for FACTCHECKER in "${FACTCHECKERS[@]}"; do
        echo "Processing presupposition pipeline with ${MODEL} (RAG=${RAG}, factchecker=${FACTCHECKER})"
        for DATASET in "${DATASETS[@]}"; do
            PRESUPPOSITION_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Presupposition_Extraction/${MODEL}"
            FACTCHECK_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Presupposition_Extraction/${MODEL}"
            FINAL_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Final_Response/${MODEL}"
            PRESUPPOSITION_FILE="${PRESUPPOSITION_OUTPUT_DIR}/RAG=0.jsonl"
            if [[ "${FACTCHECKER}" == "llm" ]]; then
                SUFFIX="gemini_checked"
            else
                SUFFIX="minichecked"
            fi
            FACTCHECKED_FILE="${FACTCHECK_OUTPUT_DIR}/RAG=${RAG}_${SUFFIX}.jsonl"
            FINAL_RESPONSE_FILE="${FINAL_OUTPUT_DIR}/RAG=${RAG}_${SUFFIX}.jsonl"

            if [[ -f "${FINAL_RESPONSE_FILE}" ]]; then
                echo "Skipping ${DATASET}: final response exists at ${FINAL_RESPONSE_FILE}"
                continue
            fi

            python -m prompting.run_presupposition_pipeline \
                --backend gemini \
                --model_name "${MODEL}" \
                --dataset_path "${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl" \
                --presupposition_file "${PRESUPPOSITION_FILE}" \
                --factchecked_file "${FACTCHECKED_FILE}" \
                --presupposition_output_dir "${PRESUPPOSITION_OUTPUT_DIR}" \
                --factcheck_output_dir "${FACTCHECK_OUTPUT_DIR}" \
                --output_dir "${FINAL_OUTPUT_DIR}" \
                --RAG "${RAG}" \
                --factchecker "${FACTCHECKER}" \
                --thinking "${THINKING}" \
                --batching "${BATCHING}"
        done
    done
done
