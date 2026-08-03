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
MODELS=(
    Meta-Llama-3-8B-Instruct
    Qwen2.5-7B-Instruct
    gemma-4-E4B-it
    Olmo-3-7B-Instruct
)
RAG_CONDITIONS=(0 all 4)
THINKING=false
BATCHING=true
BATCH_SIZE=4

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME="${HF_HOME}/${MODEL}"

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
                    SUFFIX="transformers_checked"
                else
                    SUFFIX="minichecked"
                fi
                FACTCHECKED_FILE="${FACTCHECK_OUTPUT_DIR}/RAG=${RAG}_${SUFFIX}.jsonl"
                FINAL_OUTPUT_FILE="${FINAL_OUTPUT_DIR}/RAG=${RAG}_${SUFFIX}.jsonl"

                if [[ -f "${FINAL_OUTPUT_FILE}" ]]; then
                    echo "Skipping ${DATASET}: final output exists at ${FINAL_OUTPUT_FILE}"
                    continue
                fi

                python -m prompting.run_presupposition_pipeline \
                    --backend transformers \
                    --model_name "${MODEL_NAME}" \
                    --dataset_path "${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl" \
                    --presupposition_file "${PRESUPPOSITION_FILE}" \
                    --factchecked_file "${FACTCHECKED_FILE}" \
                    --presupposition_output_dir "${PRESUPPOSITION_OUTPUT_DIR}" \
                    --factcheck_output_dir "${FACTCHECK_OUTPUT_DIR}" \
                    --output_dir "${FINAL_OUTPUT_DIR}" \
                    --RAG "${RAG}" \
                    --factchecker "${FACTCHECKER}" \
                    --thinking "${THINKING}" \
                    --batching "${BATCHING}" \
                    --batch_size "${BATCH_SIZE}"
            done
        done
    done
done
