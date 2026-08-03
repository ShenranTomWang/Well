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
BATCHING=true
BATCH_SIZE=4

for MODEL in "${MODELS[@]}"; do
    MODEL_NAME="${HF_HOME}/${MODEL}"

    THINKING_CONDITIONS=(false)
    if [[ "${MODEL}" == "gemma-4-E4B-it" ]]; then
        THINKING_CONDITIONS+=(true)
    fi

    for THINKING in "${THINKING_CONDITIONS[@]}"; do
        OUTPUT_MODEL="${MODEL}"
        if [[ "${THINKING}" == "true" ]]; then
            OUTPUT_MODEL="${MODEL}_thinking"
        fi
        for RAG in "${RAG_CONDITIONS[@]}"; do
            echo "Processing direct QA with ${MODEL} (RAG=${RAG}, thinking=${THINKING})"
            for DATASET in "${DATASETS[@]}"; do
                OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA/${OUTPUT_MODEL}"
                FINAL_OUTPUT_FILE="${OUTPUT_DIR}/RAG=${RAG}.jsonl"

                if [[ -f "${FINAL_OUTPUT_FILE}" ]]; then
                    echo "Skipping ${DATASET}: final output exists at ${FINAL_OUTPUT_FILE}"
                    continue
                fi

                python -m prompting.run_direct_qa_pipeline \
                    --backend transformers \
                    --model_name "${MODEL_NAME}" \
                    --dataset_path "${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl" \
                    --output_dir "${OUTPUT_DIR}" \
                    --RAG "${RAG}" \
                    --thinking "${THINKING}" \
                    --batching "${BATCHING}" \
                    --batch_size "${BATCH_SIZE}"
            done
        done
    done
done
