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
        echo "Processing FP identification with ${MODEL} (RAG=${RAG})"
        for DATASET in "${DATASETS[@]}"; do
            IDENTIFICATION_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FP_Identification/${MODEL}"
            FINAL_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FP_Identification_Final_Response/${MODEL}"
            FINAL_OUTPUT_FILE="${FINAL_OUTPUT_DIR}/RAG=${RAG}.jsonl"

            if [[ -f "${FINAL_OUTPUT_FILE}" ]]; then
                echo "Skipping ${DATASET}: final output exists at ${FINAL_OUTPUT_FILE}"
                continue
            fi

            python -m prompting.run_fp_identification_pipeline \
                --backend transformers \
                --model_name "${MODEL_NAME}" \
                --dataset_path "${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl" \
                --identification_output_dir "${IDENTIFICATION_OUTPUT_DIR}" \
                --output_dir "${FINAL_OUTPUT_DIR}" \
                --RAG "${RAG}" \
                --thinking "${THINKING}" \
                --batching "${BATCHING}" \
                --batch_size "${BATCH_SIZE}"
        done
    done
done
