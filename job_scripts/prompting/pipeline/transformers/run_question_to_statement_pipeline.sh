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
            echo "Processing question-to-statement pipeline with ${MODEL} (RAG=${RAG}, factchecker=${FACTCHECKER})"
            for DATASET in "${DATASETS[@]}"; do
                DECOMPOSITION_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Question_To_Statement/${MODEL}"
                PRESUPPOSITION_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Presupposition_Extraction/${MODEL}"
                KNOWLEDGE_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Knowledge_Generation/${MODEL}"
                FACTCHECK_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Presupposition_Factcheck/${MODEL}"
                FINAL_OUTPUT_DIR="${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Final_Response/${MODEL}"

                QUESTION_DECOMPOSED_FILE="${DECOMPOSITION_OUTPUT_DIR}/RAG=0.jsonl"
                PRESUPPOSITION_FILE="${PRESUPPOSITION_OUTPUT_DIR}/RAG=0.jsonl"
                KNOWLEDGE_FILE="${KNOWLEDGE_OUTPUT_DIR}/RAG=0.jsonl"
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

                python -m prompting.run_question_to_statement_pipeline \
                    --backend transformers \
                    --model_name "${MODEL_NAME}" \
                    --dataset_path "${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl" \
                    --question_decomposed_file "${QUESTION_DECOMPOSED_FILE}" \
                    --presupposition_file "${PRESUPPOSITION_FILE}" \
                    --knowledge_file "${KNOWLEDGE_FILE}" \
                    --factchecked_file "${FACTCHECKED_FILE}" \
                    --question_decomposition_output_dir "${DECOMPOSITION_OUTPUT_DIR}" \
                    --presupposition_output_dir "${PRESUPPOSITION_OUTPUT_DIR}" \
                    --knowledge_output_dir "${KNOWLEDGE_OUTPUT_DIR}" \
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
