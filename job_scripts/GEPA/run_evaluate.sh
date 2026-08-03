source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    CancerMyth
    CancerMythNFP
    QA2FPQ
    QA2TPQ
    SynQA2FPQ
    SynQA2TPQ
    CREPEFPQ
    CREPETPQ
)

MODELS=(
    gemini-3-flash-preview
    gemma-4-E4B-it
)
THINKING_MODELS=(
    gemma-4-E4B-it
)
RAG_CONFIGS=(
    0
    all
    4
    web
)
for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for RAG_CONFIG in "${RAG_CONFIGS[@]}"; do
            FILE=${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA_GEPA/${MODEL}/RAG=${RAG_CONFIG}.jsonl
            EVALUATED_FILE=${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA_GEPA/${MODEL}/RAG=${RAG_CONFIG}_response_level_score_evaluated.jsonl

            if [[ -f "${EVALUATED_FILE}" ]]; then
                echo "Skipping ${DATASET}: final output exists at ${EVALUATED_FILE}"
                continue
            fi

            echo "Submitting response level score eval for GEPA direct QA with ${MODEL} on ${DATASET} (RAG=${RAG_CONFIG})"
            python run_response_level_score.py \
                response_level_score_submit \
                    --file ${FILE} \
                    --dataset ${DATASET}
            echo "Finished ${DATASET} with ${MODEL} (RAG=${RAG_CONFIG})"
        done
    done

    for MODEL in "${THINKING_MODELS[@]}"; do
        for RAG_CONFIG in "${RAG_CONFIGS[@]}"; do
            FILE=${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA_GEPA/${MODEL}/RAG=${RAG_CONFIG}_thinking.jsonl
            EVALUATED_FILE=${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA_GEPA/${MODEL}/RAG=${RAG_CONFIG}_thinking_response_level_score_evaluated.jsonl

            if [[ -f "${EVALUATED_FILE}" ]]; then
                echo "Skipping ${DATASET}: final output exists at ${EVALUATED_FILE}"
                continue
            fi

            echo "Submitting response level score eval for GEPA direct QA with ${MODEL} on ${DATASET} (RAG=${RAG_CONFIG}, thinking)"
            python run_response_level_score.py \
                response_level_score_submit \
                    --file ${FILE} \
                    --dataset ${DATASET}
            echo "Finished ${DATASET} with ${MODEL} (RAG=${RAG_CONFIG}, thinking)"
        done
    done
done
