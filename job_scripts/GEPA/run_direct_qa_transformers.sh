source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    CancerMyth/CancerMyth
    CancerMythNFP/CancerMyth
    QA2FPQ/QA2FPQ
    QA2TPQ/QA2FPQ
    SynQA2FPQ/SynQA2FPQ
    SynQA2TPQ/SynQA2FPQ
    CREPEFPQ/CREPEFPQ
    CREPETPQ/CREPEFPQ
)
GEPA_ROOT="${SCRATCH_DIR}/FP_Hallucination/GEPA"
MODELS=(gemma-4-E4B-it)

run_direct_qa() {
    local dataset_name=$1
    local model=$2
    local prompt_file=$3
    local rag=$4
    local thinking=$5
    local output_model=$model
    if [[ ${thinking} == true ]]; then
        output_model=${model}_thinking
    fi

    echo "Running transformers direct QA (RAG=${rag}, thinking=${thinking}) for ${dataset_name} using ${model}"
    python -m prompting.run_direct_qa_pipeline \
        --backend transformers \
        --model_name ${HF_HOME}/${model} \
        --dataset_path ${SCRATCH_DIR}/datasets/${dataset_name}/test.jsonl \
        --output_dir ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA/${output_model} \
        --system_prompt_file ${prompt_file} \
        --device cuda:0 \
        --RAG ${rag} \
        --thinking ${thinking} \
        --batching true \
        --batch_size 12
}

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        IFS='/' read -ra DATASET_SPLIT <<< "${DATASET}"
        dataset_name=${DATASET_SPLIT[0]}
        GEPA_prompt_src=${DATASET_SPLIT[1]}

        PROMPT_FILE="${GEPA_ROOT}/${GEPA_prompt_src}/${MODEL}/best_system_prompt.txt"
        for RAG in 0 all 4; do
            run_direct_qa "${dataset_name}" "${MODEL}" "${PROMPT_FILE}" "${RAG}" false
        done

        PROMPT_FILE="${GEPA_ROOT}/${GEPA_prompt_src}/${MODEL}_thinking/best_system_prompt.txt"
        for RAG in 0 all 4; do
            run_direct_qa "${dataset_name}" "${MODEL}" "${PROMPT_FILE}" "${RAG}" true
        done
    done
done
