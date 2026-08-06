#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=72:0:0
#SBATCH --partition=nlpgpo

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
MODELS=(google/gemma-4-E4B-it)

run_direct_qa() {
    local dataset_name=$1
    local model=$2
    local prompt_file=$3
    local rag=$4
    local thinking=$5
    local model_name=${model##*/}
    local output_suffix=""
    local thinking_args=()
    local source_args=()
    if [[ ${thinking} == true ]]; then
        output_suffix="_thinking"
        thinking_args=(--enable_thinking)
    fi
    case ${rag} in
        0) source_args=(no_passages) ;;
        all) source_args=(use_passages) ;;
        *) source_args=(use_RAG --k "${rag}") ;;
    esac

    echo "Running transformers direct QA (RAG=${rag}, thinking=${thinking}) for ${dataset_name} using ${model}"
    python -m prompting.run_direct_qa \
        transformers \
        --model_name ${model} \
        --dataset_path ${SCRATCH_DIR}/datasets/${dataset_name}/test.jsonl \
        --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA_balanced/${model_name}/RAG=${rag}${output_suffix}.jsonl \
        --system_prompt_file ${prompt_file} \
        --device cuda:0 \
        "${thinking_args[@]}" \
        "${source_args[@]}"
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
