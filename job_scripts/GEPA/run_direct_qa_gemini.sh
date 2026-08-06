#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=72:0:0
#SBATCH --partition=ubcml-nlp

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

MODEL=gemini-3-flash-preview
GEPA_ROOT="${SCRATCH_DIR}/FP_Hallucination/GEPA"

for DATASET in "${DATASETS[@]}"; do
    IFS='/' read -ra DATASET_SPLIT <<< "${DATASET}"
    dataset_name=${DATASET_SPLIT[0]}
    GEPA_prompt_src=${DATASET_SPLIT[1]}
    PROMPT_FILE="${GEPA_ROOT}/${GEPA_prompt_src}/${MODEL}/best_system_prompt.txt"

    echo "Running Gemini direct QA with GEPA system prompt for ${dataset_name} using ${MODEL}"
    python -m prompting.run_direct_qa \
        gemini_submit \
            --model_name ${MODEL} \
            --dataset ${dataset_name} \
            --system_prompt_file ${PROMPT_FILE} \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA/${MODEL}/RAG=0.jsonl \
            --dataset_path ${SCRATCH_DIR}/datasets/${dataset_name}/test.jsonl \
            --disable_few_shot \
            no_passages
    echo "Finished ${dataset_name} with ${MODEL}"

    echo "Running Gemini direct QA with all passages and GEPA system prompt for ${dataset_name} using ${MODEL}"
    python -m prompting.run_direct_qa \
        gemini_submit \
            --model_name ${MODEL} \
            --dataset ${dataset_name} \
            --system_prompt_file ${PROMPT_FILE} \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA/${MODEL}/RAG=all.jsonl \
            --dataset_path ${SCRATCH_DIR}/datasets/${dataset_name}/test.jsonl \
            --disable_few_shot \
            use_passages
    echo "Finished ${dataset_name} with ${MODEL} using all passages"

    echo "Running Gemini direct QA with top-4 RAG and GEPA system prompt for ${dataset_name} using ${MODEL}"
    python -m prompting.run_direct_qa \
        gemini_submit \
            --model_name ${MODEL} \
            --dataset ${dataset_name} \
            --system_prompt_file ${PROMPT_FILE} \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA/${MODEL}/RAG=4.jsonl \
            --dataset_path ${SCRATCH_DIR}/datasets/${dataset_name}/test.jsonl \
            --disable_few_shot \
            use_RAG
    echo "Finished ${dataset_name} with ${MODEL} using top-4 RAG"

    echo "Running Gemini direct QA with web search and GEPA system prompt for ${dataset_name} using ${MODEL}"
    python -m prompting.run_direct_qa \
        gemini_submit \
            --model_name ${MODEL} \
            --dataset ${dataset_name} \
            --system_prompt_file ${PROMPT_FILE} \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA/${MODEL}/RAG=web.jsonl \
            --dataset_path ${SCRATCH_DIR}/datasets/${dataset_name}/test.jsonl \
            --disable_few_shot \
            no_passages
    echo "Finished ${dataset_name} with ${MODEL} using web search"
done
