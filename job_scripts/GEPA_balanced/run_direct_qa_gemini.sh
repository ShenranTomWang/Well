source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    CancerMyth/CancerMyth
    CancerMythNFP/CancerMyth
    QA2FPQ/QA2
    QA2TPQ/QA2
    SynQA2FPQ/SynQA2
    SynQA2TPQ/SynQA2
    CREPEFPQ/CREPE
    CREPETPQ/CREPE
)

MODEL=gemini-3-flash-preview
GEPA_ROOT="${SCRATCH_DIR}/FP_Hallucination/GEPA_balanced"

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
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA_balanced/${MODEL}/RAG=0.jsonl \
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
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA_balanced/${MODEL}/RAG=all.jsonl \
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
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA_balanced/${MODEL}/RAG=4.jsonl \
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
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${dataset_name}/Direct_QA_GEPA_balanced/${MODEL}/RAG=web.jsonl \
            --dataset_path ${SCRATCH_DIR}/datasets/${dataset_name}/test.jsonl \
            --disable_few_shot \
            no_passages
    echo "Finished ${dataset_name} with ${MODEL} using web search"
done
