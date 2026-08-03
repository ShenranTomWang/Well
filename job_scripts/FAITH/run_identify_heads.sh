source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

MODELS=(
    Meta-Llama-3-8B-Instruct
    Qwen2.5-7B-Instruct
    gemma-4-E4B-it
    Olmo-3-7B-Instruct
)
for MODEL in "${MODELS[@]}"; do
    echo "Identifying heads for ${MODEL}..."
    python -m FAITH.identify_heads \
        --model-name-or-path ${HF_HOME}/${MODEL} \
        --dataset-file ${SCRATCH_DIR}/datasets/Movies/wikidata_movies.json \
        --output-dir ${SCRATCH_DIR}/FP_Hallucination/FAITH/heads/${MODEL}
    echo "Finished identifying heads for ${MODEL}"
done
