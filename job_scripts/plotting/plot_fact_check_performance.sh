source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

OUT_ROOT=${SCRATCH_DIR}/FP_Hallucination/out
PLOT_ROOT=${SCRATCH_DIR}/FP_Hallucination/plots

CANCER_DATASETS=(
    "CancerMyth/0"
    "CancerMythNFP/1"
)

SYNQA2_DATASETS=(
    "SynQA2FPQ/0"
    "SynQA2TPQ/1"
)

SYNQA2_SETTINGS=(
    "No RAG/RAG=0"
    "No RAG + reasoning/RAG=0_thinking"
    "Top-4 RAG/RAG=4"
    "Top-4 RAG + reasoning/RAG=4_thinking"
    "All passages/RAG=all"
    "All passages + reasoning/RAG=all_thinking"
    "Web search/RAG=web"
    "Web search + reasoning/RAG=web_thinking"
)

CANCER_SETTINGS=(
    "No RAG/RAG=0"
    "No RAG + reasoning/RAG=0_thinking"
    "No RAG + context/RAG=0_ablation"
    "No RAG + reasoning + context/RAG=0_thinking_ablation"
    "Top-4 RAG/RAG=4"
    "Top-4 RAG + reasoning/RAG=4_thinking"
    "Top-4 RAG + context/RAG=4_ablation"
    "Top-4 RAG + reasoning + context/RAG=4_thinking_ablation"
    "All passages/RAG=all"
    "All passages + reasoning/RAG=all_thinking"
    "All passages + context/RAG=all_ablation"
    "All passages + reasoning + context/RAG=all_thinking_ablation"
    "Web search/RAG=web"
    "Web search + reasoning/RAG=web_thinking"
    "Web search + context/RAG=web_ablation"
    "Web search + reasoning + context/RAG=web_thinking_ablation"
)

cancer_out_file="${PLOT_ROOT}/CancerMyth_vs_CancerMythNFP/fact_check_performance.png"
python -m plotting.plot_fact_check_performance \
    --out_root "${OUT_ROOT}" \
    --datasets "${CANCER_DATASETS[@]}" \
    --settings "${CANCER_SETTINGS[@]}" \
    --out "${cancer_out_file}" \
    --title "CancerMyth vs. CancerMythNFP fact-checking performance"
echo "Saved ${cancer_out_file}"

synqa2_out_file="${PLOT_ROOT}/SynQA2FPQ_vs_SynQA2TPQ/fact_check_performance.png"
python -m plotting.plot_fact_check_performance \
    --out_root "${OUT_ROOT}" \
    --datasets "${SYNQA2_DATASETS[@]}" \
    --settings "${SYNQA2_SETTINGS[@]}" \
    --out "${synqa2_out_file}" \
    --title "SynQA2 FPQ vs. TPQ fact-checking performance"
echo "Saved ${synqa2_out_file}"
