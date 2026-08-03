source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

OUT_ROOT=${SCRATCH_DIR}/FP_Hallucination/out
PLOT_ROOT=${SCRATCH_DIR}/FP_Hallucination/plots

DATASET_GROUPS=(
    "CancerMyth CancerMythNFP SynQA2FPQ SynQA2TPQ cancer_myth_and_synqa2"
    "QA2FPQ QA2TPQ CREPEFPQ CREPETPQ qa2_and_crepe"
)

SETTINGS=(
    "Direct QA/Direct_QA"
    "GEPA (FPQ)/Direct_QA_GEPA"
    "GEPA (FPQ + TPQ)/Direct_QA_GEPA_balanced"
    "Presupposition Extraction + Fact Checking/Final_Response"
    "PreWoMe/Feedback_Action_Final_Response"
    "FAITH/FAITH"
    "FP Identification/FP_Identification_Final_Response"
    "Question to Statement/Statement_Final_Response"
    "Self-Dual-Critique/SDualCritique"
    "Fine-tuning/FalseQA"
)

for group in "${DATASET_GROUPS[@]}"; do
    read -r fpq_dataset_1 tpq_dataset_1 fpq_dataset_2 tpq_dataset_2 group_name <<< "${group}"
    out_file="${PLOT_ROOT}/${group_name}/pareto_frontier.png"

    echo "Plotting ${fpq_dataset_1} vs ${tpq_dataset_1} and ${fpq_dataset_2} vs ${tpq_dataset_2}"
    if python -m plotting.plot_scatter_plot \
        --fpq_roots "${OUT_ROOT}/${fpq_dataset_1}" "${OUT_ROOT}/${fpq_dataset_2}" \
        --tpq_roots "${OUT_ROOT}/${tpq_dataset_1}" "${OUT_ROOT}/${tpq_dataset_2}" \
        --settings "${SETTINGS[@]}" \
        --out "${out_file}" \
        --titles "${fpq_dataset_1} vs ${tpq_dataset_1}" "${fpq_dataset_2} vs ${tpq_dataset_2}"; then
        echo "Saved ${out_file}"
    else
        echo "Plotting failed for ${group_name}."
    fi
done

echo "All Pareto plots saved under ${PLOT_ROOT}"
