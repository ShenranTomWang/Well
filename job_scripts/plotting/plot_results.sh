source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

OUT_ROOT=${SCRATCH_DIR}/FP_Hallucination/out

DATASET_PAIRS=(
    "CancerMyth CancerMythNFP"
    "CancerMyth-dev CancerMythNFP-dev"
    "QA2FPQ QA2TPQ"
    "SynQA2FPQ SynQA2TPQ"
)

PLOT_FAMILIES=(
    "DirectQA|Direct QA|Direct_QA"
    "DirectQA_GEPA|Direct QA with GEPA|Direct_QA_GEPA"
    "Pipeline|Presupposition Extraction + Fact Checking|Final_Response"
    "PreWoMe|PreWoMe|Feedback_Action_Final_Response"
    "FactCheckLogical|Fact Checking + Logical Form|Fact_Check_Logical_Form_Final_Response"
    "FAITH|FAITH|FAITH"
    "FP_Identification|FP Identification|FP_Identification_Final_Response"
    "Question_To_Statement|Question to Statement|Statement_Final_Response"
    "SDualCritique|Self-Dual-Critique|SDualCritique"
)

plot_family() {
    local fpq_dataset=$1
    local tpq_dataset=$2
    local slug=$3
    local title=$4
    local setting=$5
    local plot_root=$6

    echo "Discovering models for ${fpq_dataset} vs ${tpq_dataset} / ${setting}"
    mapfile -t models < <(
        find \
            "${OUT_ROOT}/${fpq_dataset}/${setting}" \
            "${OUT_ROOT}/${tpq_dataset}/${setting}" \
            -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
            | sort -u
    )

    if [ "${#models[@]}" -eq 0 ]; then
        echo "No model directories found for ${fpq_dataset} vs ${tpq_dataset} / ${setting}; skipping."
        return
    fi

    for model in "${models[@]}"; do
        local out_dir="${plot_root}/${setting}/${model}"
        local out_file="${out_dir}/${slug}.png"
        local fpq_dir="${OUT_ROOT}/${fpq_dataset}/${setting}/${model}"
        local tpq_dir="${OUT_ROOT}/${tpq_dataset}/${setting}/${model}"

        if [ ! -d "${fpq_dir}" ] || [ ! -d "${tpq_dir}" ]; then
            echo "Missing paired model directory for ${setting} / ${model}; skipping."
            continue
        fi

        mkdir -p "${out_dir}"

        echo "Plotting ${title} for ${fpq_dataset} vs ${tpq_dataset} / ${model}"
        if python -m plotting.plot_response_level_scores \
            --fpq_dir "${fpq_dir}" \
            --tpq_dir "${tpq_dir}" \
            --out "${out_file}" \
            --title "${title}: ${fpq_dataset} vs ${tpq_dataset}: ${model}"; then
            echo "Saved ${out_file}"
        else
            echo "No matching evaluated files for ${title} / ${fpq_dataset} vs ${tpq_dataset} / ${model}; skipping."
        fi
    done
}

for pair in "${DATASET_PAIRS[@]}"; do
    read -r fpq_dataset tpq_dataset <<< "${pair}"
    plot_root="${SCRATCH_DIR}/FP_Hallucination/plots/${fpq_dataset}_vs_${tpq_dataset}"

    echo "Plotting dataset pair: ${fpq_dataset} vs ${tpq_dataset}"
    for family in "${PLOT_FAMILIES[@]}"; do
        IFS='|' read -r slug title setting <<< "${family}"
        plot_family "${fpq_dataset}" "${tpq_dataset}" "${slug}" "${title}" "${setting}" "${plot_root}"
    done
    echo "Plots for ${fpq_dataset} vs ${tpq_dataset} saved under ${plot_root}"
done

echo "All plots saved under ${SCRATCH_DIR}/FP_Hallucination/plots"
