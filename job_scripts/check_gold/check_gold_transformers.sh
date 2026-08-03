source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    SynQA2FPQ/0
    SynQA2TPQ/1
    CancerMyth/0
    CancerMythNFP/1
)
MODELS=(
    Meta-Llama-3-8B-Instruct
    Qwen2.5-7B-Instruct
    gemma-4-E4B-it
    Olmo-3-7B-Instruct
    Olmo-3-7B-Instruct-SFT
    Olmo-3-7B-Instruct-DPO
)

for MODEL in "${MODELS[@]}"; do
    for COMBINATION in "${DATASETS[@]}"; do
        IFS='/' read -r -a parts <<< "$COMBINATION"
        DATASET=${parts[0]}
        EXPECTED_RESULT=${parts[1]}
        echo "Checking ${DATASET} with ${MODEL}"
        python -m prompting.run_fact_check \
            transformers \
                --model_name ${HF_HOME}/${MODEL} \
                --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
                --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0.jsonl \
                --check_gold \
                no_passages

        python run_check_gold_eval.py \
            evaluate \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0.jsonl \
                --expected_result ${EXPECTED_RESULT}
        echo "Done checking ${DATASET} with ${MODEL}"

        echo "Checking ${DATASET} with ${MODEL} + top-4 RAG passages"
        python -m prompting.run_fact_check \
            transformers \
                --model_name ${HF_HOME}/${MODEL} \
                --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
                --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4.jsonl \
                --check_gold \
                use_RAG

        python run_check_gold_eval.py \
            evaluate \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4.jsonl \
                --expected_result ${EXPECTED_RESULT}
        echo "Done checking ${DATASET} with ${MODEL} + top-4 RAG passages"

        echo "Checking ${DATASET} with ${MODEL} + all passages"
        python -m prompting.run_fact_check \
            transformers \
                --model_name ${HF_HOME}/${MODEL} \
                --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
                --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all.jsonl \
                --check_gold \
                use_passages

        python run_check_gold_eval.py \
            evaluate \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all.jsonl \
                --expected_result ${EXPECTED_RESULT}
        echo "Done checking ${DATASET} with ${MODEL} + all passages"
    done

    DATASET=CancerMythNFP
    EXPECTED_RESULT=1
    echo "Checking ${DATASET} with ${MODEL} (ablation)"
    python -m prompting.run_fact_check \
        transformers \
            --model_name ${HF_HOME}/${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0_ablation.jsonl \
            --check_gold \
            --template_class CancerMythNFPLLMCheckAblationTemplate \
            no_passages

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0_ablation.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} (ablation)"

    echo "Checking ${DATASET} with ${MODEL} + top-4 RAG passages (ablation)"
    python -m prompting.run_fact_check \
        transformers \
            --model_name ${HF_HOME}/${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4_ablation.jsonl \
            --check_gold \
            --template_class CancerMythNFPLLMCheckAblationTemplate \
            use_RAG

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4_ablation.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} + top-4 RAG passages (ablation)"

    echo "Checking ${DATASET} with ${MODEL} + all passages (ablation)"
    python -m prompting.run_fact_check \
        transformers \
            --model_name ${HF_HOME}/${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all_ablation.jsonl \
            --check_gold \
            --template_class CancerMythNFPLLMCheckAblationTemplate \
            use_passages

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all_ablation.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} + all passages (ablation)"
done