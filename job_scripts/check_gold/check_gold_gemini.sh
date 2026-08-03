source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

DATASETS=(
    SynQA2FPQ/0
    SynQA2TPQ/1
    CancerMyth/0
    CancerMythNFP/1
)
MODEL=gemini-3-flash-preview

for COMBINATION in "${DATASETS[@]}"; do
    IFS='/' read -r -a parts <<< "$COMBINATION"
    DATASET=${parts[0]}
    EXPECTED_RESULT=${parts[1]}
    echo "Checking ${DATASET} with ${MODEL}"
    python -m prompting.run_fact_check \
        gemini_submit \
            --model_name ${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0.jsonl \
            --disable_batching \
            --check_gold \
            no_passages

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL}"

    echo "Checking ${DATASET} with ${MODEL} + top-4 RAG passages"
    python -m prompting.run_fact_check \
        gemini_submit \
            --model_name ${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4.jsonl \
            --disable_batching \
            --check_gold \
            use_RAG

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} + top-4 RAG passages"

    echo "Checking ${DATASET} with ${MODEL} + all passages"
    python -m prompting.run_fact_check \
        gemini_submit \
            --model_name ${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all.jsonl \
            --disable_batching \
            --check_gold \
            use_passages

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} + all passages"

    echo "Checking ${DATASET} with ${MODEL} + web search"
    python -m prompting.run_fact_check \
        gemini_submit \
            --model_name ${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=web.jsonl \
            --disable_batching \
            --check_gold \
            --web_search \
            no_passages

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=web.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} + web search"

    echo "Checking ${DATASET} with ${MODEL} (thinking)"
    python -m prompting.run_fact_check \
        gemini_submit \
            --model_name ${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0_thinking.jsonl \
            --disable_batching \
            --check_gold \
            --thinking_level high \
            no_passages

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0_thinking.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} (thinking)"

    echo "Checking ${DATASET} with ${MODEL} + top-4 RAG passages (thinking)"
    python -m prompting.run_fact_check \
        gemini_submit \
            --model_name ${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4_thinking.jsonl \
            --disable_batching \
            --check_gold \
            --thinking_level high \
            use_RAG

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4_thinking.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} + top-4 RAG passages (thinking)"

    echo "Checking ${DATASET} with ${MODEL} + all passages (thinking)"
    python -m prompting.run_fact_check \
        gemini_submit \
            --model_name ${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all_thinking.jsonl \
            --disable_batching \
            --check_gold \
            --thinking_level high \
            use_passages

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all_thinking.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} + all passages (thinking)"

    echo "Checking ${DATASET} with ${MODEL} + web search (thinking)"
    python -m prompting.run_fact_check \
        gemini_submit \
            --model_name ${MODEL} \
            --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
            --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=web_thinking.jsonl \
            --disable_batching \
            --check_gold \
            --web_search \
            --thinking_level high \
            no_passages

    python run_check_gold_eval.py \
        evaluate \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=web_thinking.jsonl \
            --expected_result ${EXPECTED_RESULT}
    echo "Done checking ${DATASET} with ${MODEL} + web search (thinking)"
done

DATASET=CancerMythNFP
EXPECTED_RESULT=1
echo "Checking ${DATASET} with ${MODEL} (ablation)"
python -m prompting.run_fact_check \
    gemini_submit \
        --model_name ${MODEL} \
        --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
        --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0_ablation.jsonl \
        --disable_batching \
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
    gemini_submit \
        --model_name ${MODEL} \
        --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
        --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4_ablation.jsonl \
        --disable_batching \
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
    gemini_submit \
        --model_name ${MODEL} \
        --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
        --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all_ablation.jsonl \
        --disable_batching \
        --check_gold \
        --template_class CancerMythNFPLLMCheckAblationTemplate \
        use_passages

python run_check_gold_eval.py \
    evaluate \
        --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all_ablation.jsonl \
        --expected_result ${EXPECTED_RESULT}
echo "Done checking ${DATASET} with ${MODEL} + all passages (ablation)"

echo "Checking ${DATASET} with ${MODEL} + web search (ablation)"
python -m prompting.run_fact_check \
    gemini_submit \
        --model_name ${MODEL} \
        --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
        --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=web_ablation.jsonl \
        --disable_batching \
        --check_gold \
        --web_search \
        --template_class CancerMythNFPLLMCheckAblationTemplate \
        no_passages

python run_check_gold_eval.py \
    evaluate \
        --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=web_ablation.jsonl \
        --expected_result ${EXPECTED_RESULT}
echo "Done checking ${DATASET} with ${MODEL} + web search (ablation)"

echo "Checking ${DATASET} with ${MODEL} (ablation) (thinking)"
python -m prompting.run_fact_check \
    gemini_submit \
        --model_name ${MODEL} \
        --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
        --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0_thinking_ablation.jsonl \
        --disable_batching \
        --check_gold \
        --thinking_level high \
        --template_class CancerMythNFPLLMCheckAblationTemplate \
        no_passages

python run_check_gold_eval.py \
    evaluate \
        --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=0_thinking_ablation.jsonl \
        --expected_result ${EXPECTED_RESULT}
echo "Done checking ${DATASET} with ${MODEL} (ablation) (thinking)"

echo "Checking ${DATASET} with ${MODEL} + top-4 RAG passages (ablation) (thinking)"
python -m prompting.run_fact_check \
    gemini_submit \
        --model_name ${MODEL} \
        --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
        --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4_thinking_ablation.jsonl \
        --disable_batching \
        --check_gold \
        --thinking_level high \
        --template_class CancerMythNFPLLMCheckAblationTemplate \
        use_RAG

python run_check_gold_eval.py \
    evaluate \
        --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=4_thinking_ablation.jsonl \
        --expected_result ${EXPECTED_RESULT}
echo "Done checking ${DATASET} with ${MODEL} + top-4 RAG passages (ablation) (thinking)"

echo "Checking ${DATASET} with ${MODEL} + all passages (ablation) (thinking)"
python -m prompting.run_fact_check \
    gemini_submit \
        --model_name ${MODEL} \
        --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
        --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all_thinking_ablation.jsonl \
        --disable_batching \
        --check_gold \
        --thinking_level high \
        --template_class CancerMythNFPLLMCheckAblationTemplate \
        use_passages

python run_check_gold_eval.py \
    evaluate \
        --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=all_thinking_ablation.jsonl \
        --expected_result ${EXPECTED_RESULT}
echo "Done checking ${DATASET} with ${MODEL} + all passages (ablation) (thinking)"

echo "Checking ${DATASET} with ${MODEL} + web search (ablation) (thinking)"
python -m prompting.run_fact_check \
    gemini_submit \
        --model_name ${MODEL} \
        --file ${SCRATCH_DIR}/datasets/${DATASET}/test.jsonl \
        --out_file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=web_thinking_ablation.jsonl \
        --disable_batching \
        --check_gold \
        --web_search \
        --thinking_level high \
        --template_class CancerMythNFPLLMCheckAblationTemplate \
        no_passages

python run_check_gold_eval.py \
    evaluate \
        --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Check_Gold/${MODEL}/RAG=web_thinking_ablation.jsonl \
        --expected_result ${EXPECTED_RESULT}
echo "Done checking ${DATASET} with ${MODEL} + web search (ablation) (thinking)"