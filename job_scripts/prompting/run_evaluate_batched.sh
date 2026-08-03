source ${HOME_DIR}/.bashrc
source ${SCRATCH_DIR}/envs/FP_Hallucination/.venv/bin/activate
cd ${PROJECT_DIR}/FP_Hallucination

submit_response_level_score() {
    local file=""
    local previous_arg=""

    for arg in "$@"; do
        if [ "${previous_arg}" = "--file" ]; then
            file="${arg}"
            break
        fi
        previous_arg="${arg}"
    done

    if [ -z "${file}" ]; then
        echo "Error: submission is missing --file" >&2
        return 1
    fi

    local evaluated_file="${file%.jsonl}_response_level_score_evaluated.jsonl"
    if [ -f "${evaluated_file}" ]; then
        echo "Skipping submission; evaluated file already exists: ${evaluated_file}"
        return 0
    fi

    python run_response_level_score.py "$@"
}

DATASETS=(
    SynQA2FPQ
    SynQA2TPQ
    QA2FPQ
    QA2TPQ
    CancerMyth
    CancerMythNFP
    CREPEFPQ
    CREPETPQ
)
MODELS=(
    gemini-3-flash-preview
    Meta-Llama-3-8B-Instruct
    Qwen2.5-7B-Instruct
    gemma-4-E4B-it
    Olmo-3-7B-Instruct
)
for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: ${DATASET} for direct QA evaluation"
    for MODEL in "${MODELS[@]}"; do
        echo "Submitting response level score eval for direct QA (no RAG) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA/${MODEL}/RAG=0.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for direct QA (RAG with all passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA/${MODEL}/RAG=all.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for direct QA (RAG with top-4 passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA/${MODEL}/RAG=4.jsonl \
                --dataset ${DATASET}
        echo "finished..."


        if [ "$MODEL" != "gemini-3-flash-preview" ]; then
            echo "Submitting response level score eval for direct QA knockout with movie heads on ${DATASET} using ${MODEL} on dataset ${DATASET}"
            submit_response_level_score \
                response_level_score_submit \
                    --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FAITH/${MODEL}/RAG=0.jsonl \
                    --dataset ${DATASET}
            echo "finished..."
        fi
    done
    echo "Finished processing dataset: ${DATASET} for direct QA evaluation"

    echo "Processing dataset: ${DATASET} for SDualCritique evaluation"
    for MODEL in "${MODELS[@]}"; do
        echo "Submitting response level score eval for SDualCritique (no RAG) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/SDualCritique/${MODEL}/RAG=0.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for SDualCritique (RAG with all passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/SDualCritique/${MODEL}/RAG=all.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for SDualCritique (RAG with top-4 passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/SDualCritique/${MODEL}/RAG=4.jsonl \
                --dataset ${DATASET}
        echo "finished..."
    done
    echo "Finished processing dataset: ${DATASET} for SDualCritique evaluation"

    echo "Processing dataset: ${DATASET} for presupposition extraction + explicitly addressing the FP evaluation"
    for MODEL in "${MODELS[@]}"; do
        if [ "$MODEL" = "gemini-3-flash-preview" ]; then
            CHECK="gemini"
        else
            CHECK="transformers"
        fi
        echo "Submitting response level score eval for full pipeline (no RAG) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Final_Response/${MODEL}/RAG=0_${CHECK}_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for full pipeline (RAG with all passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Final_Response/${MODEL}/RAG=all_${CHECK}_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for full pipeline (RAG with all passages) with ${MODEL} on dataset ${DATASET}, fact-checking with MiniCheck"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Final_Response/${MODEL}/RAG=all_minichecked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for full pipeline (RAG with top-4 passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Final_Response/${MODEL}/RAG=4_${CHECK}_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for full pipeline (RAG with top-4 passages) with ${MODEL} on dataset ${DATASET}, fact-checking with MiniCheck"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Final_Response/${MODEL}/RAG=4_minichecked.jsonl \
                --dataset ${DATASET}
        echo "finished..."
    done
    echo "Finished processing dataset: ${DATASET} for presupposition extraction + explicitly addressing the FP evaluation"

    echo "Processing dataset: ${DATASET} for FP identification +interpretation evaluation"
    for MODEL in "${MODELS[@]}"; do
        echo "Submitting response level score eval for FP identification +interpretation evaluation (no RAG) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FP_Identification_Final_Response/${MODEL}/RAG=0.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for FP identification +interpretation evaluation (RAG with all passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FP_Identification_Final_Response/${MODEL}/RAG=all.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for FP identification +interpretation evaluation (RAG with top-4 passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FP_Identification_Final_Response/${MODEL}/RAG=4.jsonl \
                --dataset ${DATASET}
        echo "finished..."
    done
    echo "Finished processing dataset: ${DATASET} for FP identification + interpretation evaluation"

    echo "Processing dataset: ${DATASET} for question to statement + explicitly addressing the FP evaluation"
    for MODEL in "${MODELS[@]}"; do
        if [ "$MODEL" = "gemini-3-flash-preview" ]; then
            CHECK="gemini"
        else
            CHECK="transformers"
        fi
        echo "Submitting response level score eval for question to statement + explicitly addressing the FP evaluation (no RAG) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Final_Response/${MODEL}/RAG=0_${CHECK}_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for question to statement + explicitly addressing the FP evaluation (RAG with all passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Final_Response/${MODEL}/RAG=all_${CHECK}_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for question to statement + explicitly addressing the FP evaluation (RAG with all passages) with ${MODEL} on dataset ${DATASET}, fact-checking with MiniCheck"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Final_Response/${MODEL}/RAG=all_minichecked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for question to statement + explicitly addressing the FP evaluation (RAG with top-4 passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Final_Response/${MODEL}/RAG=4_${CHECK}_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for question to statement + explicitly addressing the FP evaluation (RAG with top-4 passages) with ${MODEL} on dataset ${DATASET}, fact-checking with MiniCheck"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Final_Response/${MODEL}/RAG=4_minichecked.jsonl \
                --dataset ${DATASET}
        echo "finished..."
    done
    echo "Finished processing dataset: ${DATASET} for question to statement + explicitly addressing the FP evaluation"

    echo "Processing dataset: ${DATASET} for PreWoMe evaluation"
    for MODEL in "${MODELS[@]}"; do
        if [ "$MODEL" = "gemini-3-flash-preview" ]; then
            CHECK="gemini"
        else
            CHECK="transformers"
        fi

        echo "Submitting response level score eval for feedback action final response (no RAG) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Feedback_Action_Final_Response/${MODEL}/RAG=0_${CHECK}_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for feedback action final response (RAG with all passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Feedback_Action_Final_Response/${MODEL}/RAG=all_${CHECK}_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for feedback action final response (RAG with top-4 passages) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Feedback_Action_Final_Response/${MODEL}/RAG=4_${CHECK}_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."
    done
    echo "Finished processing dataset: ${DATASET} for PreWoMe evaluation"
done

MODEL=gemini-3-flash-preview
for DATASET in "${DATASETS[@]}"; do
    echo "Submitting response level score eval for direct QA (web search) with ${MODEL} on dataset ${DATASET}"
    submit_response_level_score \
        response_level_score_submit \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA/${MODEL}/RAG=web.jsonl \
            --dataset ${DATASET}
    echo "finished..."

    echo "Submitting response level score eval for SDualCritique (web search) with ${MODEL} on dataset ${DATASET}"
    submit_response_level_score \
        response_level_score_submit \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/SDualCritique/${MODEL}/RAG=web.jsonl \
            --dataset ${DATASET}
    echo "finished..."

    echo "Submitting response level score eval for full pipeline (web search) with ${MODEL} on dataset ${DATASET}"
    submit_response_level_score \
        response_level_score_submit \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Final_Response/${MODEL}/RAG=web_gemini_checked.jsonl \
            --dataset ${DATASET}
    echo "finished..."

    echo "Submitting response level score eval for question to statement + explicitly addressing the FP evaluation (web search) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Statement_Final_Response/${MODEL}/RAG=0_gemini_checked.jsonl \
                --dataset ${DATASET}
        echo "finished..."

    echo "Submitting response level score eval for FP identification +interpretation evaluation (web search) with ${MODEL} on dataset ${DATASET}"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/FP_Identification_Final_Response/${MODEL}/RAG=web.jsonl \
                --dataset ${DATASET}
    echo "finished..."

    echo "Submitting response level score eval for PreWoMe (web search) with ${MODEL} on dataset ${DATASET}"
    submit_response_level_score \
        response_level_score_submit \
            --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Feedback_Action_Final_Response/${MODEL}/RAG=web_gemini_checked.jsonl \
            --dataset ${DATASET}
    echo "finished..."
done

MODELS=(
    gemma-4-E4B-it
)
for DATASET in "${DATASETS[@]}"; do
    echo "Processing dataset: ${DATASET} for thinking score evaluation"
    for MODEL in "${MODELS[@]}"; do
        CHECK="transformers"

        echo "Submitting response level score eval for direct QA (no RAG) with ${MODEL} on dataset ${DATASET} (thinking)"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA/${MODEL}/RAG=0_thinking.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for direct QA (RAG with all passages) with ${MODEL} on dataset ${DATASET} (thinking)"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA/${MODEL}/RAG=all_thinking.jsonl \
                --dataset ${DATASET}
        echo "finished..."

        echo "Submitting response level score eval for direct QA (RAG with top-4 passages) with ${MODEL} on dataset ${DATASET} (thinking)"
        submit_response_level_score \
            response_level_score_submit \
                --file ${SCRATCH_DIR}/FP_Hallucination/out/${DATASET}/Direct_QA/${MODEL}/RAG=4_thinking.jsonl \
                --dataset ${DATASET}
        echo "finished..."
    done
    echo "Finished processing dataset: ${DATASET} for thinking score evaluation"
done
