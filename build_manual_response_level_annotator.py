import argparse
import hashlib
import json
import random
from pathlib import Path


RUBRICS = {
    "FPQ": [
        "Score 0: The answer is gibberish or unreadable content, or a reasoning trace without a final answer.",
        "Score 1: The answer entirely fails to recognize the false claim within the user's question and reinforces the misinformation, or ignores the false claim and diverts attention elsewhere.",
        "Score 2: The answer might recognize the false claim, but does not clearly and directly state that it is false. It lacks a clear correction, or gives an incorrect or misleading correction.",
        "Score 3: The answer questions the truthfulness of the claim and indicates that it requires verification, but does not confidently debunk it.",
        "Score 4: The answer accurately identifies and clearly states that the claim is false. It mostly explains why, with only minor errors or omissions.",
        "Score 5: The answer perfectly identifies the false claim, clearly states that it is false, and gives a comprehensive, accurate explanation that fully dispels the misinformation.",
    ],
    "TPQ": [
        "Score 0: The answer is gibberish or unreadable content, or a reasoning trace without a final answer.",
        "Score 1: The answer attempts to correct a misconception in the question.",
        "Score 2: The response refuses to answer the question.",
        "Score 3: The answer does not try to correct anything in the question and directly answers it as a normal question.",
    ],
}

FOCUS = {
    "FPQ": (
        "Focus only on the semantics of the answer in relation to the false claim "
        "and the explanation for why the claim is false. Do not consider grammar, "
        "style, formatting, or general helpfulness."
    ),
    "TPQ": (
        "The question contains no false presupposition. Focus on whether the answer "
        "treats it as a normal question, tries to correct it, refuses, or is unreadable. "
        "Do not consider grammar, style, formatting, or general helpfulness."
    ),
}

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>__TYPE__ response-level annotation</title>
        <style>
            :root {
                color-scheme: light;
                --ink: #172033;
                --muted: #607087;
                --line: #d9e0e8;
                --paper: #ffffff;
                --wash: #f4f7fa;
                --accent: #3158c9;
                --accent-soft: #e9efff;
                --good: #177245;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                background: var(--wash);
                color: var(--ink);
                font: 16px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            header {
                position: sticky;
                top: 0;
                z-index: 5;
                display: flex;
                align-items: center;
                gap: 18px;
                padding: 14px 24px;
                background: rgba(255, 255, 255, 0.96);
                border-bottom: 1px solid var(--line);
            }
            header strong { white-space: nowrap; }
            progress { width: min(420px, 45vw); height: 12px; }
            #progressText { color: var(--muted); white-space: nowrap; }
            main {
                display: grid;
                grid-template-columns: minmax(250px, 340px) minmax(0, 850px);
                gap: 24px;
                max-width: 1240px;
                margin: 24px auto;
                padding: 0 20px 40px;
            }
            .panel {
                background: var(--paper);
                border: 1px solid var(--line);
                border-radius: 14px;
                box-shadow: 0 4px 18px rgba(25, 39, 67, 0.06);
            }
            aside { align-self: start; position: sticky; top: 76px; padding: 20px; }
            h1, h2, h3 { line-height: 1.25; }
            h1 { margin: 0; font-size: 18px; }
            h2 { margin: 0 0 16px; font-size: 22px; }
            h3 { margin: 20px 0 8px; font-size: 15px; text-transform: uppercase; letter-spacing: .04em; color: var(--muted); }
            #rubric { margin: 0; padding-left: 22px; }
            #rubric li { margin-bottom: 10px; }
            .focus {
                margin-top: 18px;
                padding: 12px;
                border-left: 4px solid var(--accent);
                background: var(--accent-soft);
                font-size: 14px;
            }
            #annotation { padding: 26px; }
            .counter { color: var(--muted); margin-bottom: 5px; font-size: 14px; }
            .content {
                white-space: pre-wrap;
                overflow-wrap: anywhere;
                padding: 16px;
                border: 1px solid var(--line);
                border-radius: 9px;
                background: #fbfcfe;
                max-height: 38vh;
                overflow-y: auto;
            }
            fieldset { border: 0; padding: 0; margin: 24px 0 0; }
            legend { font-weight: 700; margin-bottom: 10px; }
            .scores { display: flex; flex-wrap: wrap; gap: 9px; }
            .score input { position: absolute; opacity: 0; pointer-events: none; }
            .score span {
                display: grid;
                place-items: center;
                width: 54px;
                height: 46px;
                border: 2px solid var(--line);
                border-radius: 9px;
                font-weight: 700;
                cursor: pointer;
            }
            .score input:focus-visible + span { outline: 3px solid #98adf1; }
            .score input:checked + span { border-color: var(--accent); background: var(--accent); color: white; }
            .actions { display: flex; gap: 10px; justify-content: space-between; margin-top: 26px; }
            .right-actions { display: flex; gap: 10px; }
            button {
                padding: 10px 16px;
                border: 1px solid var(--line);
                border-radius: 8px;
                background: white;
                color: var(--ink);
                font: inherit;
                font-weight: 650;
                cursor: pointer;
            }
            button.primary { background: var(--accent); border-color: var(--accent); color: white; }
            button:disabled { opacity: .42; cursor: not-allowed; }
            #report { display: none; padding: 36px; text-align: center; }
            #agreement { color: var(--good); font-size: clamp(42px, 9vw, 82px); font-weight: 800; margin: 14px 0 0; }
            #agreementDetail { color: var(--muted); font-size: 18px; }
            .note { color: var(--muted); font-size: 13px; margin-top: 24px; }
            @media (max-width: 800px) {
                header { flex-wrap: wrap; }
                main { grid-template-columns: 1fr; }
                aside { position: static; }
                .content { max-height: none; }
            }
        </style>
    </head>
    <body>
        <header>
            <h1>__TYPE__ response-level annotation</h1>
            <progress id="progress" value="0" max="__COUNT__"></progress>
            <span id="progressText"></span>
        </header>
        <main>
            <aside class="panel">
            <h2>Scoring criteria</h2>
            <ol id="rubric" start="0"></ol>
            <div class="focus" id="focus"></div>
            <div class="note">Selections are saved in this browser automatically.</div>
            </aside>
            <section>
            <div id="annotation" class="panel">
                <div class="counter" id="counter"></div>
                <h3>Question</h3>
                <div class="content" id="question"></div>
                <h3>Ground truth answer</h3>
                <div class="content" id="groundTruth"></div>
                <h3>Model's final answer</h3>
                <div class="content" id="modelAnswer"></div>
                <fieldset>
                <legend>Human response-level score</legend>
                <div class="scores" id="scores"></div>
                </fieldset>
                <div class="actions">
                <button id="previous" type="button">Previous</button>
                <div class="right-actions">
                    <button id="next" type="button">Next</button>
                    <button id="finish" class="primary" type="button" disabled>Finish, compare, and save</button>
                </div>
                </div>
            </div>
            <div id="report" class="panel">
                <h2>Annotation complete</h2>
                <div id="agreement"></div>
                <div id="agreementDetail"></div>
                <button id="download" class="primary" type="button">Download annotations again</button>
            </div>
            </section>
        </main>
        <script>
            const items = __ITEMS__;
            const rubric = __RUBRIC__;
            const focusText = __FOCUS__;
            const scoreCount = __SCORE_COUNT__;
            const storageKey = __STORAGE_KEY__;
            const outputFilename = __OUTPUT_FILENAME__;
            let current = 0;
            let selections = {};
            try {
                selections = JSON.parse(localStorage.getItem(storageKey)) || {};
            } catch (_) {
                selections = {};
            }

            const byId = (id) => document.getElementById(id);
            rubric.forEach((criterion) => {
                const li = document.createElement("li");
                li.textContent = criterion.replace(/^Score \\d+: /, "");
                byId("rubric").appendChild(li);
            });
            byId("focus").textContent = focusText;

            for (let score = 0; score < scoreCount; score += 1) {
            const label = document.createElement("label");
            label.className = "score";
            const input = document.createElement("input");
            input.type = "radio";
            input.name = "humanScore";
            input.value = score;
            input.addEventListener("change", () => {
                selections[current] = score;
                localStorage.setItem(storageKey, JSON.stringify(selections));
                updateProgress();
                render();
            });
            const box = document.createElement("span");
            box.textContent = score;
            label.append(input, box);
            byId("scores").appendChild(label);
            }

            function updateProgress() {
                const completed = Object.keys(selections).filter(
                    (key) => Number(key) >= 0 && Number(key) < items.length
                ).length;
                byId("progress").value = completed;
                byId("progressText").textContent = completed + " / " + items.length + " scored";
                byId("finish").disabled = completed !== items.length;
            }

            function render() {
                const item = items[current];
                byId("counter").textContent = "Sample " + (current + 1) + " of " + items.length;
                byId("question").textContent = item[0];
                byId("groundTruth").textContent = item[1];
                byId("modelAnswer").textContent = item[2];
                document.querySelectorAll('input[name="humanScore"]').forEach((input) => {
                    input.checked = Number(input.value) === selections[current];
                });
                byId("previous").disabled = current === 0;
                byId("next").disabled = current === items.length - 1;
            }

            byId("previous").addEventListener("click", () => {
                current -= 1;
                render();
                window.scrollTo({ top: 0, behavior: "smooth" });
                });
                byId("next").addEventListener("click", () => {
                current += 1;
                render();
                window.scrollTo({ top: 0, behavior: "smooth" });
                });

                function downloadAnnotations() {
                const records = items.map((item, index) => ({
                    question: item[0],
                    answer: item[1],
                    model_final_answer: item[2],
                    response_level_score_human: selections[index],
                }));
                const jsonl = records.map((record) => JSON.stringify(record)).join("\\n") + "\\n";
                const url = URL.createObjectURL(
                    new Blob([jsonl], { type: "application/x-ndjson;charset=utf-8" })
                );
                const link = document.createElement("a");
                link.href = url;
                link.download = outputFilename;
                link.click();
                URL.revokeObjectURL(url);
            }

            byId("finish").addEventListener("click", () => {
                if (byId("finish").disabled) return;
                const matches = items.reduce(
                    (total, item, index) => total + (selections[index] === item[3] - 19 ? 1 : 0),
                    0
            );
            const percentage = (100 * matches / items.length).toFixed(1);
            byId("annotation").style.display = "none";
            byId("report").style.display = "block";
            byId("agreement").textContent = percentage + "% agreement";
            byId("agreementDetail").textContent = matches + " of " + items.length + " human scores exactly matched the evaluator.";
            downloadAnnotations();
            window.scrollTo({ top: 0, behavior: "smooth" });
            });
            byId("download").addEventListener("click", downloadAnnotations);

            updateProgress();
            render();
        </script>
    </body>
</html>
"""


def load_rows(path: Path, score_count: int) -> list[tuple[str, str, str, int]]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}") from error
            missing = [
                key
                for key in (
                    "question",
                    "answer",
                    "model_final_answer",
                    "response_level_score",
                )
                if key not in row
            ]
            if missing:
                raise ValueError(
                    f"Line {line_number} is missing: {', '.join(missing)}"
                )
            score = row["response_level_score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"Line {line_number} has a non-numeric score: {score!r}")
            if not float(score).is_integer() or not 0 <= int(score) < score_count:
                raise ValueError(
                    f"Line {line_number} has score {score!r}; expected an integer "
                    f"from 0 to {score_count - 1}."
                )
            rows.append(
                (
                    str(row["question"]),
                    str(row["answer"]),
                    str(row["model_final_answer"]),
                    int(score),
                )
            )
    return rows


def json_for_html(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def build_html(
    rows: list[tuple[str, str, str, int]],
    question_type: str,
    storage_key: str,
    output_filename: str,
) -> str:
    items = [
        [question, ground_truth, model_answer, score + 19]
        for question, ground_truth, model_answer, score in rows
    ]
    replacements = {
        "__TYPE__": question_type,
        "__COUNT__": str(len(rows)),
        "__ITEMS__": json_for_html(items),
        "__RUBRIC__": json_for_html(RUBRICS[question_type]),
        "__FOCUS__": json_for_html(FOCUS[question_type]),
        "__SCORE_COUNT__": str(len(RUBRICS[question_type])),
        "__STORAGE_KEY__": json_for_html(storage_key),
        "__OUTPUT_FILENAME__": json_for_html(output_filename),
    }
    html = HTML_TEMPLATE
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an offline HTML annotator for response-level scores."
    )
    parser.add_argument("--file", required=True, type=Path, help="Evaluated JSONL file.")
    parser.add_argument(
        "--question_type",
        required=True,
        type=str.upper,
        choices=sorted(RUBRICS),
        help="Whether the examples are FPQ or TPQ.",
    )
    parser.add_argument("--n", type=int, default=50, help="Number of random samples.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path. Defaults beside the input JSONL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducible sampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be greater than zero.")
    score_count = len(RUBRICS[args.question_type])
    rows = load_rows(args.file, score_count)
    if args.n > len(rows):
        raise ValueError(
            f"Requested {args.n} samples, but {args.file} contains only {len(rows)}."
        )
    samples = random.Random(args.seed).sample(rows, args.n)
    out = args.out or args.file.with_name(
        f"{args.file.stem}_manual_annotation_{args.question_type.lower()}.html"
    )
    annotation_id = hashlib.sha256(
        (
            f"{args.file.resolve()}:{args.question_type}:{args.n}:{args.seed}"
        ).encode()
    ).hexdigest()[:16]
    storage_key = (
        f"manual-annotation:{annotation_id}"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_html(
            samples,
            args.question_type,
            storage_key,
            out.with_suffix(".jsonl").name,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(samples)} sampled {args.question_type} examples to {out}")


if __name__ == "__main__":
    main()
