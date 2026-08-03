"""
Build an offline annotator for preprocessed WildChat questions.

Example:
    python data_gen/WildChat/build_question_annotator.py \
        --file /path/to/wildchat_user_questions.jsonl \
        --n 100 \
        --total 500 \
        --out /path/to/wildchat_annotator.html \
        --seed 42
"""

import argparse
import hashlib
import json
import random
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>WildChat question annotation</title>
    <style>
        :root {
            color-scheme: light;
            --ink: #182033;
            --muted: #647187;
            --line: #d9e0e8;
            --paper: #fff;
            --wash: #f4f7fa;
            --yes: #b52e3b;
            --no: #247348;
            --invalid: #566274;
            --accent: #3158c9;
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
            z-index: 2;
            display: flex;
            align-items: center;
            gap: 18px;
            padding: 15px 24px;
            background: rgba(255, 255, 255, .97);
            border-bottom: 1px solid var(--line);
        }
        header h1 { margin: 0; font-size: 19px; white-space: nowrap; }
        progress { width: min(480px, 45vw); height: 13px; }
        #progressText { color: var(--muted); white-space: nowrap; }
        main {
            display: grid;
            grid-template-columns: minmax(260px, 340px) minmax(0, 900px);
            gap: 24px;
            max-width: 1280px;
            margin: 28px auto;
            padding: 0 20px 50px;
        }
        .panel {
            padding: 34px;
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 15px;
            box-shadow: 0 5px 22px rgba(25, 39, 67, .07);
        }
        .counter { color: var(--muted); font-size: 14px; }
        h2 { margin: 8px 0 18px; font-size: 23px; }
        h3 { margin: 20px 0 8px; font-size: 16px; }
        aside { align-self: start; position: sticky; top: 78px; padding: 24px; }
        aside h2 { margin-top: 0; }
        aside p { margin: 0 0 12px; }
        aside ul, aside ol { margin: 8px 0 0; padding-left: 22px; }
        aside li { margin-bottom: 8px; }
        .discard-guidance {
            margin-top: 20px;
            padding: 14px;
            border-left: 4px solid var(--invalid);
            background: #f1f3f6;
        }
        .discard-guidance h3 { margin-top: 0; }
        #question {
            min-height: 190px;
            max-height: 52vh;
            overflow-y: auto;
            padding: 21px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            background: #fbfcfe;
            border: 1px solid var(--line);
            border-radius: 10px;
            font-size: 18px;
        }
        .prompt { margin: 25px 0 12px; font-weight: 750; }
        .buttons { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        button {
            padding: 14px 18px;
            border: 0;
            border-radius: 9px;
            color: white;
            font: inherit;
            font-weight: 750;
            cursor: pointer;
        }
        button:hover { filter: brightness(.94); }
        button:focus-visible { outline: 3px solid #9eafe8; outline-offset: 2px; }
        #yes { background: var(--yes); }
        #no { background: var(--no); }
        #invalid { grid-column: 1 / -1; background: var(--invalid); }
        #download { margin-top: 22px; background: var(--accent); }
        .hint { margin: 16px 0 0; color: var(--muted); font-size: 13px; }
        #report { display: none; text-align: center; }
        #report h2 { font-size: 30px; }
        #summary { color: var(--muted); font-size: 18px; }
        @media (max-width: 650px) {
            header { flex-wrap: wrap; }
            main { grid-template-columns: 1fr; }
            aside { position: static; }
            .panel { padding: 22px; }
            .buttons { grid-template-columns: 1fr; }
            #invalid { grid-column: auto; }
        }
    </style>
</head>
<body>
    <header>
        <h1>WildChat annotation</h1>
        <progress id="progress" value="0" max="__TARGET__"></progress>
        <span id="progressText"></span>
    </header>
    <main>
        <aside class="panel" aria-labelledby="manual-title">
            <h2 id="manual-title">Instruction manual</h2>
            <p>
                Decide whether the question contains a <strong>false presupposition</strong>:
                a claim the question assumes to be true even though that claim is false.
            </p>
            <h3>How to annotate</h3>
            <ol>
                <li>Identify the claim or claims the question takes for granted.</li>
                <li>Select <strong>Yes</strong> if any required presupposition is false.</li>
                <li>Select <strong>No</strong> if the question has no false presupposition.</li>
            </ol>
            <div class="discard-guidance">
                <h3>When to discard</h3>
                <p>Select <strong>Discard</strong> if:</p>
                <ul>
                    <li>you cannot understand the datapoint;</li>
                    <li>the datapoint is not a question; or</li>
                    <li>judging whether it contains a false presupposition is beyond your capabilities.</li>
                </ul>
                <p>Do not guess when one of these conditions applies.</p>
            </div>
        </aside>
        <section id="annotation" class="panel">
            <div id="counter" class="counter"></div>
            <h2>Question</h2>
            <div id="question"></div>
            <div class="prompt">Does this question contain a false presupposition?</div>
            <div class="buttons">
                <button id="yes" type="button">Yes — false presupposition</button>
                <button id="no" type="button">No — no false presupposition</button>
                <button id="invalid" type="button">Discard this datapoint</button>
            </div>
            <p class="hint">
                Invalid questions do not count toward the target. Progress is saved in this browser.
            </p>
        </section>
        <section id="report" class="panel">
            <h2 id="reportTitle"></h2>
            <div id="summary"></div>
            <button id="download" type="button">Download annotations again</button>
        </section>
    </main>
    <script>
        const records = __RECORDS__;
        const target = __TARGET__;
        const storageKey = __STORAGE_KEY__;
        const outputFilename = __OUTPUT_FILENAME__;
        let state = { cursor: 0, valid: [], invalidCount: 0 };

        try {
        const saved = JSON.parse(localStorage.getItem(storageKey));
        if (
            saved &&
            Number.isInteger(saved.cursor) &&
            Array.isArray(saved.valid) &&
            Number.isInteger(saved.invalidCount)
        ) {
            state = saved;
        }
        } catch (_) {
            state = { cursor: 0, valid: [], invalidCount: 0 };
        }

        const byId = (id) => document.getElementById(id);

        function saveState() {
            localStorage.setItem(storageKey, JSON.stringify(state));
        }

        function updateProgress() {
        byId("progress").value = state.valid.length;
        byId("progressText").textContent =
            state.valid.length + " / " + target + " valid · " +
            state.invalidCount + " discarded";
        }

        function downloadAnnotations() {
            const jsonl = state.valid
                .slice(0, target)
                .map((record) => JSON.stringify(record))
                .join("\\n") + "\\n";
            const url = URL.createObjectURL(
                new Blob([jsonl], { type: "application/x-ndjson;charset=utf-8" })
            );
            const link = document.createElement("a");
            link.href = url;
            link.download = outputFilename;
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 0);
        }

        function finish(completed) {
            const positiveCount = state.valid.filter(
                (record) => record.has_false_presupposition === true
            ).length;
            const positiveRate = state.valid.length
                ? (100 * positiveCount / state.valid.length).toFixed(1)
                : "0.0";
            byId("annotation").style.display = "none";
            byId("report").style.display = "block";
            byId("reportTitle").textContent = completed
                ? "Annotation target complete"
                : "No questions remain";
            const completionSummary = completed
                ? target + " valid annotated questions are ready."
                : "Only " + state.valid.length + " valid questions were found after reviewing all " +
                records.length + " input records.";
            byId("summary").textContent =
                completionSummary + " " + positiveCount + " of " + state.valid.length +
                " valid questions contain a false presupposition (" + positiveRate + "%).";
            byId("download").textContent = completed
                ? "Download annotations again"
                : "Download partial annotations";
            updateProgress();
            downloadAnnotations();
        }

        function render() {
            updateProgress();
            if (state.valid.length >= target) {
                finish(true);
                return;
            }
            if (state.cursor >= records.length) {
                finish(false);
                return;
            }
            byId("counter").textContent =
                "Reviewed " + state.cursor + " of " + records.length +
                " available questions";
            byId("question").textContent = records[state.cursor].question;
        }

        function annotate(hasFalsePresupposition) {
            const record = {
                ...records[state.cursor],
                has_false_presupposition: hasFalsePresupposition,
            };
            state.valid.push(record);
            state.cursor += 1;
            saveState();
            render();
        }

        byId("yes").addEventListener("click", () => annotate(true));
        byId("no").addEventListener("click", () => annotate(false));
        byId("invalid").addEventListener("click", () => {
            state.invalidCount += 1;
            state.cursor += 1;
            saveState();
            render();
        });
        byId("download").addEventListener("click", downloadAnnotations);

        render();
    </script>
</body>
</html>
"""


def load_records(path: Path) -> list[dict[str, object]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Line {line_number} has no non-empty string 'question'."
                )
            records.append(record)
    return records


def json_for_html(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def build_html(
    records: list[dict[str, object]],
    target: int,
    storage_key: str,
    output_filename: str,
) -> str:
    replacements = {
        "__RECORDS__": json_for_html(records),
        "__TARGET__": str(target),
        "__STORAGE_KEY__": json_for_html(storage_key),
        "__OUTPUT_FILENAME__": json_for_html(output_filename),
    }
    html = HTML_TEMPLATE
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an offline annotator for false presuppositions in preprocessed "
            "WildChat questions."
        )
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Preprocessed WildChat .jsonl file.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        help="Number of valid questions to annotate.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=500,
        help="Number of input lines to sample into the annotator.",
    )
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
        help="Optional seed for reproducible question order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.file.suffix.lower() != ".jsonl":
        raise ValueError(f"--file must end with .jsonl: {args.file}")
    if args.n <= 0:
        raise ValueError("--n must be greater than zero.")
    if args.total <= 0:
        raise ValueError("--total must be greater than zero.")
    if args.n > args.total:
        raise ValueError("--n cannot be greater than --total.")

    records = load_records(args.file)
    if args.total > len(records):
        raise ValueError(
            f"Requested a sample of {args.total} questions, but the input has only "
            f"{len(records)} questions."
        )
    records = random.Random(args.seed).sample(records, args.total)

    out = args.out or args.file.with_name(f"{args.file.stem}_annotator.html")
    if out.suffix.lower() != ".html":
        raise ValueError(f"--out must end with .html: {out}")
    annotation_id = hashlib.sha256(
        f"{args.file.resolve()}:{args.n}:{args.total}:{args.seed}".encode()
    ).hexdigest()[:16]
    storage_key = f"wildchat-annotation:{annotation_id}"
    output_filename = f"{out.stem}_annotated.jsonl"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_html(records, args.n, storage_key, output_filename),
        encoding="utf-8",
    )
    print(
        f"Wrote an annotator with a sample of {len(records)} questions to {out}. "
        f"It will stop after {args.n} valid annotations."
    )


if __name__ == "__main__":
    main()
