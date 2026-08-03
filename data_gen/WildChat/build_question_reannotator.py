"""
Build an offline blind reannotator for WildChat question annotations.

Example:
    python data_gen/WildChat/build_question_reannotator.py \
        --file /path/to/wildchat_annotator_annotated.jsonl \
        --index_slice 0:50 \
        --out /path/to/wildchat_reannotator.html \
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
    <title>WildChat blind question reannotation</title>
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
        aside { align-self: start; position: sticky; top: 78px; padding: 24px; }
        h2 { margin: 8px 0 18px; font-size: 23px; }
        h3 { margin: 20px 0 8px; font-size: 16px; }
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
        .counter { color: var(--muted); font-size: 14px; }
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
        .buttons button.selected { outline: 4px solid #9eafe8; outline-offset: 2px; }
        .navigation, .report-actions {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            margin-top: 22px;
        }
        .navigation button, #review { background: var(--invalid); }
        .navigation button:disabled { opacity: .4; cursor: not-allowed; }
        #download { background: var(--accent); }
        .hint { margin: 16px 0 0; color: var(--muted); font-size: 13px; }
        #report { display: none; text-align: center; }
        #report h2 { font-size: 30px; }
        #agreement { color: var(--good); font-size: clamp(42px, 9vw, 82px); font-weight: 800; }
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
        <h1>WildChat blind reannotation</h1>
        <progress id="progress" value="0" max="__COUNT__"></progress>
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
                <li>Select <strong>No</strong> if the question has no false presupposition or does not presuppose anything.</li>
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
            <div class="navigation">
                <button id="previous" type="button">Previous question</button>
                <button id="next" type="button">Next question</button>
            </div>
            <p class="hint">
                The prior annotation is hidden. Discard is a third annotation option
                and counts toward agreement. Progress is saved in this browser.
            </p>
        </section>
        <section id="report" class="panel">
            <h2>Reannotation complete</h2>
            <div id="agreement"></div>
            <div id="summary"></div>
            <div class="report-actions">
                <button id="review" type="button">Review annotations</button>
                <button id="download" type="button">Download annotations again</button>
            </div>
        </section>
    </main>
    <script>
        const items = __ITEMS__;
        const storageKey = __STORAGE_KEY__;
        const outputFilename = __OUTPUT_FILENAME__;
        let state = { cursor: 0, annotations: [], discarded: 0 };

        try {
            const saved = JSON.parse(localStorage.getItem(storageKey));
            if (
                saved &&
                Number.isInteger(saved.cursor) &&
                Array.isArray(saved.annotations) &&
                Number.isInteger(saved.discarded)
            ) {
                state = saved;
            }
        } catch (_) {
            state = { cursor: 0, annotations: [], discarded: 0 };
        }

        const byId = (id) => document.getElementById(id);

        function saveState() {
            localStorage.setItem(storageKey, JSON.stringify(state));
        }

        function updateProgress() {
            byId("progress").value = state.annotations.length;
            byId("progressText").textContent =
                state.annotations.length + " / " + items.length + " reviewed · " +
                state.discarded + " discarded";
        }

        function outputRecords() {
            return state.annotations.map((annotation) => {
                const item = items[annotation.index];
                const reannotation = annotation.value;
                return {
                    ...item.record,
                    has_false_presupposition_reannotation: reannotation,
                    reannotation_discarded: reannotation === null,
                    annotations_agree: item.original === reannotation,
                };
            });
        }

        function downloadAnnotations() {
            const jsonl = outputRecords()
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

        function finish() {
            const records = outputRecords();
            const matches = records.filter((record) => record.annotations_agree).length;
            const percentage = records.length
                ? (100 * matches / records.length).toFixed(1)
                : "N/A";
            byId("annotation").style.display = "none";
            byId("report").style.display = "block";
            byId("agreement").textContent =
                percentage === "N/A" ? "No comparable questions" : percentage + "% agreement";
            byId("summary").textContent =
                matches + " of " + records.length + " questions matched the prior annotation. " +
                state.discarded + " discard annotations were included as disagreements.";
            updateProgress();
            downloadAnnotations();
        }

        function render() {
            updateProgress();
            if (state.cursor >= items.length) {
                finish();
                return;
            }
            byId("counter").textContent =
                "Question " + (state.cursor + 1) + " of " + items.length;
            byId("question").textContent = items[state.cursor].record.question;
            const existing = state.annotations.find(
                (annotation) => annotation.index === state.cursor
            );
            byId("yes").classList.toggle("selected", existing?.value === true);
            byId("no").classList.toggle("selected", existing?.value === false);
            byId("invalid").classList.toggle("selected", existing?.value === null);
            byId("previous").disabled = state.cursor === 0;
            byId("next").disabled = state.cursor >= state.annotations.length;
        }

        function annotate(value) {
            const existingIndex = state.annotations.findIndex(
                (annotation) => annotation.index === state.cursor
            );
            const annotation = { index: state.cursor, value };
            if (existingIndex === -1) {
                state.annotations.push(annotation);
            } else {
                state.annotations[existingIndex] = annotation;
            }
            state.discarded = state.annotations.filter(
                (saved) => saved.value === null
            ).length;
            state.cursor += 1;
            saveState();
            render();
        }

        byId("yes").addEventListener("click", () => annotate(true));
        byId("no").addEventListener("click", () => annotate(false));
        byId("invalid").addEventListener("click", () => annotate(null));
        byId("previous").addEventListener("click", () => {
            if (state.cursor === 0) return;
            state.cursor -= 1;
            saveState();
            render();
        });
        byId("next").addEventListener("click", () => {
            if (state.cursor >= state.annotations.length) return;
            state.cursor += 1;
            saveState();
            render();
        });
        byId("review").addEventListener("click", () => {
            state.cursor = items.length - 1;
            byId("report").style.display = "none";
            byId("annotation").style.display = "block";
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
            annotation = record.get("has_false_presupposition")
            if not isinstance(annotation, bool):
                raise ValueError(
                    f"Line {line_number} has no boolean 'has_false_presupposition'."
                )
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no annotated records.")
    return records


def json_for_html(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def parse_index_slice(value: str) -> slice:
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "expected START:STOP, such as 0:50, 50:, or :50"
        )
    try:
        start, stop = (int(part) if part else None for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "slice bounds must be integers, such as 0:50 or 50:"
        ) from error
    return slice(start, stop)


def build_html(
    records: list[dict[str, object]],
    storage_key: str,
    output_filename: str,
) -> str:
    items = [
        {
            "record": record,
            "original": record["has_false_presupposition"],
        }
        for record in records
    ]
    replacements = {
        "__ITEMS__": json_for_html(items),
        "__COUNT__": str(len(records)),
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
            "Build an offline blind reannotator that compares false-presupposition "
            "labels with annotations from build_question_annotator.py."
        )
    )
    parser.add_argument(
        "--file",
        required=True,
        type=Path,
        help="Annotated JSONL produced by build_question_annotator.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output HTML path. Defaults beside the input JSONL.",
    )
    parser.add_argument(
        "--index_slice",
        type=parse_index_slice,
        default=None,
        metavar="START:STOP",
        help=(
            "Optional Python-style index slice applied in file order before "
            "shuffling, for example 0:50 or 50:."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducibly shuffling question order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.file.suffix.lower() != ".jsonl":
        raise ValueError(f"--file must end with .jsonl: {args.file}")

    records = load_records(args.file)
    if args.index_slice is not None:
        records = records[args.index_slice]
        if not records:
            raise ValueError("--index_slice selected no records.")
    if args.seed is not None:
        random.Random(args.seed).shuffle(records)

    out = args.out or args.file.with_name(f"{args.file.stem}_reannotator.html")
    if out.suffix.lower() != ".html":
        raise ValueError(f"--out must end with .html: {out}")

    input_hash = hashlib.sha256(args.file.read_bytes()).hexdigest()[:16]
    annotation_id = hashlib.sha256(
        (
            f"v3:{args.file.resolve()}:{input_hash}:{args.index_slice}:"
            f"{args.seed}"
        ).encode()
    ).hexdigest()[:16]
    storage_key = f"wildchat-reannotation:{annotation_id}"
    output_filename = f"{out.stem}_annotated.jsonl"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        build_html(records, storage_key, output_filename),
        encoding="utf-8",
    )
    print(
        f"Wrote a blind reannotator with {len(records)} questions to {out}. "
        "Agreement will be shown after every question is reviewed."
    )


if __name__ == "__main__":
    main()
