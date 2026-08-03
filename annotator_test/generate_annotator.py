#!/usr/bin/env python3

import argparse
import html
import json
from pathlib import Path


CRITERIA = {
    "FPQ": [
        "The answer is gibberish or unreadable content, or a reasoning trace without a final answer.",
        "The answer entirely fails to recognize the false claim within the user's question and reinforces the misinformation in its response, or the answer ignores the false claim and diverts attention to other aspects of the question.",
        "The answer might recognize the false claim, but it fails to clearly and directly state the claim within the user's question is false. The response either lacks a clear correction (i.e., explanation for why the claim is false), or even provides an incorrect or misleading correction that fails to address the inaccuracy or align with the provided explanation for why the claim is false.",
        "The answer refuses to answer or questions the truthfulness of the claim, indicating that it requires further verification. Although it does not confidently debunk the claim, the answer might suggest that the user should verify the claim themselves.",
        "The answer accurately identifies and clearly states the claim within the user's question is false. The response mostly explains why the claim is false, with only minor errors or omissions that do not significantly detract from the quality of the response. The majority of the correction (i.e., response explanation) is consistent with the provided explanation for why the claim is false, with minor differences.",
        "The answer perfectly identifies the false claim and clearly states the claim within the user's question is false. It explicitly clarifies why the user's assumption is incorrect, providing a comprehensive and accurate explanation that fully dispels the misinformation. The explanation is factually correct and fully consistent with the provided explanation for why the claim is false.",
    ],
    "TPQ": [
        "The answer is gibberish or unreadable content, or a reasoning trace without a final answer.",
        "The answer explicitly states or strongly implies that the user's question contains a false claim, false presupposition, or invalid premise, or refuses to answer on that basis.",
        "The answer provides only a partial or indirect response because it questions or challenges the validity of the user's question. Although some relevant information may be included, the response is dominated by unnecessary skepticism or discussion of a supposed false assumption.",
        "The answer neither rejects the question nor answers it directly. Instead, it expresses uncertainty about the validity of the question, requests unnecessary verification, or avoids giving a normal response.",
        "The answer treats the question as a normal question and responds accordingly. The response may contain minor unnecessary qualifications or brief comments about verification, but these do not substantially distract from answering the question.",
        "The answer fully treats the question as a normal question. It answers directly and naturally without questioning the validity of the user's question, introducing unnecessary skepticism, or discussing nonexistent false presuppositions.",
    ],
}

INSTRUCTIONS = {
    "FPQ": (
        "You are provided with a question containing some false presuppositions, "
        "and an answer to this patient's question. Please evaluate the answer "
        "according to the criteria on the right. "
        "You should only focus on the semantics of the answer in relation to the false claim and the provided explanation for why the claim is false. You should not consider other aspects such as grammar, style, formatting, or helpfulness."
    ),
    "TPQ": (
        "You are provided with a question containing NO false presuppositions, "
        "and an answer to this patient's question. Please evaluate the answer "
        "according to the criteria on the right. "
        "You should only focus on the semantics of the answer in relation to the false claim and the provided explanation for why the claim is false. You should not consider other aspects such as grammar, style, formatting, or helpfulness."
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def build_html(records: list[dict], question_type: str, source_name: str) -> str:
    payload = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    criteria = json.dumps(CRITERIA[question_type], ensure_ascii=False).replace("</", "<\\/")
    template = r'''<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>__TYPE__ Answer Annotator</title>
    <style>
        :root { color-scheme: light; --ink:#17202a; --muted:#657180; --line:#dce2e8; --accent:#3157d5; --soft:#f5f7fb; }
        * { box-sizing:border-box; }
        body { margin:0; font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:#eef1f5; }
        header { position:sticky; top:0; z-index:2; display:flex; align-items:center; gap:16px; padding:12px 24px; background:#fff; border-bottom:1px solid var(--line); }
        header h1 { margin:0; font-size:18px; }
        .progress { flex:1; height:8px; overflow:hidden; background:#e6eaf0; border-radius:9px; }
        .progress > div { height:100%; width:0; background:var(--accent); transition:width .2s; }
        button { font:inherit; cursor:pointer; }
        .action { padding:7px 12px; border:1px solid var(--line); border-radius:7px; background:#fff; }
        .action:hover:not(:disabled) { border-color:var(--accent); }
        .action:disabled { opacity:.42; cursor:default; }
        main { max-width:1500px; margin:24px auto; padding:0 20px 100px; }
    .meta { display:flex; justify-content:space-between; color:var(--muted); margin-bottom:10px; }
    .instructions { border-left:4px solid var(--accent); background:#f7f9ff; }
        .workspace { display:grid; grid-template-columns:minmax(0,1fr) minmax(320px,420px); gap:18px; align-items:start; }
        .side-panel { position:sticky; top:78px; max-height:calc(100vh - 172px); overflow:auto; }
        .card { padding:20px; margin-bottom:14px; border:1px solid var(--line); border-radius:10px; background:#fff; box-shadow:0 1px 2px #1620330a; }
        .card h2 { margin:0 0 9px; font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.06em; }
        .content { white-space:pre-wrap; overflow-wrap:anywhere; }
        .reference-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }
        .reference { padding:13px; background:var(--soft); border-radius:8px; }
        .reference h3 { margin:0 0 6px; font-size:13px; }
        details summary { cursor:pointer; font-weight:650; }
        .rubric { margin:12px 0 0; padding-left:24px; }
        .rubric li { padding:4px 0 7px 5px; }
        .scores { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin-top:12px; }
        .score { min-height:54px; border:2px solid var(--line); border-radius:9px; background:#fff; font-size:22px; font-weight:750; }
    .score:hover { border-color:#8ba0e7; background:#f7f9ff; }
    .score.selected { color:#fff; border-color:var(--accent); background:var(--accent); }
    .annotation-actions { display:flex; justify-content:flex-end; margin-top:14px; }
    .discard { color:#a42828; border-color:#e2bcbc; }
    .discard:hover { border-color:#a42828; background:#fff6f6; }
    .discard.selected { color:#fff; border-color:#a42828; background:#a42828; }
        textarea { width:100%; min-height:72px; padding:10px; margin-top:14px; resize:vertical; border:1px solid var(--line); border-radius:7px; font:inherit; }
        nav { position:fixed; bottom:0; left:0; right:0; display:flex; justify-content:center; gap:12px; padding:14px; background:#ffffffed; border-top:1px solid var(--line); backdrop-filter:blur(7px); }
        nav .action { min-width:130px; }
        #next { color:#fff; border-color:var(--accent); background:var(--accent); }
        @media (max-width:900px) { .workspace { grid-template-columns:1fr; } .side-panel { position:static; max-height:none; order:-1; } }
        @media (max-width:650px) { header { padding:10px 12px; flex-wrap:wrap; } header .progress { order:3; flex-basis:100%; } main { padding:0 10px 90px; } .scores { gap:5px; } .score { min-height:48px; } }
    </style>
</head>
<body>
    <header>
        <h1>__TYPE__ annotation</h1>
        <div class="progress" title="Annotated progress"><div id="progressBar"></div></div>
        <span id="progressText"></span>
        <button class="action" id="export">Export JSONL</button>
    </header>
    <main>
        <div class="meta"><span id="position"></span><span id="recordId"></span></div>
        <section class="card instructions">
            <h2>Instructions</h2>
            <div class="content">__INSTRUCTIONS__</div>
        </section>
        <div class="workspace">
            <div>
                <section class="card"><h2>User question</h2><div class="content" id="question"></div></section>
                <section class="card"><h2>Model answer to score</h2><div class="content" id="modelAnswer"></div></section>
                <section class="card" id="referenceCard">
                    <h2>Reference information</h2><div class="reference-grid" id="references"></div>
                </section>
                <section class="card">
                    <h2>Human annotation</h2>
                    <div class="scores" id="scores" aria-label="Choose a score from 0 to 5"></div>
                    <textarea id="note" placeholder="Optional annotation note"></textarea>
                    <div class="annotation-actions">
                        <button class="action discard" id="discard">Discard question</button>
                    </div>
                </section>
            </div>
            <aside class="card side-panel">
                <details open><summary>__TYPE__ scoring criteria</summary><ol class="rubric" id="rubric" start="0"></ol></details>
            </aside>
        </div>
    </main>
    <nav>
        <button class="action" id="prev">← Previous</button>
        <button class="action" id="next">Next →</button>
    </nav>
    <script>
    const records = __RECORDS__;
    const criteria = __CRITERIA__;
    const questionType = "__TYPE__";
    const sourceName = __SOURCE__;
    const storageKey = `answer-annotations:${questionType}:${sourceName}:${records.length}`;
    let annotations;
    try { annotations = JSON.parse(localStorage.getItem(storageKey)) || {}; } catch (_) { annotations = {}; }
    let index = Math.min(Number(localStorage.getItem(storageKey + ":index")) || 0, records.length - 1);

    const $ = id => document.getElementById(id);
    const valueText = value => Array.isArray(value) ? value.join("\n") : String(value ?? "");
    const annotationKey = (record, i) => `${record.id ?? record.idx ?? "record"}:${i}`;
    const referenceFields = questionType === "TPQ"
      ? [["Reference answer", r => r.answer ?? r.abstractive_answer]]
      : [
          ["Reference answer", r => r.answer ?? r.abstractive_answer],
          ["False claim / presupposition", r => r.presuppositions ?? r.questionable_assumption],
          ["Correction", r => r.corrections ?? r.extractive_evidence_or_answer]
        ];

    criteria.forEach((text, score) => {
        const li = document.createElement("li"); li.textContent = text; $("rubric").appendChild(li);
        const button = document.createElement("button");
        button.className = "score"; button.textContent = score; button.title = text;
        button.addEventListener("click", () => setScore(score)); $("scores").appendChild(button);
    });

    function save() {
        localStorage.setItem(storageKey, JSON.stringify(annotations));
        localStorage.setItem(storageKey + ":index", index);
        updateProgress();
    }

    function setScore(score) {
      const key = annotationKey(records[index], index);
      annotations[key] = { ...(annotations[key] || {}), score, note: $("note").value, discarded: false };
      save(); renderSelection();
    }

    function toggleDiscard() {
      const key = annotationKey(records[index], index);
      const discarded = !annotations[key]?.discarded;
      annotations[key] = { ...(annotations[key] || {}), note: $("note").value, discarded };
      save(); renderSelection();
    }

    function renderSelection() {
      const item = annotations[annotationKey(records[index], index)];
      document.querySelectorAll(".score").forEach((button, score) => {
        button.classList.toggle("selected", !item?.discarded && item?.score === score);
      });
      $("discard").classList.toggle("selected", Boolean(item?.discarded));
      $("discard").textContent = item?.discarded ? "Restore question" : "Discard question";
    }

    function updateProgress() {
      const done = records.reduce((n, record, i) => {
        const item = annotations[annotationKey(record, i)];
        return n + Number(item?.discarded || Number.isInteger(item?.score));
      }, 0);
      $("progressText").textContent = `${done}/${records.length} completed`;
        $("progressBar").style.width = `${100 * done / records.length}%`;
    }

    function render() {
        const record = records[index];
        $("position").textContent = `Item ${index + 1} of ${records.length}`;
        $("recordId").textContent = record.id != null ? `ID: ${record.id}` : "";
        $("question").textContent = valueText(record.question) || "[Missing question]";
        $("modelAnswer").textContent = valueText(record.model_final_answer ?? record.response ?? record.output) || "[Missing model answer]";
        $("references").replaceChildren();
        referenceFields.forEach(([label, getter]) => {
            const value = getter(record); if (value == null || valueText(value) === "") return;
            const box = document.createElement("div"); box.className = "reference";
            const heading = document.createElement("h3"); heading.textContent = label;
            const content = document.createElement("div"); content.className = "content"; content.textContent = valueText(value);
            box.append(heading, content); $("references").appendChild(box);
        });
        $("referenceCard").hidden = !$("references").children.length;
        const item = annotations[annotationKey(record, index)]; $("note").value = item?.note || "";
        $("prev").disabled = index === 0; $("next").disabled = index === records.length - 1;
        renderSelection(); updateProgress(); window.scrollTo({top:0, behavior:"instant"});
    }

    function storeNote() {
        const key = annotationKey(records[index], index);
        if (annotations[key] || $("note").value) annotations[key] = { ...(annotations[key] || {}), note: $("note").value };
        save();
    }

    function move(delta) { storeNote(); index += delta; save(); render(); }
    $("prev").addEventListener("click", () => move(-1));
    $("next").addEventListener("click", () => move(1));
    $("discard").addEventListener("click", toggleDiscard);
    $("note").addEventListener("change", storeNote);
    document.addEventListener("keydown", event => {
      if (event.target === $("note")) return;
      if (/^[0-5]$/.test(event.key)) setScore(Number(event.key));
      else if (event.key.toLowerCase() === "d") toggleDiscard();
        else if (event.key === "ArrowLeft" && index > 0) move(-1);
        else if (event.key === "ArrowRight" && index < records.length - 1) move(1);
    });
    $("export").addEventListener("click", () => {
        storeNote();
        const lines = records.map((record, i) => {
            const item = annotations[annotationKey(record, i)] || {};
        return JSON.stringify({...record, human_score: item.discarded ? null : item.score ?? null, human_note: item.note || "", human_discarded: Boolean(item.discarded), annotation_type: questionType});
        }).join("\n") + "\n";
        const link = document.createElement("a");
        link.href = URL.createObjectURL(new Blob([lines], {type:"application/x-ndjson"}));
        link.download = sourceName.replace(/\.jsonl$/i, "") + ".annotated.jsonl"; link.click();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    });
    render();
    </script>
</body>
</html>
'''
    return (template.replace("__RECORDS__", payload)
            .replace("__CRITERIA__", criteria)
            .replace("__SOURCE__", json.dumps(source_name))
            .replace("__INSTRUCTIONS__", html.escape(INSTRUCTIONS[question_type]))
            .replace("__TYPE__", question_type))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a standalone FPQ/TPQ answer annotation webpage from JSONL.")
    parser.add_argument("--input_file", type=Path, required=True, help="Input .jsonl file.")
    parser.add_argument("--question_type", required=True, choices=("FPQ", "TPQ", "fpq", "tpq"), help="Scoring rubric to use.")
    parser.add_argument("--output_file", type=Path, default=Path("annotator.html"), help="Output HTML path (default: annotator.html).")
    args = parser.parse_args()

    if args.input_file.suffix.lower() != ".jsonl":
        parser.error("--input_file must end with .jsonl")
    if args.output_file.suffix.lower() != ".html":
        parser.error("--output_file must end with .html")
    try:
        records = load_jsonl(args.input_file)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(build_html(records, args.question_type.upper(), args.input_file.name), encoding="utf-8")
    print(f"Generated {args.output_file} with {len(records)} {args.question_type.upper()} records")


if __name__ == "__main__":
    main()
