#!/usr/bin/env python3
"""Generate a self-contained browser UI for annotating CancerMythNFP JSONL."""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number} must contain a JSON object.")
            if "question" not in row:
                raise ValueError(f"Line {line_number} is missing: question")
            rows.append(row)
    if not rows:
        raise ValueError(f"No datapoints found in {path}")
    return rows


def render_html(rows: list[dict], source: Path, start_idx: int, end_idx: int) -> str:
    # Prevent dataset text containing an HTML closing tag from ending this script block.
    embedded_data = json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    source_name = json.dumps(source.name)
    return f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>CancerMythNFP annotation [{start_idx}:{end_idx}] — {source.name}</title>
    <style>
        :root {{ color-scheme: light; --ink:#17202a; --muted:#667085; --line:#d0d5dd; --accent:#175cd3; }}
        * {{ box-sizing:border-box; }}
        body {{ margin:0; background:#f5f7fa; color:var(--ink); font:16px/1.55 system-ui,sans-serif; }}
        main {{ max-width:960px; margin:auto; padding:28px 20px 110px; }}
        header {{ display:flex; justify-content:space-between; gap:20px; align-items:center; margin-bottom:20px; }}
        h1 {{ margin:0; font-size:1.35rem; }}
        #progress {{ color:var(--muted); white-space:nowrap; }}
        .bar {{ height:8px; background:#e4e7ec; border-radius:8px; overflow:hidden; margin:12px 0 24px; }}
        .bar span {{ display:block; height:100%; background:#12b76a; transition:width .2s; }}
        section {{ background:white; border:1px solid var(--line); border-radius:12px; padding:20px; margin:14px 0; }}
        h2, label {{ display:block; margin:0 0 8px; font-size:.85rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; font-weight:700; }}
        .content {{ white-space:pre-wrap; overflow-wrap:anywhere; }}
        textarea {{ width:100%; min-height:100px; resize:vertical; padding:12px; border:1px solid #98a2b3; border-radius:8px; font:inherit; }}
        textarea:focus {{ outline:3px solid #d1e9ff; border-color:var(--accent); }}
        .annotation-row {{ display:flex; flex-direction:column; gap:8px; margin-top:16px; }}
        .annotation-controls {{ display:flex; justify-content:space-between; align-items:center; gap:12px; }}
        .suggestion-label {{ display:flex; align-items:center; gap:7px; color:var(--ink); text-transform:none; letter-spacing:0; font-size:.9rem; cursor:pointer; }}
        .suggestion-label input {{ width:18px; height:18px; }}
        .remove {{ padding:8px 12px; color:#b42318; }}
        .add {{ margin-top:12px; color:var(--accent); border-color:var(--accent); }}
        .checkbox-label {{ display:flex; align-items:center; gap:10px; color:var(--ink); text-transform:none; letter-spacing:0; font-size:1rem; cursor:pointer; }}
        .checkbox-label input {{ width:20px; height:20px; }}
        footer {{ position:fixed; bottom:0; left:0; right:0; background:rgba(255,255,255,.96); border-top:1px solid var(--line); padding:14px 20px; }}
        .actions {{ max-width:920px; margin:auto; display:flex; gap:10px; align-items:center; }}
        button {{ border:1px solid #98a2b3; background:white; border-radius:8px; padding:10px 16px; cursor:pointer; font-weight:650; }}
        button.primary {{ background:var(--accent); border-color:var(--accent); color:white; }}
        button:disabled {{ opacity:.45; cursor:not-allowed; }}
        #status {{ color:var(--muted); margin-left:auto; font-size:.9rem; }}
        @media(max-width:600px) {{ .actions {{ flex-wrap:wrap; }} #status {{ width:100%; margin:0; }} }}
    </style>
</head>
<body>
    <main>
        <header><h1>CancerMythNFP annotation [{start_idx}:{end_idx}]</h1><div id="progress"></div></header>
        <section>
            <h2>Instructions</h2>
            <div class="content">For each question, write the presuppositions implied by the question. You are given the example assumption as well, which is the annotated presupposition + correction.
            1. Each presupposition should be a single line of text without additional context or explanation.
            2. Please begin your presupposition with "presupposes that ..."
            3. Please check "This presupposition is explicitly suggested by a doctor in the question" only when applicable
            4. Use Add presupposition when more than one is needed\nBelow are some examples:
            [Question] I've been diagnosed with early-stage hypopharyngeal cancer, and my doctor has suggested surgery as part of the treatment plan. I'm concerned about the potential for rapid metastasis post-surgery. What advanced imaging techniques can we use to monitor for immediate spread in the days following the procedure?
            [Presupposition] presupposes that surgery for hypopharyngeal cancer leads to rapid metastasis
            [Suggested by a doctor in the question?] No
            [Explanation] The question implies a concern about rapid metastasis post-surgery, but it is not explicitly stated by the doctor. It comes from the patient's concern, not the doctor's suggestion.

            [Question] My mother has an islet cell tumor and is undergoing chemotherapy. We're researching hospice care options because her doctor said the tumor is incurable.  We want to make sure she's comfortable in her final months. Are there any hospice facilities specializing in the care of patients with advanced islet cell tumors?
            [Presupposition] presupposes that islet cell tumors are incurable
            [Suggested by a doctor in the question?] Yes
            [Explanation] The question states that the doctor said the tumor is incurable, which is an explicit suggestion from the doctor. Therefore, this presupposition is explicitly suggested by a doctor in the question.

            [Question] My sister has metastatic dysgerminoma and is about to start radiation therapy.  She's worried about skin irritation and hair loss during treatment. Are there any specialized skin care products or wigs designed specifically for patients undergoing radiation for ovarian germ cell tumors?
            [Presupposition] presupposes that radiation therapy is the standard treatment for metastatic dysgerminoma
            [Suggested by a doctor in the question?] Yes
            [Explanation] Though not directly stated, we can infer that the radiation therapy is prescribed by the doctor for this patient. Therefore, this presupposition is explicitly suggested by a doctor in the question.
            
            [Question] My uncle has transitional cell carcinoma of the ureter, and the doctor said it's advanced. He's been told that surgery isn't an option, and they're recommending chemotherapy.  Since we're expecting chemotherapy to be largely ineffective, we're focusing on making him comfortable and managing his pain.  Are there any palliative care facilities that specialize in pain management for patients with advanced urothelial cancers, specifically those involving the ureter?
            [Presupposition 1] presupposes that surgery is not an option for advanced transitional cell carcinoma of the ureter
            [Suggested by a doctor in the question?] Yes
            [Explanation] The question states that the doctor said surgery isn't an option, which is an explicit suggestion from the doctor. Therefore, this presupposition is explicitly suggested by a doctor in the question.
            [Presupposition 2] presupposes that chemotherapy is largely ineffective for advanced transitional cell carcinoma of the ureter
            [Suggested by a doctor in the question?] No
            [Explanation] The question implies that chemotherapy is expected to be largely ineffective, but it is not explicitly stated by the doctor. It comes from the patient's expectation, not the doctor's suggestion.
            </div>
            </div>
        </section>
        <div class="bar"><span id="bar"></span></div>
        <section><h2>Question</h2><div id="question" class="content"></div></section>
        <section><h2>Source cancer</h2><div id="cancer" class="content"></div></section>
        <section>
            <label>Presuppositions (one per box)</label>
            <div id="annotations"></div>
            <button id="add" class="add" type="button">+ Add presupposition</button>
        </section>
    </main>
    <footer><div class="actions">
        <button id="previous">← Previous</button>
        <button id="next" class="primary">Save &amp; Next →</button>
        <button id="download">Download JSONL</button>
        <span id="status">Changes are saved in this browser.</span>
    </div></footer>
    <script>
        const original = {embedded_data};
        const sourceName = {source_name};
        const storageKey = "CancerMythNFP-annotation:" + sourceName + ":{start_idx}:{end_idx}";
        let state;
        try {{ state = JSON.parse(localStorage.getItem(storageKey)); }} catch (_) {{ state = null; }}
        if (!state || !Array.isArray(state.annotations) || state.annotations.length !== original.length) {{
            state = {{ index: 0, annotations: original.map(row =>
                Array.isArray(row.presuppositions) && row.presuppositions.length
                    ? row.presuppositions.map(String) : [""]),
                doctorSuggestions: original.map(row => {{
                    const count = Array.isArray(row.presuppositions) && row.presuppositions.length
                        ? row.presuppositions.length : 1;
                    if (Array.isArray(row.doctor_suggestion))
                        return Array.from({{length: count}}, (_, i) => Boolean(row.doctor_suggestion[i]));
                    return Array.from({{length: count}}, (_, i) => i === 0 && Boolean(row.doctor_suggestion));
                }}) }};
        }} else {{
            // Migrate progress saved by the older, single-presupposition annotator.
            state.annotations = state.annotations.map(value =>
                Array.isArray(value) ? value.map(String) : [String(value ?? "")]);
        }}
        if (!Array.isArray(state.doctorSuggestions) || state.doctorSuggestions.length !== original.length) {{
            const oldValues = Array.isArray(state.doctorSuggestion) ? state.doctorSuggestion : [];
            state.doctorSuggestions = state.annotations.map((values, i) =>
                values.map((_, j) => j === 0 && Boolean(oldValues[i])));
        }}
        state.index = Math.max(0, Math.min(original.length - 1, Number(state.index) || 0));

        const el = id => document.getElementById(id);
        function persist() {{
            state.annotations[state.index] = [...document.querySelectorAll(".annotation-input")]
                .map(input => input.value.replace(/[\\r\\n]+/g, " ").trim());
            state.doctorSuggestions[state.index] = [...document.querySelectorAll(".doctor-suggestion-input")]
                .map(input => input.checked);
            localStorage.setItem(storageKey, JSON.stringify(state));
            el("status").textContent = "Saved locally at " + new Date().toLocaleTimeString();
        }}
        function addInput(value = "", doctorSuggestion = false, focus = false) {{
            const row = document.createElement("div");
            row.className = "annotation-row";
            const input = document.createElement("textarea");
            input.className = "annotation-input";
            input.placeholder = "Write a presupposition here…";
            input.value = value;
            input.addEventListener("input", () => {{ clearTimeout(window.saveTimer); window.saveTimer = setTimeout(persist, 300); }});
            input.addEventListener("keydown", event => {{ if ((event.ctrlKey || event.metaKey) && event.key === "Enter") move(1); }});
            const remove = document.createElement("button");
            remove.type = "button";
            remove.className = "remove";
            remove.textContent = "Remove";
            remove.onclick = () => {{ row.remove(); if (!document.querySelector(".annotation-input")) addInput(); persist(); }};
            const suggestionLabel = document.createElement("label");
            suggestionLabel.className = "suggestion-label";
            const suggestion = document.createElement("input");
            suggestion.type = "checkbox";
            suggestion.className = "doctor-suggestion-input";
            suggestion.checked = doctorSuggestion;
            suggestion.onchange = persist;
            suggestionLabel.append(suggestion, "This presupposition is explicitly suggested by a doctor in the question");
            const controls = document.createElement("div");
            controls.className = "annotation-controls";
            controls.append(suggestionLabel, remove);
            row.append(input, controls);
            el("annotations").append(row);
            if (focus) input.focus();
        }}
        function render() {{
            const row = original[state.index];
            el("question").textContent = row.question ?? "";
            el("cancer").textContent = row.source_cancer ?? row.cancer ?? "(not provided)";
            el("annotations").replaceChildren();
            const values = state.annotations[state.index]?.length ? state.annotations[state.index] : [""];
            const suggestions = state.doctorSuggestions[state.index] || [];
            values.forEach((value, i) => addInput(value, Boolean(suggestions[i])));
            const done = state.annotations.filter(values => values.some(value => value.trim())).length;
            el("progress").textContent = `Datapoint ${{state.index + 1}} of ${{original.length}} · ${{done}} annotated`;
            el("bar").style.width = `${{100 * done / original.length}}%`;
            el("previous").disabled = state.index === 0;
            el("next").textContent = state.index === original.length - 1 ? "Save & Finish" : "Save & Next →";
            document.querySelector(".annotation-input").focus();
        }}
        function move(delta) {{
            persist(); state.index = Math.max(0, Math.min(original.length - 1, state.index + delta)); localStorage.setItem(storageKey, JSON.stringify(state)); render();
        }}
        function download() {{
            persist();
            const rows = original.map((row, i) => {{
                const entries = state.annotations[i].map((value, j) => ({{
                    value: value.trim(), doctorSuggestion: Boolean(state.doctorSuggestions[i][j])
                }})).filter(entry => entry.value);
                return {{
                    ...row,
                    presuppositions: entries.map(entry => entry.value),
                    doctor_suggestion: entries.map(entry => entry.doctorSuggestion)
                }};
            }});
            const jsonl = rows.map(row => JSON.stringify(row)).join("\\n") + "\\n";
            const blob = new Blob([jsonl], {{type:"application/x-ndjson;charset=utf-8"}});
            const link = document.createElement("a");
            link.href = URL.createObjectURL(blob);
            link.download = sourceName.replace(/\\.jsonl$/i, "") + "_annotated.jsonl";
            link.click();
            setTimeout(() => URL.revokeObjectURL(link.href), 1000);
        }}
        el("add").onclick = () => {{
            persist();
            state.annotations[state.index].push("");
            state.doctorSuggestions[state.index].push(false);
            addInput("", false, true);
        }};
        el("previous").onclick = () => move(-1);
        el("next").onclick = () => {{ if (state.index === original.length - 1) {{ persist(); download(); }} else move(1); }};
        el("download").onclick = download;
        window.addEventListener("beforeunload", persist);
        render();
    </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a self-contained HTML annotator from a JSONL file.")
    parser.add_argument("--file", required=True, type=Path, help="Preprocessed CancerMythNFP .jsonl file")
    parser.add_argument(
        "--output", type=Path, help="Output HTML path (default: <input>_<start>_<end>_annotator.html)"
    )
    parser.add_argument("--start_idx", type=int, default=0, help="Inclusive starting index (default: 0)")
    parser.add_argument(
        "--end_idx",
        type=int,
        default=None,
        help="Exclusive ending index; supports negative indexing (default: all)",
    )
    args = parser.parse_args()

    source = args.file.expanduser().resolve()
    if source.suffix.lower() != ".jsonl":
        parser.error("--file must point to a .jsonl file")
    if not source.is_file():
        parser.error(f"file does not exist: {source}")
    rows = load_jsonl(source)
    start_idx, end_idx, step = slice(args.start_idx, args.end_idx).indices(len(rows))
    selected_rows = rows[start_idx:end_idx:step]
    if not selected_rows:
        parser.error(
            f"selected interval [{start_idx}:{end_idx}] is empty for a dataset with {len(rows)} rows"
        )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else source.with_name(f"{source.stem}_{start_idx}_{end_idx}_annotator.html")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(selected_rows, source, start_idx, end_idx), encoding="utf-8")
    print(
        f"Wrote annotator for rows [{start_idx}:{end_idx}] "
        f"({len(selected_rows)} datapoints) to {output}"
    )


if __name__ == "__main__":
    main()
