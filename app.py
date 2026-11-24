"""
Local LLM vs Jev-style decisions — classroom demo
=================================================

Same local model (Qwen3-8B), two ways of using it:

1. Normal LLM way:  chat completion, the model *writes* an answer token by
   token (optionally with Qwen3 "thinking"). Confidence, if any, is just text.

2. Jev-style way:   no text generation. Every decision question is turned into
   a prompt whose next token is the answer (A/B/C... or Yes/No). The server
   runs ONE forward pass per question (all questions in parallel) and we read
   the probabilities of the allowed answer tokens directly. Output = typed
   answer + probability, like TypeSafe's Jev.

   NOTE: this *imitates* Jev's interface. It is not TypeSafe's model, and
   Qwen3's raw probabilities are not calibrated the way Jev claims to be.

Backends (BACKEND env var):
  llamacpp (default) — Mac / Apple Silicon, CPU, or any GPU:
      brew install llama.cpp
      llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M --jinja --reasoning-format deepseek \
          -c 16384 -np 4 --port 8080
  sglang — Linux + NVIDIA GPU:
      python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --reasoning-parser qwen3

Then:
    pip install flask requests
    python app.py              # http://127.0.0.1:5050
    MOCK=1 python app.py       # UI demo without a model (fake answers)
"""

import csv
import io
import json
import math
import os
import random
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests
from flask import Flask, Response, jsonify, render_template, request

BACKEND = os.environ.get("BACKEND", "llamacpp").lower()  # "llamacpp" or "sglang"
_DEFAULT_URL = "http://127.0.0.1:30000" if BACKEND == "sglang" else "http://127.0.0.1:8080"
SERVER_URL = os.environ.get("SERVER_URL", os.environ.get("SERVER_URL", _DEFAULT_URL)).rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
MOCK = os.environ.get("MOCK", "0") == "1"
DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "results.db"))
TIMEOUT = 900  # seconds; thinking mode can be slow on a laptop

LETTERS = "ABCDEFGH"  # up to 8 options per Choice / Score question
TOP_LOGPROBS = 20

app = Flask(__name__)


# ---------------------------------------------------------------- database --
def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                case_text TEXT NOT NULL,
                questions TEXT NOT NULL,
                settings TEXT NOT NULL,
                llm_result TEXT,
                jev_result TEXT,
                llm_seconds REAL,
                jev_seconds REAL
            )"""
        )


# ------------------------------------------------------------- validation --
def clean_questions(raw):
    """Keep only enabled, well-formed questions (max 3)."""
    out = []
    for q in (raw or [])[:3]:
        qtype = q.get("type", "choice")
        text = (q.get("text") or "").strip()
        if not text or qtype not in ("choice", "score", "noul"):
            continue
        options = [o.strip() for o in q.get("options", []) if o and o.strip()]
        if qtype == "noul":
            options = ["Yes", "No"]
        elif len(options) < 2:
            continue
        options = options[: len(LETTERS)]
        try:
            threshold = min(max(float(q.get("threshold", 0.8)), 0.0), 1.0)
        except (TypeError, ValueError):
            threshold = 0.8
        out.append({"type": qtype, "text": text, "options": options, "threshold": threshold})
    return out


# -------------------------------------------------------- normal LLM way --
def llm_messages(case_text, questions):
    lines = []
    for i, q in enumerate(questions, 1):
        if q["type"] == "noul":
            lines.append(f"{i}. {q['text']} (answer Yes or No)")
        elif q["type"] == "score":
            lines.append(f"{i}. {q['text']} (levels from lowest to highest: {', '.join(q['options'])})")
        else:
            lines.append(f"{i}. {q['text']} (options: {', '.join(q['options'])})")
    user = (
        f"Case:\n{case_text}\n\n"
        "Answer each question below. For each one give the answer, your confidence "
        "as a percentage, and one short sentence of reasoning.\n\n" + "\n".join(lines)
    )
    return [{"role": "user", "content": user}]


def split_thinking(content, reasoning):
    """Handle both: server-side reasoning parser on (reasoning separate) or off (<think> in text)."""
    content = content or ""
    if not reasoning:
        m = re.search(r"<think>(.*?)</think>", content, flags=re.S)
        if m:
            reasoning = m.group(1).strip()
            content = content[m.end():]
        elif "</think>" in content:  # template already opened <think>
            reasoning, content = content.split("</think>", 1)
    return content.strip(), (reasoning or "").strip()


def run_llm(case_text, questions, settings):
    payload = {
        "model": MODEL_NAME,
        "messages": llm_messages(case_text, questions),
        "temperature": settings["temperature"],
        "max_tokens": settings["max_tokens"],
        "chat_template_kwargs": {"enable_thinking": settings["thinking"]},
    }
    t0 = time.perf_counter()
    r = requests.post(f"{SERVER_URL}/v1/chat/completions", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    seconds = time.perf_counter() - t0
    data = r.json()
    msg = data["choices"][0]["message"]
    answer, reasoning = split_thinking(msg.get("content"), msg.get("reasoning_content"))
    usage = data.get("usage") or {}
    out_tokens = usage.get("completion_tokens") or 0
    return {
        "answer": answer,
        "reasoning": reasoning,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": out_tokens,
        "tokens_per_second": round(out_tokens / seconds, 1) if seconds > 0 else None,
        "seconds": round(seconds, 3),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }


# ---------------------------------------------------------- Jev-style way --
def jev_prompt(case_text, q):
    """Qwen3 chat template with thinking switched off, ending right where the answer token goes."""
    if q["type"] == "noul":
        body = f"Question: {q['text']}"
        rule = "Reply with exactly one word: Yes or No."
    else:
        letters = LETTERS[: len(q["options"])]
        listing = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(q["options"]))
        kind = "Levels (lowest to highest)" if q["type"] == "score" else "Options"
        body = f"Question: {q['text']}\n{kind}:\n{listing}"
        rule = f"Reply with exactly one letter: {', '.join(letters)}."
    system = "You are a decision function. Read the case and answer the question. " + rule
    user = f"Case:\n{case_text}\n\n{body}"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def first_token_candidates(meta):
    """Return [(logprob, token_text)] for the first generated position."""
    tops = meta.get("output_top_logprobs") or []
    if not tops or not tops[0]:
        return []
    out = []
    for e in tops[0]:
        if isinstance(e, (list, tuple)) and len(e) >= 3:
            lp, text = e[0], e[2]
        elif isinstance(e, dict):
            lp, text = e.get("logprob"), e.get("token") or e.get("text")
        else:
            continue
        if lp is not None and text is not None:
            out.append((float(lp), str(text)))
    return out


def logsumexp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


def score_question(q, candidates):
    """Turn top-k logprobs into a probability for every allowed answer."""
    keys = ["yes", "no"] if q["type"] == "noul" else [c.lower() for c in LETTERS[: len(q["options"])]]
    matched = {k: [] for k in keys}
    for lp, text in candidates:
        t = text.strip().lower()
        if t in matched:
            matched[t].append(lp)  # e.g. "A" and " A" both count as A
    raw = [math.exp(logsumexp(v)) if v else 0.0 for v in matched.values()]
    coverage = sum(raw)  # how much probability mass landed on valid answers
    probs = [p / coverage for p in raw] if coverage > 0 else [1 / len(raw)] * len(raw)

    best = max(range(len(probs)), key=probs.__getitem__)
    result = {
        "type": q["type"],
        "text": q["text"],
        "options": q["options"],
        "probs": [round(p, 4) for p in probs],
        "answer": q["options"][best],
        "confidence": round(probs[best], 4),
        "threshold": q["threshold"],
        "coverage": round(coverage, 4),
    }
    result["decision"] = "auto" if probs[best] >= q["threshold"] else "review"
    if q["type"] == "score":  # expected level, 0..1
        n = len(probs)
        result["expected_score"] = round(sum(i * p for i, p in enumerate(probs)) / (n - 1), 3)
    return result


def run_jev_sglang(case_text, questions):
    payload = {
        "text": [jev_prompt(case_text, q) for q in questions],  # batched = parallel
        "sampling_params": {"temperature": 0, "max_new_tokens": 1},
        "return_logprob": True,
        "top_logprobs_num": TOP_LOGPROBS,
        "return_text_in_logprobs": True,
    }
    t0 = time.perf_counter()
    r = requests.post(f"{SERVER_URL}/generate", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    seconds = time.perf_counter() - t0
    data = r.json()
    if isinstance(data, dict):
        data = [data]
    results, prompt_tokens = [], 0
    for q, item in zip(questions, data):
        meta = item.get("meta_info", {})
        prompt_tokens += meta.get("prompt_tokens") or 0
        res = score_question(q, first_token_candidates(meta))
        res["raw_token"] = item.get("text", "")
        results.append(res)
    return {"questions": results, "prompt_tokens": prompt_tokens,
            "completion_tokens": len(questions), "seconds": round(seconds, 3)}


def llamacpp_candidates(item):
    """First-token alternatives from llama-server /completion (new and old formats)."""
    steps = item.get("completion_probabilities") or []
    if not steps:
        return []
    first = steps[0]
    out = [(float(e["logprob"]), e.get("token", "")) for e in first.get("top_logprobs") or []
           if e.get("logprob") is not None]
    if not out:  # older llama.cpp: {"probs": [{"tok_str", "prob"}]}
        out = [(math.log(e["prob"]), e.get("tok_str", "")) for e in first.get("probs") or []
               if e.get("prob", 0) > 0]
    return out


def llamacpp_one(prompt):
    payload = {"prompt": prompt, "n_predict": 1, "temperature": 0,
               "n_probs": TOP_LOGPROBS, "cache_prompt": True}
    r = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def run_jev_llamacpp(case_text, questions):
    prompts = [jev_prompt(case_text, q) for q in questions]
    t0 = time.perf_counter()
    # one request per question, sent at the same time (llama-server -np slots run them in parallel)
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        data = list(pool.map(llamacpp_one, prompts))
    seconds = time.perf_counter() - t0
    results, prompt_tokens = [], 0
    for q, item in zip(questions, data):
        prompt_tokens += item.get("tokens_evaluated") or (item.get("timings") or {}).get("prompt_n") or 0
        res = score_question(q, llamacpp_candidates(item))
        res["raw_token"] = item.get("content", "")
        results.append(res)
    return {"questions": results, "prompt_tokens": prompt_tokens,
            "completion_tokens": len(questions), "seconds": round(seconds, 3)}


def run_jev(case_text, questions):
    if BACKEND == "sglang":
        return run_jev_sglang(case_text, questions)
    return run_jev_llamacpp(case_text, questions)


# ------------------------------------------------------------- mock mode --
def mock_llm(case_text, questions, settings):
    n = random.randint(900, 2200) if settings["thinking"] else random.randint(120, 260)
    seconds = n / random.uniform(45, 60)
    time.sleep(seconds)
    lines = [f"{i}. {q['options'][0]} — confidence {random.randint(70, 95)}%. "
             "(Mock answer: start the model server and run without MOCK=1 for real output.)"
             for i, q in enumerate(questions, 1)]
    return {"answer": "\n".join(lines),
            "reasoning": "Mock reasoning..." if settings["thinking"] else "",
            "prompt_tokens": 180, "completion_tokens": n,
            "tokens_per_second": round(n / seconds, 1), "seconds": round(seconds, 3),
            "finish_reason": "stop"}


def mock_jev(case_text, questions):
    seconds = random.uniform(0.08, 0.25)
    time.sleep(seconds)
    results = []
    for q in questions:
        w = [random.random() ** 3 + 1e-3 for _ in q["options"]]
        mass = random.uniform(0.85, 0.99)  # a real model leaks some mass to other tokens
        w = [x / sum(w) * mass for x in w]
        cands = [(math.log(x + 1e-9), "Yes" if q["type"] == "noul" and i == 0 else
                  "No" if q["type"] == "noul" else LETTERS[i]) for i, x in enumerate(w)]
        res = score_question(q, cands)
        res["raw_token"] = cands[max(range(len(w)), key=w.__getitem__)][1]
        results.append(res)
    return {"questions": results, "prompt_tokens": 150 * len(questions),
            "completion_tokens": len(questions), "seconds": round(seconds, 3)}


# ---------------------------------------------------------------- routes --
def read_request():
    body = request.get_json(force=True) or {}
    case_text = (body.get("case") or "").strip()
    questions = clean_questions(body.get("questions"))
    s = body.get("settings") or {}
    settings = {
        "thinking": bool(s.get("thinking", False)),
        "temperature": float(s.get("temperature", 0.6)),
        "max_tokens": int(s.get("max_tokens", 2048)),
    }
    return case_text, questions, settings


def fail(msg, code=400):
    return jsonify({"error": msg}), code


@app.get("/")
def index():
    return render_template("index.html", model=MODEL_NAME, mock=MOCK)


@app.get("/history")
def history():
    with db() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC").fetchall()
    runs = []
    for r in rows:
        d = dict(r)
        for k in ("questions", "settings", "llm_result", "jev_result"):
            d[k] = json.loads(d[k]) if d[k] else None
        runs.append(d)
    return render_template("history.html", runs=runs, model=MODEL_NAME)


@app.get("/api/health")
def health():
    if MOCK:
        return jsonify({"ok": True, "mock": True, "model": MODEL_NAME, "backend": BACKEND})
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=3)
        return jsonify({"ok": r.ok, "mock": False, "model": MODEL_NAME, "backend": BACKEND,
                        "loading": r.status_code == 503})
    except requests.RequestException as e:
        return jsonify({"ok": False, "mock": False, "error": str(e)})


@app.post("/api/llm")
def api_llm():
    case_text, questions, settings = read_request()
    if not case_text or not questions:
        return fail("Write a case and at least one complete decision question.")
    try:
        fn = mock_llm if MOCK else run_llm
        return jsonify(fn(case_text, questions, settings))
    except requests.RequestException as e:
        return fail(f"Model server request failed at {SERVER_URL}: {e}", 502)


@app.post("/api/jev")
def api_jev():
    case_text, questions, _ = read_request()
    if not case_text or not questions:
        return fail("Write a case and at least one complete decision question.")
    try:
        return jsonify(mock_jev(case_text, questions) if MOCK else run_jev(case_text, questions))
    except requests.RequestException as e:
        return fail(f"Model server request failed at {SERVER_URL}: {e}", 502)


@app.post("/api/runs")
def save_run():
    body = request.get_json(force=True) or {}
    llm, jev = body.get("llm"), body.get("jev")
    with db() as conn:
        cur = conn.execute(
            """INSERT INTO runs (created_at, case_text, questions, settings,
               llm_result, jev_result, llm_seconds, jev_seconds)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(timespec="seconds"),
                body.get("case", ""),
                json.dumps(body.get("questions", [])),
                json.dumps(body.get("settings", {})),
                json.dumps(llm) if llm else None,
                json.dumps(jev) if jev else None,
                (llm or {}).get("seconds"),
                (jev or {}).get("seconds"),
            ),
        )
    return jsonify({"id": cur.lastrowid})


@app.delete("/api/runs/<int:run_id>")
def delete_run(run_id):
    with db() as conn:
        conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    return jsonify({"deleted": run_id})


@app.get("/export.csv")
def export_csv():
    with db() as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY id").fetchall()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "created_at", "case", "question", "type", "jev_answer",
                "jev_probability", "jev_decision", "jev_seconds", "llm_seconds",
                "llm_tokens", "speedup", "llm_answer"])
    for r in rows:
        jev = json.loads(r["jev_result"]) if r["jev_result"] else {"questions": []}
        llm = json.loads(r["llm_result"]) if r["llm_result"] else {}
        speed = (round(r["llm_seconds"] / r["jev_seconds"], 1)
                 if r["llm_seconds"] and r["jev_seconds"] else "")
        for q in jev["questions"] or [{}]:
            w.writerow([r["id"], r["created_at"], r["case_text"], q.get("text", ""),
                        q.get("type", ""), q.get("answer", ""), q.get("confidence", ""),
                        q.get("decision", ""), r["jev_seconds"], r["llm_seconds"],
                        llm.get("completion_tokens", ""), speed, llm.get("answer", "")])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=llm_vs_jev_results.csv"})


@app.get("/export.json")
def export_json():
    with db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM runs ORDER BY id").fetchall()]
    for d in rows:
        for k in ("questions", "settings", "llm_result", "jev_result"):
            d[k] = json.loads(d[k]) if d[k] else None
    return Response(json.dumps(rows, indent=2, ensure_ascii=False), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=llm_vs_jev_results.json"})


init_db()

if __name__ == "__main__":
    # threaded=True so the two requests from the page really run at the same time
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5050)), debug=False, threaded=True)
