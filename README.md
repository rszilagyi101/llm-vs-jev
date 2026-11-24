# Local LLM vs Jev-style decisions (classroom demo)

Same local model (Qwen3-8B), two ways to use it, run side by side:

- Normal LLM way: chat completion, the model writes its answer token by token
  (optionally with Qwen3 thinking). Confidence is only text.
- Jev-style way: each decision question (Choice / Score / Yes-No) is a prompt whose next
  token is the answer. All questions run in parallel, each generates exactly 1 token,
  and the app reads the probabilities of the allowed answers.

This imitates the interface of TypeSafe's Jev; it is not their model, and Qwen3's
probabilities are not calibrated the way Jev claims to be.

## Mac (Apple Silicon, e.g. MacBook Air M4 16 GB) — llama.cpp

    # 1. model server (first start downloads ~5 GB)
    brew install llama.cpp
    llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M --jinja --reasoning-format deepseek \
        -c 16384 -np 4 --port 8080

        joo

    llama-server -hf Qwen/Qwen3-8B-GGUF:Q4_K_M --jinja --reasoning-format deepseek -c 16384 -np 4 --port 8080

    # 2. web app, in a second terminal
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python app.py                 # open http://127.0.0.1:5050

-np 4 = 4 parallel slots (1 for the LLM + 3 decision questions), each with 16384/4 = 4096
tokens of context. If thinking answers get cut off, lower "Max tokens" in the page or use
-c 24576 (uses more memory).

Short on memory? Use Qwen/Qwen3-4B-GGUF:Q4_K_M instead (and MODEL_NAME=Qwen3-4B for the label).

## Linux + NVIDIA GPU — SGLang

    pip install sglang
    python -m sglang.launch_server --model-path Qwen/Qwen3-8B --port 30000 --reasoning-parser qwen3
    BACKEND=sglang python app.py

## Options

Environment variables: BACKEND (llamacpp | sglang), SERVER_URL, MODEL_NAME, PORT (default
5050, because macOS uses 5000 for AirPlay), DB_PATH, MOCK=1 (fake answers, no model needed).

Results are saved in results.db (SQLite). The Saved results page shows every run,
the average speedup, and CSV / JSON downloads.

## Things to show students

1. Run with thinking off, then on: the LLM lane slows down a lot, Jev-style doesn't change.
2. "valid-answer mass": how much of the model's probability landed on an allowed answer.
   Low values mean the question or options are unclear to the model.
3. Compare the LLM's written confidence ("90%") with the Jev-style measured probability.
4. Move the auto-decide threshold: the same probability becomes "auto" or "review".
