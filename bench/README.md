# bench/: measuring speed, and comparing engines

This folder holds the harness: the prompts, the client that drives a server, the format the results are written in, and the script that
turns several result files into one table. It holds no recorded data. The measurement records of the upstream project were published
separately and are not part of this port; what is here is what produces such records.

| Path | What |
|---|---|
| `bench.py` | Drives one OpenAI-compatible server, measures every run, writes one result file, prints a table |
| `compare.py` | Reads two or more result files, checks them against `schema.json`, prints one table with the fastest engine per task in bold |
| `schema.json` | The result format, as a JSON Schema document |
| `prompts/*.txt` | The 15 published prompts, byte for byte as the upstream release shipped them (Apache-2.0, same lineage as this port) |
| `results/` | Where result files go. Only `.gitkeep` is committed; the files are yours |

Python 3.10 or newer, standard library only. Nothing here imports `requests`; the client is `urllib`.

## 1. Start the server

The engine serves an OpenAI-compatible chat API. From the repository root:

```sh
PYTHONPATH=python python3 -m yasakani serve --model-dir /path/to/model --port 8000
```

`serve` takes `--host` (default `127.0.0.1`), `--port` (default 8765), `--api-key`, `--max-ctx`, `--device` and `--default-max-tokens`.
It binds the port before the model loads, so a second start fails at once instead of half-loading a second copy. `GET /health` answers
`200` with `{"status": "ok"}` once the model is ready and `503` while it is still loading; the harness does not poll it, so wait for the
line the server prints, or check `/health` yourself, before starting a measurement.

The server refuses requests whose `Host` header is not the address it is bound to, and requests that carry a foreign `Origin`, so a page
in your browser cannot reach it. For another machine use `--host 0.0.0.0 --api-key KEY` and pass the same key to `bench.py`.

Any other OpenAI-compatible server works the same way: vLLM, SGLang, llama.cpp (`llama-server`), NInfer, TensorFold. Point
`--base-url` at its `/v1` and give `--engine` a name for the table.

## 2. Run the harness

```sh
# the 15 published prompts, 2 runs each, 4096-token answers
python3 bench/bench.py --base-url http://127.0.0.1:8000/v1 --engine yasakani --out bench/results/yasakani.json

# one task, one run, a short answer: a smoke test
python3 bench/bench.py --tasks chat --runs 1 --max-tokens 256 --out /tmp/smoke.json

# greedy, thinking off, fixed seed, through this engine's extra fields
python3 bench/bench.py --temperature 0 --seed 1234 --runs 2 \
    --extra '{"chat_template_kwargs": {"enable_thinking": false}}' \
    --out bench/results/yasakani_greedy.json
```

| Option | Default | What it does |
|---|---|---|
| `--base-url` | `http://127.0.0.1:8000/v1` | Base URL of the server; `/chat/completions` is appended |
| `--engine` | `yasakani` | Name of this engine in the result file and in the comparison table |
| `--model` | `qwen3.8-27b` | Model id sent in the request body |
| `--api-key` | `$OPENAI_API_KEY` | Sent as `Authorization: Bearer` |
| `--tasks` | `all` | Comma list of task names, or `all` |
| `--runs` | `2` | Runs per task |
| `--max-tokens` | `4096` | Answer limit of every run |
| `--temperature` | `1.0` | `0` is greedy |
| `--top-p` | `0.95` | |
| `--top-k` | not sent | Sent only when given; not every server accepts it |
| `--seed` | not sent | Sent only when given; run *N* uses `seed + N` |
| `--timeout` | `1800` | Seconds per request |
| `--warmup` | `1` | Short requests before measuring, not recorded |
| `--extra` | none | JSON object merged into every request body |
| `--out` | `results/<engine>_<time>.json` | Where the result file goes |

A task is a file in `prompts/`. The 15 published ones run in the order of the upstream tables; any other `*.txt` you drop in that folder is
appended alphabetically and becomes a task with the file name as its name. The prompt is sent as a single user message; the file content is
used as it is, newline at the end included.

One request at a time, one server at a time, with nothing else using the GPU. That is what makes the numbers comparable; two clients
against one server measure the queue, not the engine.

## 3. What the numbers mean

For every run the harness records the answer tokens the **server** reports, the arrival time of the first chunk that carried text, and the
arrival time of the last one. The speed is

```
tok/s = (completion_tokens - 1) / (time from the first to the last text chunk)
```

**Reading the prompt is not part of that time.** The wait before the first chunk is reported on its own, as time to first token, and the
prompt-reading throughput as `prefill_tok_per_s = prompt_tokens / ttft`. The `- 1` is there because the first token is already counted in
the time to first token: the interval between the first and the last chunk covers the remaining `completion_tokens - 1` of them.

The token count comes from `usage.completion_tokens`, with `stream_options: {"include_usage": true}` in the request. It is never the
number of chunks. A chunk is not a token: one character can span several tokens and then arrives with the last of them, so counting chunks
undercounts the answer and reports a speed higher than the engine reached. Servers that ignore `stream_options` and send no usage block
leave the harness with nothing to count; such a run is flagged `no_usage` and left out rather than guessed at.

With thinking on, reasoning chunks count exactly like answer chunks: the first one starts the clock, the last one stops it. That matches
the upstream definition, where the reasoning is part of the output.

### Caveats

* **This is not a comparison with every setting made equal.** Weights, quantization, KV cache, context length and speculative-decoding
  settings stay each engine's own; the table says how fast each engine ran as configured, not how the engines compare at identical settings.
* **A short answer is noisy.** A run of 40 tokens is dominated by the first few chunks; the upstream tables used 4,096-token limits for this
  reason. With `--runs 1` a single hiccup is the whole number.
* **The answer limit is part of the test.** Many runs stop at the limit rather than at an end of text, so the number says how fast the
  engine wrote, not how quickly it reached an answer. `compare.py` refuses to mix files with different `--max-tokens` for this reason.
* **The client is in the loop.** A slow client, a busy machine or a network hop adds to the interval between chunks. Measure on the machine
  that runs the server, and keep other work off the GPU.
* **Warm-up matters.** The first request after a server start pays for lazy initialization. The harness sends `--warmup` short requests
  first and does not record them.
* **A looping answer is faster than a real one.** A model stuck repeating the same few tokens writes them quickly and would raise the
  average; that is why such runs are flagged and left out (below).

## 4. Runs that are left out

Two kinds of run are flagged, kept in the result file, and excluded from the summary and from the comparison table:

**A repetition loop.** The rule, applied to the text of the answer (reasoning and answer together):

1. Take the last 8,192 characters of the text.
2. For every period `p` from 1 to 256 characters, work out how many copies of `p` are needed to cover 128 characters, and never fewer than
   4 copies.
3. Test whether the last `copies * p` characters are the last `p` characters repeated. The last copy may be cut short, because a run can
   stop at its token limit in the middle of a cycle.
4. The smallest `p` that passes is the period and the run is flagged `loop`; no `p` passing means no loop.

The 128-character minimum span is what keeps ordinary text out. A line of dots, four trailing spaces or a short repeated phrase is not a
loop; 128 characters of the same few tokens, or a sentence repeated eight times, is. The period cap of 256 characters catches repeated
sentences and repeated code lines but not a long paragraph that happens to recur, which is rare enough to be left to a human reading the
answer. The detected period and the last 80 characters are stored in the run so the decision can be checked afterwards.

**An unrequested tool call.** The harness sends no `tools` and no `functions`, so any `tool_calls` delta or a `finish_reason` of
`tool_calls` is the server answering a question nobody asked. Such a run is flagged `tool_calls`. Upstream never counted these: an engine
that stops after 150 tokens with a tool call has not written an answer, and its speed is not the speed of writing one.

A run with no usage block is flagged `no_usage`, and a run that produced no text at all is flagged `empty`. Both are left out for the same
reason: there is no honest number to take from them.

`bench.py` prints the excluded runs under the table. `compare.py --no-exclude-flagged` averages everything in, which is useful for seeing
how much the exclusion moved a number; the default is to exclude, matching the upstream rule.

## 5. Compare engines

```sh
python3 bench/compare.py bench/results/*.json
python3 bench/compare.py bench/results/ --format tsv
python3 bench/compare.py bench/results/ --format json > /tmp/table.json
```

```
| Engine | chat | code | explain |
|---|---|---|---|
| engine-a | 98.0 | 98.5 | 98.5 |
| engine-b | **132.3** | **133.0** | **133.0** |
| fake-loop | — (2 excl.) | — (2 excl.) | — (2 excl.) |
```

Rows are engines, columns are tasks, cells are tok/s, and the fastest engine of each column is in bold. A cell that had runs excluded says
so: `— (2 excl.)` means every run of that task was flagged and nothing is left to average.

Before printing anything, `compare.py` checks each file against `schema.json` by hand (no `jsonschema` dependency) and then checks the set:

* **Different `max_tokens`** — refused, with the value of every file listed.
* **Different prompt sets** — refused, naming the tasks that only one file has and the tasks whose content hash differs.

Both are refusals rather than warnings because they are the two ways these tables lie: a 4,096-token run and a 512-token run are not the
same test, and two "chat" columns built from different prompts are not the same column. Exit status is `2` on any refusal, `0` on success,
so a script can check it.

## 6. The result format

`schema.json` is the format; `bench.py` writes it and `compare.py` reads it. One file is one engine, one model and one setting.

```json
{
  "schema_version": 1,
  "engine": "yasakani",
  "model": "qwen3.8-27b",
  "base_url": "http://127.0.0.1:8000/v1",
  "started_at": "2026-11-13T22:39:43+09:00",
  "max_tokens": 4096,
  "settings": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": null, "runs": 2},
  "prompts": {"chat": "4e18ff402bd91b2db1c6e0c27deaa0a06d2aea7e20d3943f0df028165910186d"},
  "tasks": ["chat"],
  "gpu": {"name": "NVIDIA GeForce RTX 5090", "driver": "616.92", "memory_total_mib": 32607, "power_limit_w": 600},
  "runs": [
    {"task": "chat", "run": 0, "prompt_tokens": 61, "completion_tokens": 4096,
     "ttft_ms": 21.0, "gen_ms": 41300.5, "tok_per_s": 99.2, "prefill_tok_per_s": 2905.4,
     "wall_ms": 41321.5, "finish_reason": "length", "flags": []}
  ],
  "summary": {"per_task": {"chat": {"mean": 99.2, "min": 99.0, "max": 99.4, "n_used": 2, "n_excluded": 0}}}
}
```

`prompts` is the fingerprint of the prompt set: task name to the SHA-256 of the file. That is what `compare.py` compares when it decides
whether two files ran the same test — not the task names, which can be reused for different content. `gpu` is whatever `nvidia-smi` said
when the run started; when it is not available the fields are empty and `note` says why, and the run goes ahead, because a missing
`nvidia-smi` is not a reason to lose a measurement.

## 7. Adding another engine

Nothing in the harness is specific to this engine. To add one:

1. Start it so that it serves an OpenAI-compatible chat API with streaming, and note the base URL. vLLM and SGLang: their own `serve`
   command, then `http://127.0.0.1:PORT/v1`. llama.cpp: `llama-server -m model.gguf --port PORT`, then `http://127.0.0.1:PORT/v1`.
2. Run the harness with the same settings you use for every other engine — the same `--tasks`, the same `--runs`, the same `--max-tokens`,
   the same sampling. Only `--engine`, `--model` and `--base-url` change:

   ```sh
   python3 bench/bench.py --base-url http://127.0.0.1:8001/v1 --engine vllm --model qwen3.8-27b \
       --max-tokens 4096 --runs 2 --out bench/results/vllm.json
   ```
3. If the server wants its own fields, pass them with `--extra` instead of editing the harness: `--extra '{"ignore_eos": true}'`, or
   `--extra '{"chat_template_kwargs": {"enable_thinking": false}}'`.
4. Compare: `python3 bench/compare.py bench/results/*.json`.

If a server rejects a field the harness sends, drop it rather than editing `bench.py`: `--top-k` is only sent when you pass it, and
`--temperature`/`--top-p` can be set to values the server accepts. A server that refuses `stream_options` cannot be measured honestly —
without `usage.completion_tokens` there is no token count — so such a run is flagged `no_usage` instead of being counted from chunks.

Record what you changed: the engine version, the model file, and any flag that affects speed (context length, KV cache type,
speculative decoding). Two rows of a table mean nothing without them, which is why `gpu`, `model` and `base_url` are in every result file.
