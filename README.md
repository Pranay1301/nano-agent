# nano-agent

A complete LLM agent runtime in **one Python file, standard library only**.

Tool registry, step-by-step reasoning loop, persistent memory, self-correction,
loop guard, and a full HTML trace of every decision - in ~250 lines you can
read in one sitting.

## Why

Most agent frameworks hide the loop behind thousands of lines and a pile of
dependencies. nano-agent is the opposite: the whole runtime is a single file
with zero dependencies, so you can read it, audit it, and modify it. It is a
teaching tool and a working tool.

## How it runs

```bash
python3 nano_agent.py "What is 12*(3+4)? Remember it under 'total'."
```

No pip install. Python 3.8+ is all you need. Point it at any
OpenAI-compatible chat endpoint with three environment variables:

```bash
export NANO_API_KEY=...            # your key (never stored in the repo)
export NANO_BASE_URL=https://openrouter.ai/api/v1   # or Groq, Gemini, local
export NANO_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
python3 nano_agent.py "your task"
```

Verified-zero-cost route: OpenRouter's `:free` models (checked on the public
pricing API 2026-11-08: prompt $0, completion $0). The demo run in
`live-trace.jsonl` used `nvidia/nemotron-3-ultra-550b-a55b:free` - it solved
the task in 3 steps with 0 corrections, using the calculator and its own
persistent memory. No credits were charged.

## The loop

The model must answer in one strict format:

```
THOUGHT: <one sentence>
ACTION: <tool_name> <json args>
```
or
```
THOUGHT: <one sentence>
FINAL: <answer>
```

Tool results come back as `OBSERVATION: ...`. The runtime enforces:

- **Self-correction**: a reply that does not match the format is returned to
  the model with a format hint (up to 3 corrections per run).
- **Loop guard**: the same tool called with the same args twice in a row is
  intercepted with a hint instead of burning the step budget.
- **Unknown-tool and tool-error recovery**: failures are fed back as
  observations so the model can adjust.
- **Step budget**: `max_steps` caps runaway loops; exhaustion is reported,
  not hidden.

## Built-in tools

| tool | what it does |
|---|---|
| `calc` | safe arithmetic via an AST whitelist - **no `eval`**, no names, no calls |
| `clock` | current UTC time |
| `remember` / `recall` | persistent key-value memory (`memory.json`) that survives across runs |
| `text_stats` | word/char counts |

Adding a tool is one line: `Tool("name", "doc", {"arg": "type"}, fn)`.

## Tracing

Every step (thought, action, observation, corrections, guards) lands in
`trace.jsonl`. Turn it into a readable HTML timeline:

```bash
python3 nano_agent.py --render-trace trace.jsonl > trace.html
```

`live-trace.jsonl` / `live-trace.html` in this repo are a **real run** against
the live model named above - not a fixture.

## Tests

```bash
python3 -m unittest discover tests -v
```

13 tests: format parsing, calc-injection safety (tries `__import__('os')`
and friends against the AST whitelist), multi-step tool use, self-correction,
loop guard, unknown-tool recovery, error surfacing, budget exhaustion, and
trace rendering - all on a deterministic scripted mock backend, so the suite
needs no network and no key.

## Honest limits

- Small models sometimes ignore the output format; the self-correction hint
  recovers many but not all failures (budget: 3 corrections).
- The loop is single-threaded, single-agent, text-only by design.
- The HTML trace is a static render, not a live dashboard.
- "Persistent memory" is a JSON file in the working directory - enough for
  cross-run facts, not a vector store.

## License

MIT
