"""nano-agent: a complete LLM agent runtime in one Python file. Stdlib only.

Loop: the model writes THOUGHT, then either ACTION: <tool> <json-args> or
FINAL: <answer>. Tool results return as OBSERVATION. Malformed replies are
fed back as correction hints (self-correction), repeated identical actions
trigger a loop guard, and every step lands in a JSONL trace that
--render-trace turns into a visual HTML timeline.

Backends: any OpenAI-compatible chat endpoint (free tiers: Groq, OpenRouter
:free models, Gemini OpenAI-compat) via env var, or a scripted mock for tests.
"""
import ast, json, os, re, sys, time, urllib.request

# ---------------- tools ----------------
class Tool:
    def __init__(self, name, doc, args, fn):
        self.name, self.doc, self.args, self.fn = name, doc, args, fn

def _calc(expr):
    """Safe arithmetic: AST whitelist, no names, no calls."""
    def ev(node):
        if isinstance(node, ast.Expression): return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)): return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow)):
            l, r = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(r) > 1000: raise ValueError("exponent too large")
            return {ast.Add: l + r, ast.Sub: l - r, ast.Mult: l * r,
                    ast.Div: l / r, ast.Mod: l % r, ast.Pow: l ** r}[type(node.op)]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            v = ev(node.operand); return v if isinstance(node.op, ast.UAdd) else -v
        raise ValueError("only arithmetic expressions are allowed")
    return ev(ast.parse(expr, mode="eval"))

def _clock(fmt="%Y-%m-%d %H:%M:%S UTC"):
    return time.strftime(fmt, time.gmtime())

class Memory:
    """Tiny persistent key-value store the agent can use across runs."""
    def __init__(self, path="memory.json"):
        self.path = path
        self.data = json.load(open(path)) if os.path.exists(path) else {}
    def remember(self, key, value):
        self.data[str(key)] = str(value); json.dump(self.data, open(self.path, "w"))
        return f"remembered {key}"
    def recall(self, key):
        return self.data.get(str(key), "nothing stored under that key")
    def keys(self):
        return ", ".join(sorted(self.data)) or "memory empty"

def default_tools(memory=None):
    mem = memory or Memory()
    return [
        Tool("calc", "evaluate an arithmetic expression", {"expr": "string"}, lambda a: str(_calc(a["expr"]))),
        Tool("clock", "current UTC time", {"fmt": "optional strftime"}, lambda a: _clock(a.get("fmt", "%Y-%m-%d %H:%M:%S UTC"))),
        Tool("remember", "store a fact persistently", {"key": "str", "value": "str"}, lambda a: mem.remember(a["key"], a["value"])),
        Tool("recall", "read a stored fact", {"key": "str"}, lambda a: mem.recall(a["key"])),
        Tool("text_stats", "word/char count of text", {"text": "str"}, lambda a: f"{len(a['text'].split())} words, {len(a['text'])} chars"),
    ]

# ---------------- model backends ----------------
def openai_backend(base_url, model, api_key_env, timeout=90):
    def call(messages):
        key = os.environ.get(api_key_env, "")
        if not key: raise RuntimeError(f"env var {api_key_env} not set")
        body = json.dumps({"model": model, "messages": messages, "temperature": 0}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    return call

def mock_backend(script):
    """Deterministic scripted replies for tests: list of strings, cycled once."""
    it = iter(script)
    return lambda messages: next(it, "FINAL: done")

# ---------------- the runtime ----------------
PROMPT = """You are an agent that solves tasks step by step with tools.
Reply in EXACTLY this format, nothing else:
THOUGHT: <one sentence of reasoning>
ACTION: <tool_name> <json args>
or, when the task is fully solved:
THOUGHT: <one sentence>
FINAL: <final answer>

Available tools:
{tool_docs}
Rules: one action per reply; JSON args must be valid; never invent tools;
use what OBSERVATION tells you; keep THOUGHT to one sentence."""

STEP_RE = re.compile(r"THOUGHT:\s*(.*?)\s*(?:ACTION:\s*(\w+)\s*(\{.*\})|FINAL:\s*(.*))\s*$", re.S)

def parse_step(text):
    m = STEP_RE.search(text.strip())
    if not m: return None
    thought, tool, args, final = m.groups()
    if final is not None: return {"thought": thought, "final": final}
    try: return {"thought": thought, "tool": tool, "args": json.loads(args)}
    except (json.JSONDecodeError, TypeError): return None

def run_agent(task, backend, tools, max_steps=10, trace_path="trace.jsonl", memory=None):
    tool_map = {t.name: t for t in tools}
    docs = "\n".join(f"- {t.name}({json.dumps(t.args)}): {t.doc}" for t in tools)
    messages = [{"role": "system", "content": PROMPT.format(tool_docs=docs)},
                {"role": "user", "content": "Task: " + task}]
    trace = open(trace_path, "w")
    last_sig, corrections = None, 0
    for step in range(1, max_steps + 1):
        reply = backend(messages)
        parsed = parse_step(reply)
        entry = {"step": step, "raw": reply}
        if parsed is None:
            corrections += 1
            hint = ("FORMAT ERROR: reply did not match the required THOUGHT/ACTION-or-FINAL "
                    "format. Reply again in EXACTLY that format.")
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": hint}]
            entry["event"] = "format_correction"
            trace.write(json.dumps(entry) + "\n")
            if corrections >= 3: break
            continue
        entry["thought"] = parsed["thought"]
        if "final" in parsed:
            entry["event"] = "final"; trace.write(json.dumps(entry) + "\n"); trace.close()
            return {"answer": parsed["final"], "steps": step, "corrections": corrections, "trace": trace_path}
        name, args = parsed["tool"], parsed["args"]
        sig = json.dumps([name, args], sort_keys=True)
        if sig == last_sig:
            obs = "LOOP GUARD: you already called this tool with these exact args. Use the result you have or try something different."
            entry["event"] = "loop_guard"
        elif name not in tool_map:
            obs = f"TOOL ERROR: no tool named '{name}'. Available: {', '.join(tool_map)}."
            entry["event"] = "unknown_tool"
        else:
            try:
                obs = str(tool_map[name].fn(args))[:2000]
                entry["event"] = "tool_result"
            except Exception as e:
                obs = f"TOOL ERROR: {type(e).__name__}: {e}"
                entry["event"] = "tool_error"
        last_sig = sig
        entry["action"] = {"tool": name, "args": args}; entry["observation"] = obs
        trace.write(json.dumps(entry) + "\n")
        messages += [{"role": "assistant", "content": reply},
                     {"role": "user", "content": "OBSERVATION: " + obs}]
    trace.close()
    return {"answer": None, "steps": max_steps, "corrections": corrections,
            "trace": trace_path, "error": "step/correction budget exhausted"}

# ---------------- trace -> HTML ----------------
def render_trace(trace_path):
    import html as H
    rows = []
    for line in open(trace_path):
        e = json.loads(line)
        ev = e.get("event", "?")
        color = {"final": "#176b44", "tool_result": "#1a4a8a", "tool_error": "#b3261e",
                 "format_correction": "#8a6d1a", "loop_guard": "#b3261e",
                 "unknown_tool": "#b3261e"}.get(ev, "#555")
        rows.append(f"<tr><td>{e['step']}</td><td style='color:{color};font-weight:700'>{ev}</td>"
                    f"<td>{H.escape(str(e.get('thought','')))}</td>"
                    f"<td><pre>{H.escape(json.dumps(e.get('action','')))}</pre></td>"
                    f"<td><pre>{H.escape(str(e.get('observation','')))}</pre></td></tr>")
    return ("<!doctype html><meta charset=utf-8><title>nano-agent trace</title>"
            "<style>body{font-family:system-ui;max-width:1100px;margin:2rem auto;color:#14181d}"
            "table{border-collapse:collapse;width:100%}td{border:1px solid #ddd;padding:.45rem;vertical-align:top}"
            "pre{white-space:pre-wrap;font-size:.8rem;margin:0}</style>"
            "<h1>nano-agent trace</h1><table><tr><th>#</th><th>event</th><th>thought</th>"
            f"<th>action</th><th>observation</th></tr>{''.join(rows)}</table>")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--render-trace":
        print(render_trace(sys.argv[2] if len(sys.argv) > 2 else "trace.jsonl"))
    else:
        task = " ".join(sys.argv[1:]) or "What is 12 * (3 + 4)?"
        backend = openai_backend(os.environ.get("NANO_BASE_URL", "https://api.groq.com/openai/v1"),
                                 os.environ.get("NANO_MODEL", "llama-3.1-8b-instant"),
                                 os.environ.get("NANO_API_KEY_ENV", "GROQ_API_KEY"))
        print(json.dumps(run_agent(task, backend, default_tools()), indent=2))
