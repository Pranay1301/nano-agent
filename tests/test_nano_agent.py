import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nano_agent as na

def tmpmem():
    d = tempfile.mkdtemp()
    return na.Memory(os.path.join(d, "memory.json"))

def tracefile():
    return os.path.join(tempfile.mkdtemp(), "trace.jsonl")

class TestParse(unittest.TestCase):
    def test_action(self):
        p = na.parse_step('THOUGHT: need math\nACTION: calc {"expr": "2+2"}')
        self.assertEqual(p["tool"], "calc"); self.assertEqual(p["args"], {"expr": "2+2"})
    def test_final(self):
        p = na.parse_step("THOUGHT: done\nFINAL: the answer is 4")
        self.assertEqual(p["final"], "the answer is 4")
    def test_garbage(self):
        self.assertIsNone(na.parse_step("I think the answer might be 4, let me explain at length"))
    def test_bad_json(self):
        self.assertIsNone(na.parse_step('THOUGHT: x\nACTION: calc {expr: 2+2}'))

class TestCalcSafety(unittest.TestCase):
    def test_math(self): self.assertEqual(na._calc("12*(3+4)"), 84)
    def test_injection(self):
        for bad in ["__import__('os').system('id')", "open('x')", "a+1", "[1][0]"]:
            with self.assertRaises(ValueError): na._calc(bad)

class TestAgentLoop(unittest.TestCase):
    def test_multi_step_mock(self):
        mem = tmpmem()
        script = [
            'THOUGHT: compute first\nACTION: calc {"expr": "12*(3+4)"}',
            'THOUGHT: store result\nACTION: remember {"key": "answer", "value": "84"}',
            'THOUGHT: done\nFINAL: 84',
        ]
        r = na.run_agent("compute and store", na.mock_backend(script), na.default_tools(mem), trace_path=tracefile())
        self.assertEqual(r["answer"], "84"); self.assertEqual(r["steps"], 3)
        self.assertEqual(mem.recall("answer"), "84")
    def test_self_correction(self):
        script = ["let me think out loud about this", 'THOUGHT: ok\nFINAL: fixed']
        r = na.run_agent("t", na.mock_backend(script), na.default_tools(tmpmem()), trace_path=tracefile())
        self.assertEqual(r["answer"], "fixed"); self.assertEqual(r["corrections"], 1)
    def test_loop_guard(self):
        script = ['THOUGHT: try\nACTION: calc {"expr": "1+1"}',
                  'THOUGHT: try again same\nACTION: calc {"expr": "1+1"}',
                  'THOUGHT: use result\nFINAL: 2']
        r = na.run_agent("t", na.mock_backend(script), na.default_tools(tmpmem()), trace_path=tracefile())
        self.assertEqual(r["answer"], "2")
        events = [json.loads(l)["event"] for l in open(r["trace"])]
        self.assertIn("loop_guard", events)
    def test_unknown_tool(self):
        script = ['THOUGHT: fly\nACTION: fly {"x": 1}', 'THOUGHT: nope\nFINAL: no such tool']
        r = na.run_agent("t", na.mock_backend(script), na.default_tools(tmpmem()), trace_path=tracefile())
        self.assertEqual(r["answer"], "no such tool")
    def test_budget_exhaustion(self):
        script = ['THOUGHT: loop forever\nACTION: clock {}'] * 12
        r = na.run_agent("t", na.mock_backend(script), na.default_tools(tmpmem()), max_steps=5, trace_path=tracefile())
        self.assertIsNone(r["answer"]); self.assertIn("error", r)
    def test_tool_error_surfaces(self):
        script = ['THOUGHT: divide by zero\nACTION: calc {"expr": "1/0"}', 'THOUGHT: adjust\nFINAL: cannot divide by zero']
        r = na.run_agent("t", na.mock_backend(script), na.default_tools(tmpmem()), trace_path=tracefile())
        self.assertEqual(r["answer"], "cannot divide by zero")
    def test_trace_renders_html(self):
        script = ['THOUGHT: go\nFINAL: hi']
        r = na.run_agent("t", na.mock_backend(script), na.default_tools(tmpmem()), trace_path=tracefile())
        html = na.render_trace(r["trace"])
        self.assertIn("<table>", html); self.assertIn("final", html)

if __name__ == "__main__":
    unittest.main(verbosity=2)
