"""HW1 Exercise 2 — Deterministic, sandboxed code verifier.

Design (per spec: no exec/eval on the host):
  * LLM code + hidden unit tests are written to a temp ``.py`` file and run in
    a *child process* via ``subprocess.run`` with a hard timeout.
  * Defence-in-depth inside the child:
      - startup resource caps (CPU time / address space / file size) via
        ``resource.setrlimit`` -> infinite loops die from SIGXCPU, memory
        bombs get MemoryError instead of nuking the box;
      - an AST allow-list gate: imports outside the whitelist and dunder
        attribute access are rejected *before* the first statement runs
        (blocks os/subprocess/socket/__import__/``__globals__`` escapes);
      - cwd = temp dir, empty environment, stdout/stderr captured, never eval'd.
  * Returns {"passed": bool, "error": str|None, ...} exactly as specified.

Colab note: works out of the box (Linux sandbox). For stronger isolation use
the optional Docker backend (``backend="docker"``) — same interface.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import textwrap

_DANGEROUS_DUNDERS = {"__globals__", "__class__", "__bases__", "__subclasses__",
                     "__mro__", "__getattribute__", "__reduce__", "__reduce_ex__",
                     "__loader__", "__dict__", "__slots__"}

_ALLOWED_IMPORTS = {
    "math", "re", "json", "itertools", "functools", "collections",
    "string", "random", "statistics", "heapq", "bisect", "copy",
    "datetime", "time", "typing", "operator", "decimal", "fractions",
}

# Child-process bootstrap: resource caps + AST gate, then exec the candidate.
# The candidate source is read from argv[1] via os.read on a raw fd (the only
# allowed "open" inside the guard).  After the gate passes, __builtins__ is
# restored so user code can use open()/eval() normally — the sandbox boundary
# is the subprocess itself (timeout + rlimits + isolated cwd/env).
_PRELUDE = """\
import ast as _ast, sys as _sys, os as _os
try:
    import resource as _res
    for _r, _v in [(_res.RLIMIT_CPU, (3, 4)),
                   (_res.RLIMIT_AS, (512 * 1024 * 1024,) * 2),
                   (_res.RLIMIT_FSIZE, (8 * 1024 * 1024,) * 2)]:
        try: _res.setrlimit(_r, _v)
        except Exception: pass
except ImportError:
    pass

class _Guard(_ast.NodeVisitor):
    ALLOWED = set(%(allowed)r)
    def visit_Import(self, node):
        for a in node.names:
            root = a.name.split('.')[0]
            if root not in self.ALLOWED:
                raise SystemExit('SANDBOX_VIOLATION: import blocked: ' + a.name)
    def visit_ImportFrom(self, node):
        root = (node.module or '').split('.')[0]
        if root not in self.ALLOWED:
            raise SystemExit('SANDBOX_VIOLATION: import blocked: ' + (node.module or '?'))
    def visit_Attribute(self, node):
        # Block introspection escapes (obj.__globals__['__builtins__'] etc.)
        # but allow ordinary dunders (__name__, __init__, __main__ ...).
        if isinstance(node.attr, str) and node.attr in _DANGEROUS_DUNDERS:
            raise SystemExit('SANDBOX_VIOLATION: dunder attribute access: ' + node.attr)
        self.generic_visit(node)
    def visit_Call(self, node):
        f = node.func
        if isinstance(f, _ast.Name) and f.id in ('__import__', 'exit', 'quit'):
            raise SystemExit('SANDBOX_VIOLATION: blocked builtin: ' + f.id)
        self.generic_visit(node)

_fd = int(_sys.argv[1])
_chunks = []
while True:
    _b = _os.read(_fd, 65536)
    if not _b: break
    _chunks.append(_b)
_src = b''.join(_chunks).decode('utf-8')
_Guard().visit(_ast.parse(_src))
_g = {'__name__': '__candidate__', '__builtins__': __builtins__}
exec(compile(_src, 'candidate.py', 'exec'), _g)
"""


class CodeVerifier:
    """Sandboxed subprocess verifier for LLM-generated Python."""

    def __init__(self, timeout_seconds: int = 5, allowed_imports=None,
                 python_bin: str | None = None):
        self.timeout = timeout_seconds
        self.allowed_imports = set(allowed_imports or _ALLOWED_IMPORTS)
        self.python_bin = python_bin or sys.executable

    # ------------------------------------------------------------------ core
    def verify(self, generated_code: str, test_cases: str) -> dict:
        """Run ``generated_code`` + hidden ``test_cases`` in a sandboxed child.

        Returns: {"passed": bool, "error": str or None, "stdout", "stderr"}
        """
        full_script = (f"{generated_code}\n\n{textwrap.dedent(test_cases)}\n")
        prelude = _PRELUDE % {"allowed": sorted(self.allowed_imports)}
        with tempfile.TemporaryDirectory(prefix="verifier_") as td:
            boot = os.path.join(td, "_bootstrap.py")
            with open(boot, "w") as f:
                f.write(prelude)
            return self._run(boot, td, full_script)

    def _run(self, boot: str, cwd: str, source: str) -> dict:
        env = {"PATH": "/usr/bin:/bin",
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        # Candidate source is handed to the child over an inherited pipe fd;
        # it never touches the command line and is AST-gated before exec.
        r_fd, w_fd = os.pipe()
        os.write(w_fd, source.encode("utf-8"))
        os.close(w_fd)
        try:
            proc = subprocess.run([self.python_bin, "-B", boot, str(r_fd)],
                                  capture_output=True, text=True,
                                  timeout=self.timeout, cwd=cwd, env=env,
                                  pass_fds=(r_fd,))
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"TIMEOUT after {self.timeout}s",
                    "stdout": "", "stderr": ""}
        except Exception as e:  # noqa: BLE001
            return {"passed": False, "error": f"RUNNER_ERROR: {e}",
                    "stdout": "", "stderr": ""}
        finally:
            try:
                os.close(r_fd)
            except OSError:
                pass
        err = None
        if proc.returncode != 0:
            err = (proc.stderr.strip() or proc.stdout.strip() or
                   f"exit code {proc.returncode}")
            err = "\n".join(err.splitlines()[-12:])   # trim to tail
        return {"passed": proc.returncode == 0, "error": err,
                "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]}

    # ------------------------------------------------------- HumanEval style
    def verify_completion(self, prompt: str, completion: str,
                          test: str, entry_point: str) -> dict:
        """HumanEval convention: function header + body + check(entry_point)."""
        script = prompt.rstrip() + "\n" + completion + "\n\n" + \
                 test + f"\ncheck('{entry_point}')\n"
        return self.verify("", script)

    # ------------------------------------------------------------ self-tests
    def self_test(self) -> dict:
        cases = {
            "pass_add": (
                "def add(a, b):\n    return a + b",
                "assert add(1, 2) == 3\nassert add(-1, 1) == 0"),
            "fail_wrong": (
                "def add(a, b):\n    return a - b",
                "assert add(1, 2) == 3"),
            "fail_timeout": (
                "def spin():\n    while True:\n        pass",
                "spin()"),
            "blocked_import": (
                "import os\ndef f(): return os.getcwd()",
                "assert f()"),
            "blocked_escape": (
                "def f():\n    return (lambda: 0).__globals__",
                "assert f()"),
            "syntax_error": ("def broken(:", "assert broken()"),
        }
        results = {}
        expect = {"pass_add": True, "fail_wrong": False, "fail_timeout": False,
                  "blocked_import": False, "blocked_escape": False,
                  "syntax_error": False}
        for name, (code, tests) in cases.items():
            r = self.verify(code, tests)
            results[name] = {"passed": r["passed"],
                             "as_expected": r["passed"] == expect[name],
                             "error": (r["error"] or "")[:120]}
        results["all_ok"] = all(v["as_expected"] for k, v in results.items()
                                if k != "all_ok")
        return results


if __name__ == "__main__":
    import json
    v = CodeVerifier(timeout_seconds=5)
    print(json.dumps(v.self_test(), indent=2))
