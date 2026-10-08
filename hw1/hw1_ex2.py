"""HW1 Exercise 2 — Deterministic, sandboxed code verifier.

Design (per spec: no exec/eval on the host):
  * LLM code + hidden unit tests are written to a temp ``.py`` file and run in
    a *child process* via ``subprocess.run`` with a hard wall-clock timeout.
  * Defence-in-depth inside the child:
      - startup resource caps (CPU time / address space / file size) via
        ``resource.setrlimit`` -> infinite loops die from SIGXCPU, memory
        bombs get MemoryError instead of nuking the box;
      - an AST allow-list gate: imports outside the whitelist, dangerous
        introspection dunders (__globals__, __subclasses__, ...) and blocked
        builtins (__import__/exit/quit) are rejected *before* any user
        statement runs;
      - cwd = temp dir, minimal environment, stdout/stderr captured.
  * Returns {"passed": bool, "error": str|None, ...} exactly as specified.

Colab note: works out of the box (Linux). For stronger isolation wrap
``_run`` in ``docker run --rm --network none --memory 256m`` — same interface.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

_ALLOWED_IMPORTS = {
    "math", "re", "json", "itertools", "functools", "collections",
    "string", "random", "statistics", "heapq", "bisect", "copy",
    "datetime", "time", "typing", "operator", "decimal", "fractions",
}

# Child-process bootstrap. The candidate source arrives over an inherited pipe
# fd (never argv), is checked by the AST gate, then exec'd in a fresh namespace.
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

_DANGEROUS_DUNDERS = {"__globals__", "__class__", "__bases__", "__subclasses__",
                      "__mro__", "__getattribute__", "__reduce__",
                      "__reduce_ex__", "__loader__", "__builtins__"}

class _Guard(_ast.NodeVisitor):
    ALLOWED = set(%(allowed)r)
    def visit_Import(self, node):
        for a in node.names:
            if a.name.split('.')[0] not in self.ALLOWED:
                raise SystemExit('SANDBOX_VIOLATION: import blocked: ' + a.name)
    def visit_ImportFrom(self, node):
        if (node.module or '').split('.')[0] not in self.ALLOWED:
            raise SystemExit('SANDBOX_VIOLATION: import blocked: ' + (node.module or '?'))
    def visit_Attribute(self, node):
        # Block introspection escapes but allow ordinary dunders (__name__,
        # __init__, __main__ ...) that normal code needs.
        if isinstance(node.attr, str) and node.attr in _DANGEROUS_DUNDERS:
            raise SystemExit('SANDBOX_VIOLATION: dunder access: ' + node.attr)
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
        full_script = f"{generated_code}\n\n{textwrap.dedent(test_cases)}\n"
        prelude = _PRELUDE % {"allowed": sorted(self.allowed_imports)}
        with tempfile.TemporaryDirectory(prefix="verifier_") as td:
            boot = os.path.join(td, "_bootstrap.py")
            with open(boot, "w") as f:
                f.write(prelude)
            return self._run(boot, td, full_script)

    def _run(self, boot: str, cwd: str, source: str) -> dict:
        env = {"PATH": "/usr/bin:/bin",
               "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        r_fd, w_fd = os.pipe()
        try:
            os.write(w_fd, source.encode("utf-8"))
        finally:
            os.close(w_fd)
        try:
            proc = subprocess.run([self.python_bin, "-B", boot, str(r_fd)],
                                  capture_output=True, text=True,
                                  timeout=self.timeout, cwd=cwd, env=env,
                                  stdin=subprocess.DEVNULL,
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
        """HumanEval convention: header + completion body + check(entry_point).

        NOTE: HumanEval's ``check`` takes the *function object*, so we call
        ``check(<entry_point>)`` (identifier), not ``check('<name>')`` (str).
        """
        script = (prompt.rstrip() + "\n" + completion + "\n\n" +
                  test + f"\ncheck({entry_point})\n")
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
            "humaneval_style": (
                "",
                "def has_close_elements(numbers, threshold):\n"
                "    for i in range(len(numbers)):\n"
                "        for j in range(i+1, len(numbers)):\n"
                "            if abs(numbers[i]-numbers[j]) < threshold:\n"
                "                return True\n"
                "    return False\n\n"
                "def check(candidate):\n"
                "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n"
                "    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.7) == False\n"
                "check(has_close_elements)"),
        }
        expect = {"pass_add": True, "fail_wrong": False, "fail_timeout": False,
                  "blocked_import": False, "blocked_escape": False,
                  "syntax_error": False, "humaneval_style": True}
        results = {}
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
