"""Colab bootstrap — run once at the top of any Colab notebook.

    %pip install -q -r requirements.txt        (or let this script do it)
    import colab_setup; colab_setup.setup("nvapi-XXXX")

Everything (datasets, sandbox verifier, charts) then works in the notebook.
"""
import os, sys, pathlib

def setup(api_key: str | None = None, clone_url: str | None = None):
    root = pathlib.Path("/content/avr-agent") if pathlib.Path("/content").exists() else pathlib.Path(".")
    if not (root / "hw1").exists() and clone_url:
        os.system(f"git clone --depth 1 {clone_url} {root}")
    for p in (str(root), str(root / "hw1"), str(root / "avr_agent")):
        if p not in sys.path:
            sys.path.insert(0, p)
    if api_key:
        os.environ["NVIDIA_API_KEY"] = api_key
    assert os.environ.get("NVIDIA_API_KEY"), "pass api_key or set NVIDIA_API_KEY"
    print("AVR workspace ready at", root.resolve())
    return root
