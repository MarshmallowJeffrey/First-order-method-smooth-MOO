"""Identity of a stored run: what it was computed from.

A run file records the settings it was started with (`run_spec`, written by the run scripts), the SHA-256 of the
package source (its code without comments and docstrings), the package versions, the SHA-256 of the MDP model and the SHA-256 of its .npz arrays.  The run
scripts reuse a stored run only if all of these match the current ones and the arrays are intact; otherwise they
stop with an error instead of silently skipping or overwriting.
"""
import ast
import hashlib
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

PACKAGE = Path(__file__).resolve().parent
MATCHED = ("run_spec_sha256", "source_sha256", "versions", "model_sha256")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def file_sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def code_sha256(path):
    """SHA-256 of the code of a Python file: its syntax tree without docstrings (comments are not part of it), so
    that editing comments or docstrings leaves it unchanged."""
    tree = ast.parse(Path(path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body \
                and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
    return sha256_bytes(ast.dump(tree, include_attributes=False).encode())


def source_sha256():
    h = hashlib.sha256()
    for f in sorted(PACKAGE.glob("*.py")):
        h.update(f.name.encode()); h.update(code_sha256(f).encode())
    return h.hexdigest()


def versions():
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "gymnasium", "mo-gymnasium"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def git_state():
    """Informational only (the source hash is what is matched)."""
    try:
        commit = subprocess.run(["git", "-C", str(PACKAGE), "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(PACKAGE), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return dict(commit=commit, uncommitted_changes=bool(dirty))


def model_sha256(model):
    h = hashlib.sha256()
    for key in ("P", "R", "rho0", "pi_ref"):
        h.update(np.ascontiguousarray(model[key], dtype=float).tobytes())
    h.update(json.dumps(dict(name=model["name"], gamma=model["gamma"], tau=model["tau"]), sort_keys=True).encode())
    return h.hexdigest()


def run_identity(run_spec, model):
    return dict(run_spec=run_spec, run_spec_sha256=sha256_bytes(json.dumps(run_spec, sort_keys=True).encode()),
                source_sha256=source_sha256(), versions=versions(), model_sha256=model_sha256(model), git=git_state())


def fingerprint(meta):
    """What a stored run was computed from and what it produced: its identity and the SHA-256 of its arrays."""
    stored = meta.get("identity", {})
    return dict({k: stored.get(k) for k in MATCHED}, npz_sha256=meta.get("npz_sha256"))


def reusable(json_path, run_spec, model):
    """True if the stored run has the current identity and intact arrays; False if there is no stored run;
    RuntimeError if a stored run differs."""
    json_path = Path(json_path)
    if not json_path.exists():
        return False
    meta = json.loads(json_path.read_text())
    stored, current = meta.get("identity", {}), run_identity(run_spec, model)
    differ = [k for k in MATCHED if stored.get(k) != current[k]]
    npz = json_path.with_suffix(".npz")
    if meta.get("npz_sha256") is not None and (not npz.exists() or file_sha256(npz) != meta["npz_sha256"]):
        differ.append("npz")
    if differ:
        raise RuntimeError(f"{json_path} was computed with a different {', '.join(differ)}; move it away or write "
                           f"to another --results directory")
    return True
