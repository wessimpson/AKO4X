"""Static rule checks on a submission, before it spends any GPU time or official run.

Stricter than the judge's own static checks on purpose: the judge only catches some of what
the rules forbid, and final review disqualifies the rest. Errors block a candidate; warnings
go to the agent and to the pre-submit review.

The timed path is kernel_fn and every module-level function it reaches by name; prepare and
what only it reaches run once, untimed, and may sync or allocate freely.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass

ERROR, WARN = "error", "warn"


@dataclass
class Finding:
    level: str
    rule: str
    line: int
    message: str

    def __str__(self):
        return f"{self.level} [{self.rule}] line {self.line}: {self.message}"


# The judge rejects processes, network, harness frames, secrets and edited modules outright.
BANNED_IMPORTS = {
    "subprocess": "starts processes", "multiprocessing": "starts processes", "socket": "uses the network",
    "urllib": "uses the network", "http": "uses the network", "requests": "uses the network",
    "httpx": "uses the network", "aiohttp": "uses the network", "inspect": "reads harness frames",
    "signal": "touches the harness process", "reference": "imports the reference",
    "checks": "imports the judge's checks", "pickle": "can carry state between runs",
}
SUSPECT_IMPORTS = {
    "ctypes": "raw driver access; fine for CUDA launches, but reviewed", "gc": "can walk harness objects",
    "importlib": "can reload or edit modules", "threading": "work outside the timed calls",
    "concurrent": "work outside the timed calls", "time": "timing code in a submission",
    "shutil": "file system access", "tempfile": "file system access",
}
BANNED_ATTRS = {
    "_getframe": "reads harness frames", "f_back": "reads harness frames", "f_globals": "reads harness frames",
    "f_locals": "reads harness frames", "gi_frame": "reads harness frames", "environ": "reads the environment (secrets)",
    "getenv": "reads the environment (secrets)", "system": "starts processes", "popen": "starts processes",
    "fork": "starts processes", "execv": "starts processes", "spawnv": "starts processes",
    "modules": "edits loaded modules (sys.modules)",
}
# Host syncs and value-dependent shapes: break CUDA graph capture on the timed path.
SYNC_METHODS = {"item": "host sync", "cpu": "host sync", "tolist": "host sync", "numpy": "host sync",
                "synchronize": "host sync", "nonzero": "value-dependent shape (host sync)",
                "masked_select": "value-dependent shape", "unique": "value-dependent shape (host sync)",
                "unique_consecutive": "value-dependent shape", "repeat_interleave": "host sync unless output_size is given",
                "bincount": "value-dependent shape unless minlength bounds it", "argwhere": "value-dependent shape"}
TIMING_NAMES = {"Event", "elapsed_time", "perf_counter", "perf_counter_ns", "monotonic", "CUDAGraph", "graph"}

# Dtypes each problem may not compute in. NVFP4 problems hold weights as uint8 + FP8 scales.
PRECISION = {
    "muse_bf16_proj": {
        ERROR: {"float8_e4m3fn": "FP8", "float8_e5m2": "FP8", "int8": "8-bit", "float16": "FP16",
                "half": "FP16", "__half": "FP16", "__nv_fp8": "FP8", "float4": "FP4"},
        WARN: {"uint8": "byte storage: only lossless compression is allowed"},
    },
    "muse_attention_decode": {
        ERROR: {"float8_e5m2": "E5M2 (queries may go to E4M3, nothing coarser)", "int8": "8-bit",
                "float4": "FP4", "__nv_fp8_e5m2": "E5M2"},
        WARN: {"float16": "FP16 where SGLang keeps BF16", "half": "FP16", "__half": "FP16",
               "topk": "top-k attention is forbidden (every key must be attended)"},
    },
}
for _p in ("muse_moe_w4a16", "muse_decode_layer", "muse_prefill_layer"):
    PRECISION[_p] = {
        ERROR: {"float8_e5m2": "E5M2", "int8": "8-bit activations or weights", "int4": "re-quantized weights"},
        WARN: {"float8_e4m3fn": "FP8: fine for scales, the cache and (attention) queries; not for activations",
               "float16": "FP16 where SGLang keeps BF16", "half": "FP16", "__half": "FP16",
               "__nv_fp8": "FP8 in CUDA source: scales/cache/queries only"},
    }

CUDA_SOURCE_WARN = {
    "cudaDeviceSynchronize": "host sync", "cudaStreamSynchronize": "host sync",
    "cudaMemcpy(": "synchronous copy", "cudaMalloc": "allocation on the timed path", "printf": "device printf",
}


def _attr_chain(node: ast.AST) -> list[str]:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return parts[::-1]


def _hot_functions(tree: ast.Module) -> tuple[dict[str, ast.AST], set[str]]:
    """Module-level functions, and the names reachable from kernel_fn."""
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for m in cls.body:
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.setdefault(f"{cls.name}.{m.name}", m)
    hot, todo = set(), ["kernel_fn"]
    while todo:
        name = todo.pop()
        if name in hot or name not in funcs:
            continue
        hot.add(name)
        for node in ast.walk(funcs[name]):
            if isinstance(node, ast.Name) and node.id in funcs:
                todo.append(node.id)
            elif isinstance(node, ast.Name):
                todo.extend(k for k in funcs if k.startswith(node.id + "."))
    return funcs, hot


def lint(code: str, problem: str) -> list[Finding]:
    out: list[Finding] = []
    add = lambda level, rule, node, msg: out.append(Finding(level, rule, getattr(node, "lineno", 0), msg))  # noqa: E731

    first = code.splitlines()[0] if code else ""
    if first.strip() != f"# okc: problem={problem}":
        add(ERROR, "header", None, f"first line must be '# okc: problem={problem}'")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return out + [Finding(ERROR, "syntax", e.lineno or 0, str(e))]

    funcs, hot = _hot_functions(tree)
    if "kernel_fn" not in funcs:
        add(ERROR, "entry", None, "no module-level kernel_fn")
    module_names = {t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
    imported = set()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                imported.add(top)
                if top in BANNED_IMPORTS:
                    add(ERROR, "import", node, f"imports {name}: {BANNED_IMPORTS[top]}")
                elif top in SUSPECT_IMPORTS:
                    add(WARN, "import", node, f"imports {name}: {SUSPECT_IMPORTS[top]}")
        elif isinstance(node, ast.Attribute) and node.attr in BANNED_ATTRS:
            chain = _attr_chain(node)
            if node.attr != "modules" or chain[:1] == ["sys"]:
                add(ERROR, "harness", node, f"{'.'.join(chain) or node.attr}: {BANNED_ATTRS[node.attr]}")
        elif isinstance(node, ast.Name) and node.id in ("__builtins__", "__import__", "exec", "eval", "breakpoint"):
            add(ERROR, "dynamic", node, f"{node.id}: dynamic code or builtins access")
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                chain = _attr_chain(t) if isinstance(t, ast.Attribute) else []
                if len(chain) >= 2 and chain[0] in imported | {"torch", "time", "os", "sys", "triton"}:
                    add(ERROR, "monkeypatch", node, f"assigns {'.'.join(chain)}: edits a loaded module")
        elif isinstance(node, ast.Call):
            chain = _attr_chain(node.func)
            if chain[:1] == ["setattr"] and node.args and isinstance(node.args[0], ast.Name) \
                    and node.args[0].id in imported:
                add(ERROR, "monkeypatch", node, f"setattr on module {node.args[0].id}")
            if chain and chain[-1] == "open":
                mode = node.args[1].value if len(node.args) > 1 and isinstance(node.args[1], ast.Constant) else "r"
                add(WARN, "files", node, f"open(..., {mode!r}): file access in a submission")
            if chain and chain[-1] in TIMING_NAMES:
                add(WARN, "timing", node, f"{'.'.join(chain)}: timing or graph code in a submission")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for needle, why in CUDA_SOURCE_WARN.items():
                if needle in node.value:
                    add(WARN, "cuda-source", node, f"CUDA source uses {needle.rstrip('(')}: {why}")
            if re.search(r"https?://", node.value):
                add(WARN, "network", node, "URL in a string literal")

    for name in sorted(hot):
        fn = funcs[name]
        for node in ast.walk(fn):
            if isinstance(node, ast.Call):
                chain = _attr_chain(node.func)
                last = chain[-1] if chain else ""
                if last in SYNC_METHODS and not (last == "repeat_interleave" and any(
                        k.arg == "output_size" for k in node.keywords)) and not (last == "bincount" and any(
                        k.arg == "minlength" for k in node.keywords)):
                    level = ERROR if last in ("item", "cpu", "tolist", "numpy", "synchronize", "nonzero") else WARN
                    add(level, "graph", node, f"{name}: .{last}() on the timed path: {SYNC_METHODS[last]}")
                if chain[:1] == ["print"]:
                    add(WARN, "graph", node, f"{name}: print on the timed path (a tensor's value syncs)")
                if last == "data_ptr":
                    add(WARN, "cache", node, f"{name}: data_ptr() on the timed path: never key a cache on inputs")
                if chain[:2] == ["torch", "tensor"]:
                    add(WARN, "graph", node, f"{name}: torch.tensor() on the timed path copies from the host")
                if last in ("setdefault", "update", "append", "extend") and len(chain) >= 2 \
                        and chain[0] in module_names:
                    add(WARN, "cache", node, f"{name}: mutates module-level {chain[0]} on the timed path")
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                add(ERROR, "cache", node, f"{name}: global/nonlocal state written on the timed path")
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in module_names:
                        add(WARN, "cache", node, f"{name}: writes into module-level {t.value.id} on the timed path")
            elif isinstance(node, ast.If) and any(isinstance(c, ast.Call) and _attr_chain(c.func)[-1:] in (["any"], ["all"])
                                                  for c in ast.walk(node.test)):
                add(ERROR, "graph", node, f"{name}: branches on a tensor's value (.any()/.all())")
        for deco in getattr(fn, "decorator_list", []):
            chain = _attr_chain(deco.func if isinstance(deco, ast.Call) else deco)
            if chain and chain[-1] in ("lru_cache", "cache"):
                add(WARN, "cache", deco, f"{name}: memoized on the timed path: never cache outputs")

    rules = PRECISION.get(problem, {})
    for level in (ERROR, WARN):
        for token, why in rules.get(level, {}).items():
            for i, line in enumerate(code.splitlines(), 1):
                if re.search(rf"\b{re.escape(token)}\b", line) and not line.lstrip().startswith("#"):
                    out.append(Finding(level, "precision", i, f"{token}: {why}"))
                    break
    return out


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == ERROR]


def report(findings: list[Finding]) -> str:
    return "\n".join(str(f) for f in findings) or "clean"
