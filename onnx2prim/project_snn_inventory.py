"""Read-only source inventory; never execute project code or load checkpoints.

Model IDs are source families, not generated fixtures or checkpoint names.
Expressions that cannot be evaluated by the small literal interpreter remain
source expressions. ``semantics`` retains executable statements and line numbers
so library-dependent behavior is distinguishable from project-defined equations.
"""
from __future__ import annotations

import ast
import json
import operator
from pathlib import Path


MODEL_SOURCES = {
    "shd": ("generator/shd/shd_top.py", "generator/shd/shd_input.py",
            "generator/shd/shd_hidden.py", "generator/shd/shd_output.py"),
    "SoundSNN": ("generator/SoundSNN/sound_snn.py",
                 "generator/SoundSNN/sound_snn_training.py",
                 "generator/SoundSNN/snn_config.py", "generator/SoundSNN/data_handler.py",
                 "generator/SoundSNN/g1_1core.py", "generator/SoundSNN/g1_data.py",
                 "generator/SoundSNN/sound_snn_top.py"),
    "snnlenet": ("generator/snnlenet/snnlenet.py",),
    "snnlenet/snnpredict": ("generator/snnlenet/snnpredict.py",),
    "snnlenet/train": ("generator/snnlenet/train.py",),
    "snnlenet/train_lif_float": ("generator/snnlenet/train_lif_float.py",),
    "full": ("generator/full.py",),
    "snn": ("snn.py",),
    "snn_mnist_s": ("snn_mnist_s.py",),
    "HNN": ("generator/test/beijingshi/HNN.py",),
    "SNN_forward": ("generator/test/beijingshi/SNN_forward.py",),
}

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.LShift: operator.lshift, ast.RShift: operator.rshift}


def _literal(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in env:
        return env[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [_literal(n, env) for n in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand, env)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _literal(node.left, env), _literal(node.right, env)
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise ValueError("non-numeric expression")
        return _OPS[type(node.op)](left, right)
    raise ValueError("symbolic expression")


def _value(node, env):
    try:
        return _literal(node, env)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return {"expression": ast.unparse(node)}


def _env(tree):
    env = {}
    # Top-level constants only: do not confuse same-named function locals.
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        env[target.id] = _literal(node.value, env)
                    except (ValueError, TypeError, ZeroDivisionError):
                        pass
    return env


def _fact(node, path, **extra):
    return dict(source_path=path, line=node.lineno, source=ast.unparse(node), **extra)


def _calls(tree, path, names):
    env = _env(tree)
    return [_fact(n, path, operation=ast.unparse(n.func),
                  args=[_value(a, env) for a in n.args],
                  kwargs={k.arg or "**": _value(k.value, env) for k in n.keywords})
            for n in sorted(ast.walk(tree), key=lambda n: getattr(n, "lineno", 0))
            if isinstance(n, ast.Call) and ast.unparse(n.func).split(".")[-1] in names]


def _layers(tree, path, class_name):
    env = _env(tree)
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == class_name), None)
    if cls is None:
        return []
    init = next((n for n in cls.body if isinstance(n, ast.FunctionDef)
                 and n.name == "__init__"), None)
    if init is None:
        return []
    layers = []
    for node in sorted(ast.walk(init), key=lambda n: getattr(n, "lineno", 0)):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        for target in node.targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                layers.append(_fact(node, path, name=target.attr,
                                    operation=ast.unparse(call.func),
                                    args=[_value(a, env) for a in call.args],
                                    kwargs={k.arg or "**": _value(k.value, env) for k in call.keywords}))
    return layers


def _shapes(layers):
    """Declared spatial chain for MNIST sources; never repair a wrong FC."""
    shape = [1, 28, 28]
    facts = []
    for layer in layers:
        op, args, kw = layer["operation"].split(".")[-1], layer["args"], layer["kwargs"]
        if op not in {"Conv2d", "MaxPool2d", "Linear"}:
            continue
        def param(name, index, default):
            return kw.get(name, args[index] if len(args) > index else default)
        if op == "Linear":
            observed = shape[0] * shape[1] * shape[2] if len(shape) == 3 else shape[0]
            declared = param("in_features", 0, None)
            facts.append(dict(layer=layer["name"], observed_input=observed,
                              intended_input=declared, matches=observed == declared,
                              output=param("out_features", 1, None)))
            shape = [param("out_features", 1, None)]
            continue
        kernel = param("kernel_size", 2 if op == "Conv2d" else 0, 1)
        stride = param("stride", 3 if op == "Conv2d" else 1, 1 if op == "Conv2d" else kernel)
        padding = param("padding", 4 if op == "Conv2d" else 2, 0)
        dilation = param("dilation", 5 if op == "Conv2d" else 3, 1)
        pair = lambda x: x if isinstance(x, list) else [x, x]
        values = [pair(x) for x in (kernel, stride, padding, dilation)]
        if not all(isinstance(v, int) for p in values for v in p):
            return facts
        k, s, p, d = values
        shape = [param("out_channels", 1, None) if op == "Conv2d" else shape[0]] + [
            (shape[i + 1] + 2 * p[i] - d[i] * (k[i] - 1) - 1) // s[i] + 1 for i in range(2)]
        facts.append(dict(layer=layer["name"], observed_output=shape[:]))
    return facts


def inspect_project(root) -> list[dict]:
    """Inspect known project source families using only read_text and AST.

    Absent families are omitted; missing members and syntax errors in present
    families are reported. Results are deterministic and JSON serializable.
    Requirements describe extraction/compilation needs, not installed packages.
    """
    root = Path(root)
    result = []
    for model_id, paths in MODEL_SOURCES.items():
        if not any((root / p).is_file() for p in paths):
            continue
        record = dict(model_id=model_id, source_paths=list(paths), topology=[],
                      dimensions={}, semantics={}, source_errors=[], requirements=[])
        trees = {}
        for path in paths:
            try:
                trees[path] = ast.parse((root / path).read_text(encoding="utf-8-sig"), filename=path)
            except (OSError, UnicodeError, SyntaxError) as exc:
                record["source_errors"].append(dict(code="source_unreadable", source_path=path,
                                                     message=str(exc)))
        if not trees:
            result.append(record)
            continue
        primary = paths[0]
        tree = trees.get(primary)
        semantics = record["semantics"]
        semantics["methods"] = []
        semantics["constants"] = {p: _env(t) for p, t in trees.items()}
        for path, source_tree in trees.items():
            for cls in source_tree.body:
                if isinstance(cls, ast.ClassDef):
                    for method in cls.body:
                        if isinstance(method, ast.FunctionDef) and (method.name in {"forward", "backward", "__init__"} or "mem_update" in method.name):
                            semantics["methods"].append(_fact(method, path, owner=cls.name, method=method.name))
                elif isinstance(cls, ast.FunctionDef) and cls.name == "neuron_block":
                    semantics["methods"].append(_fact(cls, path, owner=None, method=cls.name))
        semantics["neuron_calls"] = [f for p, t in trees.items() for f in _calls(
            t, p, {"Leaky", "LIFFCNeuron", "LIFSpike", "LIFNeuron", "p08"})]
        record["topology"] = [f for p, t in trees.items() for f in _calls(
            t, p, {"Conv2d", "Linear", "MaxPool2d", "LIFFCNeuron", "snn_n_model", "p41", "p02"})]
        record["requirements"] = ["Preserve source call order, state lifetimes and output reduction.",
                                   "Resolve symbolic expressions explicitly; do not import model scripts."]
        if tree is not None:
            class_name = ("SNNLeNet" if model_id == "full" else "SNN" if model_id in {"snn", "snn_mnist_s", "SoundSNN"}
                          else "LeNetLIF" if model_id.endswith("/train") else "LeNetLIF_Float" if model_id.endswith("/train_lif_float") else "LeNet")
            layers = _layers(tree, primary, class_name)
            semantics["declared_layers"] = layers
            if model_id not in {"SoundSNN", "shd", "HNN", "SNN_forward"}:
                record["dimensions"] = dict(input=["B", 1, 28, 28], spatial_chain=_shapes(layers))
                for shape in record["dimensions"]["spatial_chain"]:
                    if shape.get("matches") is False:
                        record["source_errors"].append(dict(code="fc_shape_mismatch", **shape))
        if model_id in {"snn", "snn_mnist_s", "full"} and tree is not None:
            forward = next((f for f in semantics["methods"] if f["method"] == "forward" and f["owner"] in {"SNN", "SNNLeNet"}), None)
            if forward:
                ft = ast.parse(forward["source"])
                calls = _calls(ft, primary, {"lif1", "lif2", "lif3"})
                semantics["lif_calls_per_step"] = {name: sum(c["operation"] == "self." + name for c in calls)
                                                    for name in ("lif1", "lif2", "lif3")}
                semantics["state_lifetime"] = "init_leaky at forward entry; explicit mem carried between timesteps"
                semantics["library_contract"] = (
                    "snntorch.Leaky is stateful: each call replaces self.mem with explicit mem, "
                    "computes reset from it, then assigns a new self.mem. With subtract/delayed-reset "
                    "defaults: mem_next=clamp(beta,0,1)*mem+current-reset(mem)*threshold. "
                    "Exact defaults/comparator require the target snntorch version.")
                if any(n > 1 for n in semantics["lif_calls_per_step"].values()):
                    record["source_errors"].append(dict(code="double_lif_call", message=
                        "lif1/lif2 each called twice with the same explicit old mem: first spike feeds pool; "
                        "second returned mem is carried. Module state/reset mutate twice, but explicit old "
                        "mem overrides first internal result; these are not two chained integration steps."))
            record["requirements"].append("Pin snntorch Leaky defaults, delayed reset and internal state mutation.")
        if model_id == "SoundSNN":
            layers = semantics.get("declared_layers", [])
            widths = [l["args"][:2] for l in layers if l["operation"] == "LIFFCNeuron"]
            mapping = [f for f in record["topology"] if f["source_path"].endswith("g1_1core.py") and f["operation"] == "p41"]
            record["dimensions"] = dict(input_layout="B,T,N", fc_dimensions=widths,
                                         mapping_input=mapping[0]["kwargs"].get("cin") if mapping else None)
            semantics["state_update"] = "mem += current; spike = (mem > thresh); output = spike; mem *= (1-spike.detach()); mem = mem*(decay/255)+beta"
            semantics["state_lifetime"] = "zero membrane at each neuron forward; time axis 1; mean over time at model output"
            if widths and mapping and widths[0][0] != mapping[0]["kwargs"].get("cin"):
                record["source_errors"].append(dict(code="sound_input_dimension_mismatch", observed_input=mapping[0]["kwargs"].get("cin"), intended_input=widths[0][0]))
            record["source_errors"].append(dict(code="incomplete_mapping", message="g1_1core maps only fc1 with pX5 max; no mapped LIF state update or fc2/fc3/fc4. Summary tensor writes are commented out."))
            record["requirements"].append("Implement all four FC/LIF layers and restore intermediate extraction; reconcile handler/mapping input with source input.")
        if model_id == "shd":
            hidden = [f for f in semantics["neuron_calls"] if f["source_path"].endswith("shd_hidden.py")]
            semantics["internal_states"] = [dict(source_path=f["source_path"], line=f["line"], **f["kwargs"]) for f in hidden]
            record["dimensions"] = dict(input_neurons=semantics["constants"].get("generator/shd/shd_top.py", {}).get("input_neuron_number"),
                                         hidden_neurons=semantics["constants"].get("generator/shd/shd_hidden.py", {}).get("hidden_neuron_number"),
                                         branch_fc=[f["kwargs"] for f in record["topology"] if f["source_path"].endswith("shd_hidden.py") and f["operation"] == "p41"])
            semantics["branch_structure"] = "Four 192->32 dendrite p41/p08 branches per hidden core, distinct addr_v; p02 ky=4 reduction then separate soma p08. Output p41 64->32 followed by p08. Primitive parameters are the contract, not an inferred standard LIF."
            for p, t in trees.items():
                for n in ast.walk(t):
                    if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("generator.zhl_nmi22") and not (root / (n.module.replace(".", "/") + ".py")).is_file():
                        record["source_errors"].append(_fact(n, p, code="missing_local_import", module=n.module))
            record["requirements"].append("Preserve four dendrite membranes plus soma, fire_type=2/5, adaptive leak and Tw_len=1000; resolve stale SHD imports.")
        if model_id in {"HNN", "SNN_forward"}:
            configs = _calls(tree, primary, {"snn_n_model"}) if tree else []
            record["dimensions"] = dict(input_layout="B,N,T", layer_sizes=[f["kwargs"].get("layer_size") for f in configs])
            semantics["state_update"] = "mem=last_mem*decay*(1-last_spike)+input.mm(w); spike=(mem>thresh); b argument unused"
            semantics["state_lifetime"] = "zero h_mem/h_spike at forward entry; previous spike gates next step decay"
            semantics["output"] = "sum of last-layer spikes over time" if model_id == "HNN" else "last timestep last-layer spikes"
            record["source_errors"].append(dict(code="unused_bias", message="neuron_block accepts b but never adds it"))
            record["requirements"].append("Preserve strict threshold 0.3, decay 0.5, ignored bias and source-specific reduction.")
        if model_id == "snnlenet":
            record["source_errors"].append(dict(code="hybrid_reference_mismatch", message="Python LeNet uses fc_relu2; mapping replaces fc2 activation with p08 Vth0=10, fire_type=4, Tw disabled. Python reference is not a temporal LIF model."))
            record["requirements"].append("Reconcile fc2 Python ReLU with mapped p08; preserve Cut floor/clamp and NHWC flatten order.")
        if model_id in {"snnlenet/train", "snnlenet/train_lif_float"}:
            semantics["state_update"] = "Stateless SurrogateSpike: (x >= threshold).float(); optional use_lif branch, no membrane recurrence."
        if model_id == "snnlenet/snnpredict":
            semantics["state_update"] = "V=clamp(x,min=v_lower); spk,mem=self._lif(V); return spk*mem; init_hidden=True persists module membrane between forwards."
            record["requirements"].append("Do not equate spk*mem with binary p08 output; pin snntorch reset_delay/zero-reset behavior.")
        result.append(record)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=str(Path(__file__).resolve().parents[1]))
    print(json.dumps(inspect_project(parser.parse_args().root), ensure_ascii=False, indent=2))
