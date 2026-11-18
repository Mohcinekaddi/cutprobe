import copy
import json
import os

import onnx
from onnx import TensorProto, helper

from .onnx_graph import (crossing_tensors, data_inputs, extract_submodel, initializer_names,
                         make_batch_dynamic, reachable_nodes, resolve_dangling_inputs, tensor_info)

CLEAN = "__clean"


def load_genes(path):
    with open(path) as fh:
        d = json.load(fh)
    genes = d.get("best_genes") or d.get("best_cuts")
    if not genes:
        raise ValueError(f"{path}: no best_genes / best_cuts")
    placements = d.get("best_placements") or [None] * len(genes)
    return genes, placements


def derive_taps(model, genes=None, placements=None, explicit=None, marker="DUMMY"):
    nodes = model.graph.node
    L = len(nodes)
    pos = {o: k for k, n in enumerate(nodes) for o in n.output}
    per_gene = {}
    if explicit:
        taps = list(dict.fromkeys(explicit))
        missing = [t for t in taps if t not in pos]
        if missing:
            raise ValueError(f"unknown tensors: {missing}")
    else:
        taps = set()
        for gi, g in enumerate(genes, start=1):
            cuts = sorted({int(c) for c in g if 0 < int(c) < L})
            t = [nodes[c - 1].output[0] for c in cuts]
            bad = [x for x in t if marker and marker.upper() in x.upper()]
            if bad:
                print(f"  [taps] gene {gi}: {marker} cut(s) ignored: {bad}")
            t = [x for x in t if x not in bad]
            per_gene[gi] = {"gene_values": [int(v) for v in g], "taps": t,
                            "placement": placements[gi - 1] if placements else None}
            taps.update(t)
        taps = list(taps)
    return sorted(taps, key=pos.get), per_gene


def rank_units(model, genes, placements=None, marker="DUMMY"):
    g = model.graph
    nodes = g.node
    L = len(nodes)
    inits = initializer_names(g)
    graph_in = set(data_inputs(g))
    finals = {o.name for o in g.output}
    pos = {o: k for k, n in enumerate(nodes) for o in n.output}
    cons = {}
    for k, n in enumerate(nodes):
        for x in n.input:
            if x and x not in inits:
                cons.setdefault(x, []).append(k)
    units, per_gene, by_key = {}, {}, {}
    for gi, gene in enumerate(genes, start=1):
        cuts = sorted({int(c) for c in gene if 0 < int(c) < L})
        bounds = cuts + [L]
        names, bad, layers = [], [], {}
        for r, lo in enumerate(cuts):
            hi = bounds[r + 1]
            cross = [t for t in crossing_tensors(model, lo) if t not in finals]
            recv = [t for t in cross if t not in graph_in and any(lo <= k < hi for k in cons.get(t, []))]
            if not recv:
                continue
            last = nodes[lo - 1].output[0]
            tap = last if last in recv else max(recv, key=pos.get)
            if marker and marker.upper() in tap.upper():
                bad.append(tap)
                continue
            bypass = [t for t in cross if t not in recv]
            dup = [t for t in recv if any(k >= hi for k in cons.get(t, []))]
            exact = not bypass and not dup
            key = (tap, tuple(recv), tuple(bypass),
                   tuple((d, tuple(k for k in cons[d] if k >= hi)) for d in dup))
            name = tap if exact else by_key.setdefault(key, f"{tap}@{hi}")
            if name not in units:
                units[name] = {"name": name, "tap": tap, "cut": lo, "next_cut": hi, "exact": exact,
                               "side": [t for t in recv if t != tap], "bypass": bypass, "dup": dup}
            names.append(name)
            layers[name] = r + 2
        if bad:
            print(f"  [taps] gene {gi}: {marker} cut(s) ignored: {bad}")
        per_gene[gi] = {"gene_values": [int(v) for v in gene], "taps": names,
                        "placement": placements[gi - 1] if placements else None,
                        "attack_target_layer": layers}
    order = sorted(units, key=lambda u: (pos[units[u]["tap"]], units[u]["next_cut"]))
    return [units[u] for u in order], per_gene


def device_units(model, genes, placements=None, marker="DUMMY"):
    rank, rank_per_gene = rank_units(model, genes, placements, marker=marker)
    rank = {u["name"]: u for u in rank}
    g = model.graph
    nodes = g.node
    L = len(nodes)
    inits = initializer_names(g)
    graph_in = set(data_inputs(g))
    finals = {o.name for o in g.output}
    pos = {o: k for k, n in enumerate(nodes) for o in n.output}
    cons = {}
    for k, n in enumerate(nodes):
        for x in n.input:
            if x and x not in inits:
                cons.setdefault(x, []).append(k)
    units, per_gene, by_key = {}, {}, {}
    for gi, gene in enumerate(genes, start=1):
        cuts = sorted({int(c) for c in gene if 0 < int(c) < L})
        bounds = cuts + [L]
        rank_names = iter(rank_per_gene[gi]["taps"])
        names, layers = [], {}
        for r, lo in enumerate(cuts):
            hi = bounds[r + 1]
            cross = [t for t in crossing_tensors(model, lo) if t not in finals]
            recv = [t for t in cross if t not in graph_in and any(lo <= k < hi for k in cons.get(t, []))]
            if not recv:
                continue
            last = nodes[lo - 1].output[0]
            if marker and marker.upper() in (last if last in recv else max(recv, key=pos.get)).upper():
                continue
            rank_name = next(rank_names)
            start = cuts[r - 1] if r else 0
            sent = [t for t in cross if t not in graph_in and start <= pos.get(t, -1) < lo]
            if not sent:
                print(f"  [taps] gene {gi}: the device of nodes [{start}, {lo}) sends nothing downstream")
                continue
            clean = [t for t in cross if t not in sent]
            ru = rank[rank_name]
            if set([ru["tap"]] + ru["side"]) == set(sent) and set(ru["bypass"]) == set(clean) and not ru["dup"]:
                name, unit = rank_name, ru
            else:
                tap = last if last in sent else max(sent, key=pos.get)
                key = (tap, tuple(sent), tuple(clean))
                if key not in by_key:
                    name = f"{tap}@dev{start}"
                    while name in by_key.values() or name in rank:
                        name += "+"
                    by_key[key] = name
                name = by_key[key]
                unit = {"name": name, "tap": tap, "cut": lo, "next_cut": hi, "exact": not clean,
                        "side": [t for t in sent if t != tap], "bypass": clean, "dup": []}
            units.setdefault(name, unit)
            names.append(name)
            layers[name] = r + 2
        per_gene[gi] = {"gene_values": [int(v) for v in gene], "taps": names,
                        "placement": placements[gi - 1] if placements else None,
                        "attack_target_layer": layers}
    order = sorted(units, key=lambda n: (pos[units[n]["tap"]], units[n]["next_cut"]))
    return [units[n] for n in order], per_gene


SEMANTICS = {"device": device_units, "rank": rank_units}


def attack_units(model, genes, placements=None, marker="DUMMY", semantics="device"):
    if semantics not in SEMANTICS:
        raise ValueError(f"attack.semantics {semantics!r}: expected one of {sorted(SEMANTICS)}")
    return SEMANTICS[semantics](model, genes, placements, marker=marker)


def select_units(units, wanted):
    if not wanted:
        return units
    if isinstance(wanted, str):
        wanted = [w.strip() for w in wanted.split(",") if w.strip()]
    known = {u["name"] for u in units}
    missing = [w for w in wanted if w not in known]
    if missing:
        raise ValueError(f"attack.units: unknown unit(s) {missing}; known: {sorted(known)}")
    return [u for u in units if u["name"] in set(wanted)]


def cut_units(model, taps):
    side = side_inputs(model, taps)
    return [{"name": t, "tap": t, "cut": None, "next_cut": None, "exact": True, "side": side[t],
             "bypass": [], "dup": []} for t in taps]


def side_inputs(model, taps):
    pos = {o: k for k, n in enumerate(model.graph.node) for o in n.output}
    graph_outputs = {o.name for o in model.graph.output}
    out = {}
    for t in taps:
        crossing = [x for x in crossing_tensors(model, pos[t] + 1) if x not in graph_outputs]
        out[t] = [x for x in crossing if x != t]
    return out


def non_float(model, names):
    info = tensor_info(model.graph)
    return [n for n in names if n in info and info[n][0] not in (0, TensorProto.FLOAT)]


def build_full_with_taps(model_path, names, out_path):
    m = onnx.load(model_path)
    existing = {o.name for o in m.graph.output}
    for t in names:
        if t not in existing:
            m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
            existing.add(t)
    make_batch_dynamic(m)
    onnx.save(m, out_path)
    return out_path


class SuffixBuilder:
    def __init__(self, model):
        self.model = model

    def build_unit(self, unit, finals, out_path):
        if not unit["dup"]:
            return self.build(unit["tap"], unit["side"] + unit["bypass"], finals, out_path)
        m = copy.deepcopy(self.model)
        dup = set(unit["dup"])
        for k, n in enumerate(m.graph.node):
            if k >= unit["next_cut"]:
                for i, x in enumerate(n.input):
                    if x in dup:
                        n.input[i] = x + CLEAN
        vis = {vi.name: vi for vi in list(m.graph.value_info) + list(m.graph.input)}
        for t in unit["dup"]:
            vi = copy.deepcopy(vis[t])
            vi.name = t + CLEAN
            m.graph.value_info.append(vi)
        return SuffixBuilder(m).build(unit["tap"], unit["side"] + unit["bypass"]
                                      + [t + CLEAN for t in unit["dup"]], finals, out_path)

    def build(self, tap, side, final_name, out_path):
        names = [tap] + list(side)
        finals = [final_name] if isinstance(final_name, str) else list(final_name)
        nodes = reachable_nodes(self.model, names, finals)
        m = extract_submodel(self.model, nodes)
        resolve_dangling_inputs(m, self.model)
        make_batch_dynamic(m)
        ins = [vi.name for vi in m.graph.input]
        if sorted(ins) != sorted(names):
            raise ValueError(f"suffix of {tap!r}: inputs {ins}, expected {names}")
        onnx.save(m, out_path)
        return out_path


def safe_name(name):
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path
