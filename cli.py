#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ENV_BIN = os.path.join(sys.prefix, "bin")
if os.environ.get("PATH", "").split(os.pathsep)[0] != _ENV_BIN:
    os.environ["PATH"] = _ENV_BIN + os.pathsep + os.environ.get("PATH", "")

from adstain import config as C


def _csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _results(cfg, args):
    path = getattr(args, "results", None) or cfg["search"]["results"]
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run 'search' first or pass --results")
    with open(path) as fh:
        return path, json.load(fh)


def _mapping(cfg, args):
    if args.gene is not None:
        _, r = _results(cfg, args)
        i = args.gene - 1
        if not 0 <= i < len(r["best_cuts"]):
            raise SystemExit(f"--gene must be in 1..{len(r['best_cuts'])}")
        return [int(c) for c in r["best_cuts"][i]], list(r["best_placements"][i])
    if not args.cuts:
        raise SystemExit("give --cuts [--placement] or --gene")
    cuts = [int(c) for c in _csv(args.cuts)]
    placement = _csv(args.placement) if args.placement else [r.name for r in C.resources(cfg)][:len(cuts)]
    return cuts, placement


def cmd_front(cfg, args):
    path, r = _results(cfg, args)
    print(f"{path}: {len(r['best_cuts'])} Pareto solution(s)")
    for i, (c, p, o) in enumerate(zip(r["best_cuts"], r["best_placements"], r["best_objectives"]), 1):
        print(f"gene {i:3d}  {o[0]:9.3f} img/s  {o[1]:8.1f} MB  "
              f"--cuts {','.join(map(str, c))} --placement {','.join(p)}")


def cmd_show(cfg, args):
    import yaml
    if args.resolve:
        from adstain.model import prepare
        prepare(cfg)
    cfg = C.public(cfg)
    cfg["platform"] = dict(cfg["platform"], parsed=[repr(r) for r in C.resources(cfg)])
    print(yaml.safe_dump(cfg, sort_keys=False))


def cmd_inspect(cfg, args):
    from adstain import analytical as am
    from adstain.dse import prepare
    from adstain.onnx_graph import crossing_tensors, forbidden_cuts, model_input_hw
    path, model, task = prepare(cfg)
    costs = am.LayerCosts(path)
    forb = forbidden_cuts(model, cfg["autodice"]["dummy_marker"], cfg["search"]["dummy_offsets"])
    print(f"model {cfg['model']}\nformatted {path}")
    outs = [o.name for o in model.graph.output] if task.detection else [model.graph.output[0].name]
    print(f"input {model_input_hw(model, cfg['preprocess']['input_hw'])}, "
          f"output{'s' if len(outs) > 1 else ''} {', '.join(outs)}, {costs.n} layers")
    pp = cfg["preprocess"]
    print(f"preprocess {pp['channel_order']}, x*{pp['pixel_scale']:g}, mean {pp['mean']}, std {pp['std']}, "
          + ("letterbox (pad " + str(pp.get("pad_value", 114)) + ")" if pp.get("letterbox") else
             f"short side {pp['resize_short']} + center crop" if pp["resize_short"] else "direct resize"))
    print(f"resources: {C.resources(cfg)}")
    if args.layers:
        print(f"\n{'idx':>4} {'name':<40} {'op':<20} {'GMAC':>8} {'w MB':>8} {'out MB':>8} cut")
        for j in range(costs.n):
            cross = crossing_tensors(model, j) if j else []
            print(f"{j:4d} {costs.names[j][:40]:<40} {costs.op_types[j]:<20} "
                  f"{costs.macs[j] / 1e9:8.4f} {costs.m_coeffs[j] * 4 / 1e6:8.2f} "
                  f"{costs.m_outbuffs[j] * 4 / 1e6:8.3f} "
                  + ("forbidden" if j in forb else f"{len(cross)} tensor(s)" if j else ""))
    print(f"\ntotal {costs.macs.sum() / 1e9:.3f} GMAC, {costs.m_coeffs.sum() * 4 / 1e6:.1f} MB of weights")
    multi = [j for j in range(1, costs.n) if len(crossing_tensors(model, j)) > 1]
    print(f"forbidden cut positions: {sorted(forb) or 'none'}")
    print(f"cut positions crossed by more than one tensor (skip connections): {len(multi)}")
    if args.cuts:
        cuts = [int(c) for c in _csv(args.cuts)]
        for c in cuts[1:]:
            print(f"cut {c}: {crossing_tensors(model, c)}")
        terms = costs.segment_terms(cuts)
        for s, t in enumerate(terms):
            print(f"segment {s}: {t['n_layers']:.0f} layers, {t['macs']:.3f} GMAC, "
                  f"recv {t['recv_mb']:.3f} MB, send {t['send_mb']:.3f} MB, weights {t['coeffs_mb']:.1f} MB")


def _ncnn_check(cfg, path, model, task):
    from adstain.ncnn_check import check
    try:
        check(cfg, path, model, task)
    except Exception as e:
        print(f"[ncnn-check] could not run: {str(e).splitlines()[0]}")


def cmd_check_ncnn(cfg, args):
    from adstain.dse import prepare
    from adstain.ncnn_check import check
    path, model, task = prepare(cfg)
    res = check(cfg, path, model, task, image=args.image, force=True)
    if res is not None and not res["ok"]:
        raise SystemExit(1)


def cmd_deploy(cfg, args):
    from adstain.deploy import Deployer
    from adstain.dse import prepare
    path, model, task = prepare(cfg)
    _ncnn_check(cfg, path, model, task)
    res = C.resources(cfg)
    cuts, placement = _mapping(cfg, args)
    d = Deployer(cfg, model, res, task=task)
    if args.no_bench:
        stages = d.build(cuts, placement, mode="infer")
        with open(os.path.join(d.dir, "single.txt"), "w") as fh:
            fh.write((args.image or cfg["autodice"]["bench_image"]) + "\n")
        d.run(len(stages), "single.txt", env={"RESULTS_FILE": "single.csv"})
        log = open(os.path.join(d.dir, "run.log")).read().splitlines()
        print("\n".join(l for l in log if l.startswith(("[", "rank"))))
        return
    m = d.measure(cuts, placement, args.image)
    for r, name in enumerate(placement):
        print(f"rank {r} {name:<24} compute {m['compute_ms'][r]:9.2f} ms  wall {m['wall_ms'][r]:9.2f} ms  "
              f"rss {m['memory_mb'][r]:8.1f} MB")
    print(f"throughput {1000.0 / max(m['wall_ms']):.3f} img/s, max memory {max(m['memory_mb']):.1f} MB")
    print(f"files: {d.dir}")


def _shell_env(env):
    return " ".join(f"{k}={v}" for k, v in env.items())


def cmd_build(cfg, args):
    from adstain.deploy import Deployer
    from adstain.dse import prepare
    from adstain.mpi_attack import make_image_lists
    path, model, task = prepare(cfg)
    _ncnn_check(cfg, path, model, task)
    res = C.resources(cfg)
    cuts, placement = _mapping(cfg, args)
    d = Deployer(cfg, model, res, args.out or os.path.join(cfg["workdir"], "build"), task=task)
    stages = d.build(cuts, placement, mode="infer")
    lists = make_image_lists(cfg, d.dir, args.images, args.history if args.history is not None
                             else cfg["attack"]["n_buffer"], task)
    with open(os.path.join(d.dir, "buffer_manifest.json")) as fh:
        manifest = json.load(fh)

    print(f"\nbuilt {d.dir}/multinode  ({len(stages)} ranks)")
    for st in stages:
        print(f"  rank {st.rank} {st.resource.name:<22} layers {st.nodes[0]} .. {st.nodes[-1]} "
              f"({len(st.nodes)} nodes)")
    print("\nattackable cuts (ATTACK_TARGET_LAYER = receiving rank + 1):")
    for layer in sorted({e["attack_target_layer"] for e in manifest}):
        es = [e for e in manifest if e["attack_target_layer"] == layer]
        print(f"  ATTACK_TARGET_LAYER={layer}  cut {es[0]['cut']}: "
              + ", ".join(f"{e['buffer_name']} {e['shape'][1:]}" for e in es))
    if lists:
        print(f"\nimage lists: test_images.txt ({lists[0]}), history_images.txt ({lists[1]})")
    else:
        print("\nno attack.csv in the config: write test_images.txt yourself (one path per line)")

    mpi = " ".join([cfg["autodice"]["mpirun"]] + [str(a) for a in cfg["autodice"]["mpirun_args"]]
                   + ["-np", str(len(stages)), "-rf", "rankfile", "./multinode"])
    me = f"python {os.path.abspath(__file__)} -c {cfg['config_path']}" + "".join(
        f" --set {s}" for s in (args.set_main or []) + (args.set or []))
    layer = manifest[0]["attack_target_layer"] if manifest else 2
    print(f"""
next steps:
  cd {d.dir}
  {_shell_env({"DUMP_BUFFER": 1, "FM_DIR": "fm", "RESULTS_FILE": "clean.csv"})} {mpi} test_images.txt
  {_shell_env({"DUMP_BUFFER": 1, "FM_DIR": "fm_hist", "RESULTS_FILE": "history.csv"})} {mpi} history_images.txt
  {me} payload --layer {layer} [--tag <attack tag>]
  {_shell_env({"ATTACK_ENABLED": 1, "ATTACK_TARGET_LAYER": layer, "FM_DIR": "fm", "RESULTS_FILE": "attacked.csv"})} {mpi} test_images.txt
  {me} score --layer {layer}""")


def cmd_payload(cfg, args):
    from adstain.mpi_attack import build_payload
    build_payload(cfg, args.dir or os.path.join(cfg["workdir"], "build"), args.layer, args.tag,
                  args.fm, args.history_fm, side=not args.no_side, device=args.device)


def cmd_score(cfg, args):
    from adstain.mpi_attack import score
    from adstain.attacks import build_configs
    d = args.dir or os.path.join(cfg["workdir"], "build")
    labels = {c["tag"]: c["label"] for c in build_configs(cfg["attack"]["configs"])}
    row = score(d, args.clean, args.attacked, args.layer, args.tag, labels.get(args.tag),
                plot=not args.no_plot)
    plots = row.pop("plots", [])
    for k, v in row.items():
        print(f"  {k:32s} {v}")
    print(f"saved in {os.path.join(d, 'scores.csv')}")
    for p in plots:
        print(f"plot: {p}")


def cmd_run_binary(cfg, args):
    from adstain.mpi_attack import run_binary
    run_binary(cfg, args.dir or os.path.join(cfg["workdir"], "build"), args.kind, args.layer,
               args.results_file, args.list, args.fm)


def cmd_sweep(cfg, args):
    from adstain.mpi_attack import sweep
    rows = sweep(cfg, args.dir or os.path.join(cfg["workdir"], "build"),
                 [int(x) for x in _csv(args.layers)] if args.layers else None,
                 _csv(args.tags) if args.tags else None, args.device, side=not args.no_side)
    from adstain.mpi_attack import quality_change
    print("\nlayer  tag                          ASR %   clean -> attacked")
    for r in rows:
        print(f"{r['layer']:>5}  {r['tag']:<28} {r.get('attack_success_rate_pct', float('nan')):6}   "
              f"{quality_change(r)}")


def cmd_mpi_pareto(cfg, args):
    from adstain.dse import prepare
    from adstain.mpi_attack import pareto_campaign
    _ncnn_check(cfg, *prepare(cfg))
    pareto_campaign(cfg, [int(x) for x in _csv(args.genes)] if args.genes else None,
                    _csv(args.tags) if args.tags else None, args.images, args.history,
                    args.device, side=not args.no_side, keep=args.keep_files,
                    results_path=args.results, chunk=args.chunk)


def _load_records(paths):
    from adstain import analytical as am
    recs = []
    for p in paths:
        if p.endswith(".jsonl"):
            with open(p) as fh:
                for line in fh:
                    r = json.loads(line)
                    if "metrics" in r:
                        recs.append(dict(cuts=r["cuts"], placement=r["placement"], **r["metrics"]))
        else:
            with open(p) as fh:
                recs += am.records_from_history(json.load(fh).get("raw_history", []))
    return am.records_from_history(recs)


def cmd_calibrate(cfg, args):
    from adstain import analytical as am
    from adstain.dse import calibrate_from_measurements, make_measure_evaluator, prepare
    from adstain.onnx_graph import forbidden_cuts
    path, model, task = prepare(cfg)
    res = C.resources(cfg)
    costs = am.LayerCosts(path)
    sc = cfg["search"]
    if args.history:
        records = _load_records(args.history)
        if len(records) < 2:
            raise SystemExit(f"only {len(records)} usable measurement(s)")
        print(f"{len(records)} measured mapping(s)")
        model_ = am.calibrate(costs, res, records, sc["ridge_lambda"])
        am.cross_validate(costs, res, records, sc["ridge_lambda"])
        model_.save(sc["calibration"])
        print(f"written {sc['calibration']}")
        return
    if args.plan_only:
        for j, (cuts, pl) in enumerate(am.plan_calibration(costs, res, sc["min_segment"],
                                                           sc["calib_runs"])):
            print(f"{j:2d} cuts={cuts} placement={pl}")
        return
    order = res
    if sc["resource_order"]:
        by = {r.name: r for r in res}
        order = [by[n] for n in sc["resource_order"]]
    ev = make_measure_evaluator(cfg, path, model, order, task)
    forb = forbidden_cuts(model, cfg["autodice"]["dummy_marker"], sc["dummy_offsets"])
    calibrate_from_measurements(cfg, costs, order, ev, forb, sc["calibration"])


def _status(cfg, phase, what):
    from adstain.report import runs_dir
    path = runs_dir(cfg) / "pipeline_status.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {cfg['name']} {phase} {what}\n")
    except OSError:
        pass


def _logged(cfg, phase, fn, *a, **kw):
    _status(cfg, phase, "start")
    try:
        out = fn(*a, **kw)
    except BaseException:
        _status(cfg, phase, "FAILED")
        raise
    _status(cfg, phase, "done")
    return out


def cmd_search(cfg, args):
    from adstain.dse import prepare, search
    if not args.dry_run and args.levels != "analytic":
        _ncnn_check(cfg, *prepare(cfg))
    _logged(cfg, "search", search, cfg, levels=args.levels, dry=args.dry_run)


def cmd_attack(cfg, args):
    from adstain.campaign import run
    _logged(cfg, "attack", run, cfg, taps=_csv(args.taps) if args.taps else None)
    if not args.taps and not cfg["attack"]["taps"]:
        try:
            from adstain.report import ranking
            ranking.write(cfg)
        except (Exception, SystemExit) as e:
            print(f"[rank] skipped: {e} (rerun with: cli.py rank -c <config>)")
    if cfg["report"]["enabled"] and not args.no_report:
        sets = (args.set_main or []) + (args.set or [])
        cmd = [sys.executable, os.path.abspath(__file__), "report", "-c", cfg["config_path"], "--after-attack"]
        cmd += [x for s in sets for x in ("--set", s)]
        print(f"\n[report] {' '.join(cmd)}", flush=True)
        sys.stderr.flush()
        os.execv(sys.executable, cmd)


def cmd_rank(cfg, args):
    from adstain.report import ranking
    ranking.write(cfg, attack_dirs=_csv(args.attack_dirs) if args.attack_dirs else None,
                  tags=_csv(args.tags) if args.tags else None,
                  band="default" if args.band is None else args.band,
                  out=args.out and os.path.abspath(args.out))
    out = args.out or os.path.join(cfg["attack"]["out"], "ranking")
    with open(os.path.join(out, "summary.txt")) as fh:
        print(fh.read())


def cmd_report(cfg, args):
    from adstain.report import generate
    rp = cfg["report"]
    for key, val in (("stealth_band", args.band), ("extra_bands", args.extra_bands), ("collect", args.collect),
                     ("success_asr", args.success_asr),
                     ("runs", args.runs and os.path.abspath(args.runs)),
                     ("out", args.out and os.path.abspath(args.out))):
        if val is not None:
            rp[key] = val
    th = rp["threshold"]
    for key, val in (("mode", args.threshold), ("window", args.window), ("false_alarm", args.false_alarm),
                     ("source", args.calibration_source),
                     ("validation", args.validation and os.path.abspath(args.validation))):
        if val is not None:
            th[key] = val
    if args.validation and args.calibration_source is None:
        th["source"] = "validation"
    if args.no_stealth_study:
        rp["stealth_study"] = False
    if not generate(cfg):
        if args.after_attack:
            print("[report] failed (the attack results are saved); rerun with: cli.py report -c <config>")
            return
        raise SystemExit(1)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", help="YAML configuration")
    common.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override a config value, e.g. --set attack.batch=16")
    ap = argparse.ArgumentParser(description="AutoDiCE partitioning search + StAIn GPU attacks")
    ap.add_argument("-c", "--config", dest="config_main", help="YAML configuration")
    ap.add_argument("--set", action="append", dest="set_main", metavar="KEY=VALUE")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, **kw):
        return sub.add_parser(name, parents=[common], **kw)

    p = add("show-config", help="print the resolved configuration")
    p.add_argument("--resolve", action="store_true",
                   help="analyse the model first (task, model-family preprocessing defaults)")

    p = add("inspect", help="model layers, costs, cut positions")
    p.add_argument("--layers", action="store_true", help="print every layer")
    p.add_argument("--cuts", help="comma-separated cuts (first is 0) to analyse")

    p = add("front", help="list the Pareto solutions of the search results")
    p.add_argument("--results", help="results JSON (default search.results)")

    p = add("deploy", help="generate, compile and run one mapping with AutoDiCE")
    p.add_argument("--cuts", help="first layer of each segment, e.g. 0,120,300,420")
    p.add_argument("--placement", help="resource per segment (default: config order)")
    p.add_argument("--gene", type=int, help="take cuts/placement from Pareto solution N (see 'front')")
    p.add_argument("--results", help="results JSON for --gene (default search.results)")
    p.add_argument("--image", help="input image (default: autodice.bench_image)")
    p.add_argument("--no-bench", action="store_true", help="single inference, print the prediction")

    p = add("build", help="generate + compile one partition with env-controlled dump/attack hooks")
    p.add_argument("--cuts", help="first layer of each segment, e.g. 0,120,300,420")
    p.add_argument("--placement", help="resource per segment (default: config order)")
    p.add_argument("--gene", type=int, help="take cuts/placement from Pareto solution N (see 'front')")
    p.add_argument("--results", help="results JSON for --gene (default search.results)")
    p.add_argument("--images", type=int, default=100, help="test images listed from attack.csv")
    p.add_argument("--history", type=int, help="history images for mFGSM (default attack.n_buffer)")
    p.add_argument("--out", help="output directory (default <workdir>/build)")

    p = add("payload", help="attack the dumped tensors of one cut (for ATTACK_ENABLED runs)")
    p.add_argument("--layer", type=int, required=True, help="ATTACK_TARGET_LAYER value")
    p.add_argument("--tag", help="attack configuration tag (default: first of attack.configs)")
    p.add_argument("--dir", help="build directory (default <workdir>/build)")
    p.add_argument("--fm", default="fm", help="FM_DIR of the clean dump")
    p.add_argument("--history-fm", default="fm_hist", help="FM_DIR of the history dump")
    p.add_argument("--no-side", action="store_true", help="residual cuts: leave the side branch clean")
    p.add_argument("--device", help="cpu or cuda (default: cuda if available)")

    p = add("score", help="paired ASR between a clean and an attacked run of the binary")
    p.add_argument("--dir", help="build directory (default <workdir>/build)")
    p.add_argument("--clean", default="clean.csv")
    p.add_argument("--attacked", default="attacked.csv")
    p.add_argument("--layer", type=int)
    p.add_argument("--tag")
    p.add_argument("--no-plot", action="store_true", help="skip the figures")

    p = add("run-binary", help="run the built multinode binary with the dump/attack environment")
    p.add_argument("--kind", required=True, choices=["clean", "history", "attacked", "plain"],
                   help="clean: dump test tensors; history: dump history tensors; attacked: inject payload")
    p.add_argument("--layer", type=int, help="ATTACK_TARGET_LAYER (attacked runs)")
    p.add_argument("--results-file", help="RESULTS_FILE (default clean.csv / history.csv / attacked.csv)")
    p.add_argument("--list", help="image list (default test_images.txt or history_images.txt)")
    p.add_argument("--fm", help="FM_DIR (default fm or fm_hist)")
    p.add_argument("--dir", help="build directory (default <workdir>/build)")

    p = add("sweep", help="payload + attacked run + score for several cuts and attacks")
    p.add_argument("--layers", help="ATTACK_TARGET_LAYER values (default: all cuts of the build)")
    p.add_argument("--tags", help="attack tags (default: all of attack.configs)")
    p.add_argument("--device", help="cpu or cuda for the payloads")
    p.add_argument("--no-side", action="store_true")
    p.add_argument("--dir", help="build directory (default <workdir>/build)")

    p = add("mpi-pareto", help="real-deployment attack on every Pareto solution (build, runs, sweep)")
    p.add_argument("--genes", help="Pareto solutions to attack, e.g. 1,3,5 (default: all)")
    p.add_argument("--tags", help="attack tags (default: all of attack.configs)")
    p.add_argument("--images", type=int, default=500, help="test images per solution")
    p.add_argument("--history", type=int, help="history images for mFGSM (default attack.n_buffer)")
    p.add_argument("--device", help="cpu or cuda for the payloads")
    p.add_argument("--no-side", action="store_true")
    p.add_argument("--keep-files", action="store_true",
                   help="keep the dumps and sub-model weights of each solution (several GB each)")
    p.add_argument("--results", help="search results JSON (default search.results)")
    p.add_argument("--chunk", type=int,
                   help="images dumped and attacked at a time (default: the most that fits on the disk)")

    p = add("calibrate", help="calibrate the analytical model")
    p.add_argument("--history", nargs="+", help="results JSON or measurements.jsonl to fit on")
    p.add_argument("--plan-only", action="store_true", help="print the mappings to measure")

    p = add("search", help="hierarchical NSGA-II Pareto search (throughput, memory)")
    p.add_argument("--levels", choices=["analytic+measure", "analytic", "measure"])
    p.add_argument("--dry-run", action="store_true", help="synthetic metrics, nothing deployed")

    p = add("check-ncnn", help="compare the AutoDiCE ncnn build with onnxruntime layer by layer")
    p.add_argument("--image", help="image to run (default: autodice.bench_image)")

    p = add("attack", help="StAIn attack campaign on the GPU, then the report (report.enabled)")
    p.add_argument("--taps", help="explicit cut tensors (default: from the search results)")
    p.add_argument("--no-report", action="store_true", help="skip the figures at the end")

    p = add("rank", help="robustness ranking of the Pareto partitionings after a campaign: whole budget, "
                         "per attack and per strength (also run at the end of attack)")
    p.add_argument("--tags", help="attack configurations of the budget, e.g. mgna_g0p5_s1,mfgsm_eps_0p5 "
                                  "(default: every configuration of the campaign results)")
    p.add_argument("--attack-dirs", help="campaign folders to merge, relative to the workdir; a later one wins "
                                         "(default: report.attack_dirs, then attack.out)")
    p.add_argument("--band", type=float,
                   help="also rank with only the attacks whose shift Δ stays within this %% of the clean std "
                        "(default report.stealth_band; 0 = off)")
    p.add_argument("--out", help="output folder (default <attack.out>/ranking)")

    p = add("report", help="campaign tables and stealth study for every model of the runs folder")
    p.add_argument("--band", type=float,
                   help="stealth tolerance in %% of the clean std (default report.stealth_band = 10)")
    p.add_argument("--extra-bands", type=float, nargs="+", help="more tolerances in the campaign stealth tables")
    p.add_argument("--success-asr", type=float,
                   help="ASR (%%) from which an attack is successful: fragile layer = stealthy AND ASR >= this "
                        "(default report.success_asr = 50)")
    p.add_argument("--collect", choices=["auto", "always", "never"],
                   help="feature-map statistics of this config's model (default report.collect = auto)")
    p.add_argument("--no-stealth-study", action="store_true", help="campaign tables only (no GPU pass)")
    p.add_argument("--runs", help="folder of the model workdirs (default: the parent of workdir)")
    p.add_argument("--out", help="output folder (default <runs>/report)")
    p.add_argument("--threshold", choices=["fixed", "calibrated", "fixed_window"],
                   help="fixed: ±band on every cut (halves of the test set); fixed_window: ±band on every cut over "
                        "windows of --window inferences; calibrated: one threshold per model and cut, set on clean images "
                        "for --false-alarm false alarms over windows of --window inferences (report.threshold)")
    p.add_argument("--window", type=int, help="calibrated: inferences per monitoring window (default 1000)")
    p.add_argument("--false-alarm", type=float, help="calibrated: target false-alarm rate per cut (default 0.01)")
    p.add_argument("--calibration-source", choices=["test", "validation"],
                   help="calibrated: a held-out share of the test set, or a separate validation image set")
    p.add_argument("--validation", help="calibrated: clean validation images (folder, .txt list or .csv)")
    p.add_argument("--after-attack", action="store_true", help=argparse.SUPPRESS)

    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    path = args.config or args.config_main
    if not path:
        ap.error("-c/--config is required")
    cfg = C.load(path, (args.set_main or []) + (args.set or []))
    {"show-config": cmd_show, "inspect": cmd_inspect, "front": cmd_front, "deploy": cmd_deploy,
     "build": cmd_build, "run-binary": cmd_run_binary, "sweep": cmd_sweep,
     "mpi-pareto": cmd_mpi_pareto,
     "payload": cmd_payload, "score": cmd_score, "calibrate": cmd_calibrate, "search": cmd_search,
     "attack": cmd_attack, "rank": cmd_rank, "report": cmd_report,
     "check-ncnn": cmd_check_ncnn}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
