"""
The `mns` command line. Every command prints one JSON object on stdout ({"ok": true, ...} or {"ok": false,
"error", "hint"}); everything else the simulator and its libraries print goes to <study>/.mns/cli.log.

    mns <command> [arguments] [--why "reason"] [--background] [--max-minutes N] [--no-render]

Run `mns help` for the list of commands, `mns help <command>` for one.
"""
import argparse
import io
import json
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

HELP = {
    "status": "mns status - the study at a glance: designs, running jobs, the last events.",
    "help": "mns help [command] - this list, or one command's usage. mns help properties [role] [--space neuron|"
            "activation] - every role's properties, defaults, units and choices.",
    "new-study": "mns new-study <folder> [--brief brief.md] [--from file.mns|.mad|.mfn ...] [--name text] - make a "
                 "study folder (agent files, imported starting designs).",
    "design": "mns design list | show <ID> [--spec] | new <name> --space neuron|activation|full [--template T] | "
              "derive <ID> <new name> --set p=v [--set Body.prop=v ...] [--no-build] | import <file> [--name n]",
    "templates": "mns templates - the design templates for `mns design new --template`.",
    "cad": "mns cad build <ID> | inspect <ID> | render <ID> [--views iso,front,top,right,section-x:0.5] "
           "[--exploded 0.3] [--only Body1,Body2] [--name file]",
    "check": "mns check <ID> - mesh and assemble the model: which chamber loads which membrane, inputs, paths, links, "
             "flow connections, warnings (no solve).",
    "solve": "mns solve <ID> [--set Part.field=value ...] - one static solve (neuron or full neuron) at the design's "
             "pressures; --set changes values for this run only.",
    "sweep": "mns sweep <ID> --input Chamber=from:to:n [--input Chamber2=from:to:n] [--preactivation Chamber|none] "
             "[--tolerance 1.0] [--method lowest|greedy] [--set ...] - neuron sweep, weights and neuron equation.",
    "fit": "mns fit <ID> [--tolerance 0.5] [--method lowest|greedy] - refit the neuron equation from the stored sweep.",
    "dpsweep": "mns dpsweep <ID> [--dp from:to:n] [--set study.field=v ...] - activation-function study: the Δp sweep "
               "that maps the membrane's pressure difference to tube area, flow and outputs (saved in model.mad).",
    "characterise": "mns characterise <ID> --axis Part.field=from:to:n [--axis ...] [--record key,key] [--set ...] - "
                    "full neuron over a grid of chamber parameters (stored in model.mfn).",
    "compare": "mns compare [ID ...] [--metrics m1,m2] [--x parameter --y metric] - a table of the designs' "
               "parameters and metrics; --x/--y also plot one against the other.",
    "report": "mns report design <ID> | study [--build] - write the generated LaTeX parts (tables, figures, equation); "
              "--build compiles the PDF.",
    "note": "mns note \"text\" [--design ID] - post a message to the study log (shown live in the GUI).",
    "jobs": "mns jobs - recent background jobs and their state.",
    "wait": "mns wait <job> [--timeout 540] - wait for a background job and print its output.",
    "cancel": "mns cancel <job> - stop a background job.",
}
LONG = ("solve", "sweep", "dpsweep", "characterise", "fit")


class Failure(Exception):
    pass


def build_parser():
    p = argparse.ArgumentParser(prog="mns", add_help=False)
    p.add_argument("command", nargs="?", default="help")
    p.add_argument("args", nargs="*")
    p.add_argument("--why", default="")
    p.add_argument("--background", action="store_true")
    p.add_argument("--max-minutes", type=float, default=None)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--input", action="append", default=[])
    p.add_argument("--axis", action="append", default=[])
    p.add_argument("--record", default=None)
    p.add_argument("--preactivation", default=None)
    p.add_argument("--tolerance", type=float, default=None)
    p.add_argument("--method", default="lowest")
    p.add_argument("--dp", default=None)
    p.add_argument("--space", default=None)
    p.add_argument("--template", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--views", default=None)
    p.add_argument("--exploded", type=float, default=0.0)
    p.add_argument("--only", default=None)
    p.add_argument("--spec", action="store_true")
    p.add_argument("--no-build", action="store_true")
    p.add_argument("--build", action="store_true")
    p.add_argument("--metrics", default=None)
    p.add_argument("--x", default=None)
    p.add_argument("--y", default=None)
    p.add_argument("--timeout", type=float, default=540.0)
    p.add_argument("--design", default=None)
    p.add_argument("--brief", default=None)
    p.add_argument("--from", dest="sources", action="append", default=[])
    p.add_argument("--job-output", default=None)
    p.add_argument("-h", "--help", action="store_true")
    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    real_fd = os.dup(1)
    log_path = None
    try:
        args, unknown = build_parser().parse_known_args(argv)
    except SystemExit:
        _print(real_fd, {"ok": False, "error": "could not read the arguments", "hint": "mns help"})
        return 2
    if unknown:
        _print(real_fd, {"ok": False, "error": f"unknown option(s): {' '.join(unknown)}",
                         "hint": HELP.get(args.command, "mns help")})
        return 2
    study = None
    try:
        from .study import Study
        if args.command not in ("help", "new-study"):
            study = Study.find()
            log_path = study.root / ".mns" / "cli.log"
    except Exception as exc:
        if args.command not in ("help", "new-study", "templates"):
            _print(real_fd, _error(exc))
            return 1
    _redirect(log_path)
    warnings.filterwarnings("ignore")
    started = time.time()
    job = os.environ.get("MNS_JOB")
    events = study.events if study is not None else None
    design_id = _design_arg(args)
    try:
        if args.help:
            result = {"usage": HELP.get(args.command, "mns help")}
        elif args.background and args.command in LONG and not job:
            from . import jobs
            forwarded = [a for a in argv if a != "--background"]
            job_id = jobs.start(study, forwarded, design=design_id, why=args.why)
            events.emit("command", "mns " + " ".join(forwarded), design=design_id, why=args.why or None,
                        job=job_id, background=True)
            result = {"job": job_id, "state": "running",
                      "hint": f"mns wait {job_id} (blocks up to --timeout s, default 540) to get the result."}
        else:
            if events is not None and args.command not in ("status", "help", "jobs", "wait", "note", "templates"):
                events.emit("command", "mns " + " ".join(a for a in argv if not a.startswith("--job-output")
                                                         and a != args.job_output),
                            design=design_id, why=args.why or None)
            result = dispatch(args, study)
            if events is not None and args.command not in ("status", "help", "jobs", "wait", "note", "templates"):
                events.emit("done", _one_line(args, result), design=design_id, seconds=round(time.time() - started, 1),
                            files=_files(result))
        payload = {"ok": True, **(result if isinstance(result, dict) else {"result": result})}
        code = 0
    except Exception as exc:  # reported as JSON, details in the log
        traceback.print_exc()
        payload = _error(exc)
        code = 1
        if events is not None:
            events.emit("error", payload["error"], design=design_id)
    if study is not None:
        payload = _relative(payload, str(study.root))
    if args.job_output:
        from .util import write_json
        write_json(args.job_output, payload)
        if study is not None and job:
            from . import jobs
            jobs.finish(study, job, "done" if code == 0 else "failed")
    _print(real_fd, payload)
    return code


def _relative(x, root):
    """Paths inside the study written relative to it (shorter output)."""
    if isinstance(x, dict):
        return {k: _relative(v, root) for k, v in x.items()}
    if isinstance(x, list):
        return [_relative(v, root) for v in x]
    if isinstance(x, Path):
        x = str(x)
    if isinstance(x, str) and len(x) > len(root) and x.lower().startswith(root.lower()) and x[len(root)] in "\\/":
        return x[len(root) + 1:].replace("\\", "/")
    return x


def _redirect(log_path):
    """Send everything printed by the simulator and libraries (C level too) to the log, keep stdout for JSON."""
    target = open(log_path, "ab") if log_path else open(os.devnull, "wb")
    sys.stdout.flush()
    sys.stderr.flush()
    os.dup2(target.fileno(), 1)
    os.dup2(target.fileno(), 2)
    sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", closefd=False), encoding="utf-8", errors="replace",
                                  line_buffering=True)
    sys.stderr = io.TextIOWrapper(os.fdopen(2, "wb", closefd=False), encoding="utf-8", errors="replace",
                                  line_buffering=True)


def _print(fd, payload):
    from .util import jsonable
    text = json.dumps(jsonable(payload), ensure_ascii=False, separators=(", ", ": ")) + "\n"
    os.write(fd, text.encode("utf-8"))


def _error(exc):
    from .util import ApiError
    if isinstance(exc, ApiError):
        return {"ok": False, "error": str(exc), **({"hint": exc.hint} if exc.hint else {})}
    if isinstance(exc, TimeoutError):
        return {"ok": False, "error": str(exc), "hint": "Use fewer points, or a larger --max-minutes."}
    if isinstance(exc, (ValueError, KeyError, FileNotFoundError)):
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "hint": "Check the design file and the arguments (details in .mns/cli.log)."}
    return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
            "hint": "Unexpected error inside the simulator; details are in .mns/cli.log. Record it in "
                    "feature_requests.md if it blocks you."}


def _design_arg(args):
    if args.command in ("check", "solve", "sweep", "fit", "dpsweep", "characterise") and args.args:
        return args.args[0][:4]
    if args.command in ("cad",) and len(args.args) > 1:
        return args.args[1][:4]
    if args.command == "design" and len(args.args) > 1 and args.args[0] in ("show", "derive"):
        return args.args[1][:4]
    if args.command == "report" and len(args.args) > 1 and args.args[0] == "design":
        return args.args[1][:4]
    if args.command == "note":
        return args.design
    return None


def _one_line(args, result):
    if not isinstance(result, dict):
        return f"{args.command} done"
    bits = [args.command]
    for key in ("converged", "points", "met", "max_error_kPa", "seconds", "status"):
        if key in result:
            bits.append(f"{key}={result[key]}")
    eq = result.get("equation")
    if isinstance(eq, dict) and "max_error_kPa" in eq:
        bits.append(f"equation error {eq['max_error_kPa']} kPa (met={eq['met']})")
    if result.get("warnings"):
        bits.append(f"{len(result['warnings'])} warning(s)")
    return " ".join(str(b) for b in bits)


def _files(result):
    if not isinstance(result, dict):
        return None
    files = []

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "files" and isinstance(v, dict):
                    files.extend(str(f) for f in v.values() if isinstance(f, (str, Path)))
                else:
                    walk(v)
    walk(result)
    return files or None


def _need(args, n, usage):
    if len(args.args) < n:
        from .util import ApiError
        raise ApiError("missing argument(s)", usage)
    return args.args


def _sets(args):
    from .util import parse_assignment
    return [parse_assignment(s) for s in args.set]


def dispatch(args, study):
    from .util import ApiError, parse_range
    c = args.command
    if c == "help":
        if args.args and args.args[0] == "properties":
            from .model import property_reference
            return property_reference(args.space, args.args[1] if len(args.args) > 1 else None)
        if args.args:
            return {"usage": HELP.get(args.args[0], f"unknown command {args.args[0]!r}")}
        return {"commands": HELP, "manual": "MANUAL.md in the study folder explains everything in detail."}
    if c == "new-study":
        from .workspace import create_study
        _need(args, 1, HELP["new-study"])
        return create_study(args.args[0], brief=args.brief, sources=args.sources, name=args.name)
    if c == "templates":
        from .templates import list_templates
        return {"templates": list_templates()}
    if c == "status":
        return status(study)
    if c == "note":
        _need(args, 1, HELP["note"])
        study.events.emit("note", " ".join(args.args), design=args.design)
        return {"posted": True}
    if c == "jobs":
        from . import jobs
        return {"jobs": jobs.listing(study)}
    if c == "wait":
        from . import jobs
        _need(args, 1, HELP["wait"])
        return jobs.wait(study, args.args[0], args.timeout)
    if c == "cancel":
        from . import jobs
        _need(args, 1, HELP["cancel"])
        return jobs.cancel(study, args.args[0])
    if c == "design":
        return design_command(args, study)
    if c == "cad":
        return cad_command(args, study)
    if c == "compare":
        from .compare import compare
        return compare(study, args.args, metrics=args.metrics.split(",") if args.metrics else None, x=args.x, y=args.y)
    if c == "report":
        from .report import report_design, report_study
        _need(args, 1, HELP["report"])
        if args.args[0] == "design":
            _need(args, 2, HELP["report"])
            return report_design(study.design(args.args[1]), build=args.build)
        if args.args[0] == "study":
            return report_study(study, build=args.build)
        raise ApiError(f"report {args.args[0]}?", HELP["report"])

    from . import ops
    _need(args, 1, HELP.get(c, "mns help"))
    design = study.design(args.args[0])
    render = not args.no_render
    if c == "check":
        return {"neuron": ops.neuron_check, "activation": ops.activation_check, "full": ops.full_check}[design.space](design)
    if c == "solve":
        if design.space == "neuron":
            return ops.neuron_solve(design, _sets(args), render, args.max_minutes)
        if design.space == "full":
            return ops.full_run(design, (), None, _sets(args), args.max_minutes, render)
        raise ApiError("An activation design is simulated with mns dpsweep.", HELP["dpsweep"])
    if c == "sweep":
        if design.space != "neuron":
            raise ApiError(f"mns sweep is for neuron designs; {design.id} is a {design.space} design.",
                           HELP["dpsweep"] if design.space == "activation" else HELP["characterise"])
        if not args.input:
            raise ApiError("Give at least one --input Chamber=from:to:n", HELP["sweep"])
        inputs = []
        for text in args.input:
            name, _, rng = text.partition("=")
            inputs.append((name.strip(), parse_range(rng)))
        return ops.neuron_sweep(design, inputs, args.preactivation, args.tolerance or 1.0, args.method, _sets(args),
                                args.max_minutes, render)
    if c == "fit":
        if design.space != "neuron":
            raise ApiError("mns fit is for neuron designs (after mns sweep).")
        return ops.neuron_fit(design, args.tolerance or 1.0, args.method)
    if c == "dpsweep":
        if design.space != "activation":
            raise ApiError(f"mns dpsweep is for activation designs; {design.id} is a {design.space} design.")
        return ops.activation_study(design, parse_range(args.dp) if args.dp else None, _sets(args), args.max_minutes,
                                    render)
    if c == "characterise":
        if design.space != "full":
            raise ApiError(f"mns characterise is for full-neuron designs; {design.id} is a {design.space} design.")
        axes = []
        for text in args.axis or design.spec().get("characterise", {}).get("axes", []):
            name, _, rng = text.partition("=")
            part, _, field = name.strip().partition(".")
            axes.append((part, field or "pressure", parse_range(rng)))
        if not axes:
            raise ApiError("Give at least one --axis Part.field=from:to:n", HELP["characterise"])
        record = [k.strip() for k in args.record.split(",")] if args.record else None
        return ops.full_run(design, axes, record, _sets(args), args.max_minutes, render)
    raise ApiError(f"Unknown command {c!r}", "mns help")


def design_command(args, study):
    from .study import derive
    from .util import ApiError, parse_assignment
    _need(args, 1, HELP["design"])
    sub = args.args[0]
    if sub == "list":
        return {"designs": [d.describe() | {"metrics": None} for d in study.designs()]}
    if sub == "show":
        _need(args, 2, HELP["design"])
        d = study.design(args.args[1])
        out = d.describe()
        state = d.state()
        out["runs"] = state.get("runs", {})
        out["warnings"] = (state.get("geometry") or {}).get("warnings", [])
        out["parameters"] = d.spec().get("parameters", {})
        if args.spec:
            out["design_yaml"] = d.spec_path.read_text(encoding="utf-8")
        out["files"] = sorted(str(p.relative_to(d.folder)) for p in d.folder.rglob("*") if p.is_file())
        return out
    if sub == "new":
        _need(args, 2, HELP["design"])
        from .templates import template_space, template_spec
        space = args.space or template_space(args.template) or "neuron"
        spec = template_spec(args.template, space)
        design = study.new_design(space, args.args[1], spec, why=args.why)
        return {"id": design.id, "folder": str(design.folder), "design_file": str(design.spec_path),
                "next": f"Edit {design.spec_path.name}, then mns cad build {design.id}"}
    if sub == "derive":
        _need(args, 3, HELP["design"])
        parent = study.design(args.args[1])
        sets = [parse_assignment(s) for s in args.set]
        design = derive(study, parent, args.args[2], sets, why=args.why)
        out = {"id": design.id, "folder": str(design.folder), "parent": parent.id, "changed": dict(sets)}
        if not args.no_build:
            from .model import build_design
            out["build"] = build_design(design, render=not args.no_render)
        return out
    if sub == "import":
        _need(args, 2, HELP["design"])
        from .model import import_file
        return {"imported": import_file(study, args.args[1], args.name, args.why)}
    raise ApiError(f"design {sub}?", HELP["design"])


def cad_command(args, study):
    from .model import build_design, load_model
    from .util import ApiError
    _need(args, 2, HELP["cad"])
    sub, design = args.args[0], study.design(args.args[1])
    if sub == "build":
        return build_design(design, render=not args.no_render)
    if design.space == "full":
        raise ApiError("A full neuron has no CAD of its own: inspect or render its neuron and activation designs.")
    if sub == "inspect":
        from .cadspec import geometry_report
        cad, project = load_model(design)
        report = geometry_report(cad, {p.name: p.role for p in project.parts})
        return {"design": design.id, **report}
    if sub == "render":
        from app.builder import mesh_sizes
        from .render import parse_views, render_model
        cad, project = load_model(design)
        surfaces = cad.mesh(mesh_sizes(cad, project))
        name = args.name or ("model_" + (args.views or "default").replace(",", "_").replace(":", "-")
                             + ("_exploded" if args.exploded else ""))
        path = design.path("renders", Path(name).stem + ".png")
        only = [s.strip() for s in args.only.split(",")] if args.only else None
        render_model(surfaces, project.parts, path, parse_views(args.views or "iso,front,top,right"), args.exploded,
                     only, title=f"{design.id} {design.name}")
        study.events.emit("file", f"render {path.name}", design=design.id, files=[str(path)])
        return {"design": design.id, "image": str(path)}
    raise ApiError(f"cad {sub}?", HELP["cad"])


def status(study):
    from . import jobs
    from .events import read_events
    designs = [{k: d.describe()[k] for k in ("id", "name", "space", "status", "parent")} for d in study.designs()]
    events, _ = read_events(study.root)
    recent = [{k: e.get(k) for k in ("t", "kind", "design", "text") if e.get(k)} for e in events[-8:]]
    return {"study": study.info.get("name"), "root": str(study.root), "designs": designs,
            "running_jobs": jobs.running(study), "recent_events": recent}


if __name__ == "__main__":
    sys.exit(main())
