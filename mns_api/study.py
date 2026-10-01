"""
Study folders and designs.

    <study>/
      study.json            marks the folder as a study (name, created, simulator)
      brief.md              what to find out (written by the user)
      designs/<ID>_<name>/  one folder per design, ID = N### (neuron: inputs -> pre-activation),
                            A### (activation function: pre-activation -> activation), F### (full neuron)
        design.yaml         the design file (parameters, CAD, roles and properties, settings, why)
        state.json          status, checks, run summaries and key metrics (written by the API)
        model.step          the CAD built from design.yaml
        model.mns|.mad|.mfn the simulator project (opens in the GUI)
        results/ renders/ report/
      report/               the study report
      .mns/                 events (followed by the GUI), jobs, live shapes
"""
import copy
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import yaml

from .events import EventLog
from .util import ApiError, as_number, evaluate_parameters, read_json, write_json

STUDY_FILE = "study.json"
SPACES = {"neuron": "N", "activation": "A", "full": "F"}
SPACE_NAMES = {"N": "neuron", "A": "activation", "F": "full"}
PROJECT_FILE = {"neuron": "model.mns", "activation": "model.mad", "full": "model.mfn"}


def now():
    return datetime.now().isoformat(timespec="seconds")


def slug(name):
    s = re.sub(r"[^A-Za-z0-9_-]+", "-", str(name).strip()).strip("-")
    return s[:40] or "design"


class Study:
    def __init__(self, root):
        self.root = Path(root).resolve()
        if not (self.root / STUDY_FILE).exists():
            raise ApiError(f"{self.root} is not a study folder (no {STUDY_FILE}).",
                           "Run mns from inside the study folder.")
        self.info = read_json(self.root / STUDY_FILE)
        self.designs_dir = self.root / "designs"
        self.designs_dir.mkdir(exist_ok=True)
        self.events = EventLog(self.root, job=os.environ.get("MNS_JOB"))

    @classmethod
    def find(cls, start=None):
        env = os.environ.get("MNS_STUDY")
        if env:
            return cls(env)
        here = Path(start or Path.cwd()).resolve()
        for folder in [here, *here.parents]:
            if (folder / STUDY_FILE).exists():
                return cls(folder)
        raise ApiError(f"No study folder found at or above {here}.",
                       "Run mns inside a study folder (it has a study.json).")

    @staticmethod
    def create(root, name=None, simulator=None):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        if (root / STUDY_FILE).exists():
            raise ApiError(f"{root} is already a study.")
        for sub in ("designs", "report", "inputs", ".mns"):
            (root / sub).mkdir(exist_ok=True)
        write_json(root / STUDY_FILE, {"type": "mns_study", "version": 1, "name": name or root.name,
                                       "created": now(), "simulator": str(simulator or Path(__file__).parents[1])})
        return Study(root)

    # -----------------------------
    # Designs
    # -----------------------------

    def designs(self):
        out = []
        for folder in sorted(self.designs_dir.iterdir()):
            if folder.is_dir() and (folder / "design.yaml").exists() and re.match(r"^[NAF]\d{3}", folder.name):
                out.append(Design(self, folder))
        return out

    def design(self, ref) -> "Design":
        ref = str(ref).strip()
        designs = self.designs()
        for d in designs:
            if ref in (d.id, d.folder.name, d.name):
                return d
        for d in designs:
            if ref.upper() == d.id:
                return d
        known = ", ".join(f"{d.id} ({d.name})" for d in designs) or "none yet"
        raise ApiError(f"No design {ref!r}.", f"Designs: {known}. See `mns design list`.")

    def next_id(self, space):
        prefix = SPACES[space]
        numbers = [int(d.id[1:]) for d in self.designs() if d.id.startswith(prefix)]
        return f"{prefix}{(max(numbers) + 1) if numbers else 1:03d}"

    def new_design(self, space, name, spec, why="", parent=None, design_id=None):
        if space not in SPACES:
            raise ApiError(f"Unknown space {space!r}", "Spaces: neuron, activation, full.")
        design_id = design_id or self.next_id(space)
        folder = self.designs_dir / f"{design_id}_{slug(name)}"
        if folder.exists():
            raise ApiError(f"{folder.name} already exists.")
        folder.mkdir(parents=True)
        spec = dict(spec)
        head = {"id": design_id, "name": name, "space": space, "why": why or spec.get("why", ""),
                "parent": parent, "created": now()}
        for k in head:
            spec.pop(k, None)
        design = Design(self, folder)
        design.save_spec({**head, **spec})
        design.set_state(status="created", created=now())
        self.events.emit("design", f"{design_id} {name} created" + (f" from {parent}" if parent else ""),
                         design=design_id, why=why or None, space=space)
        return design


class Design:
    def __init__(self, study, folder):
        self.study = study
        self.folder = Path(folder)
        self.id = self.folder.name[:4]
        self.space = SPACE_NAMES[self.id[0]]
        self._spec = None

    @property
    def name(self):
        return self.folder.name[5:]

    @property
    def spec_path(self):
        return self.folder / "design.yaml"

    @property
    def step_path(self):
        return self.folder / "model.step"

    @property
    def project_path(self):
        return self.folder / PROJECT_FILE[self.space]

    def path(self, sub, name=None):
        folder = self.folder / sub
        folder.mkdir(exist_ok=True)
        return folder / name if name else folder

    def spec(self, reload=False):
        if self._spec is None or reload:
            try:
                self._spec = yaml.safe_load(self.spec_path.read_text(encoding="utf-8")) or {}
            except yaml.YAMLError as exc:
                raise ApiError(f"{self.id}: design.yaml is not valid YAML: {exc}",
                               "Check the indentation and quotes around expressions with ':' or '#'.")
        return self._spec

    def save_spec(self, spec):
        text = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=110, default_flow_style=None)
        self.spec_path.write_text(text, encoding="utf-8")
        self._spec = spec

    def parameters(self):
        return evaluate_parameters(self.spec().get("parameters") or {})

    # -----------------------------
    # State (written by the API only)
    # -----------------------------

    def state(self):
        p = self.folder / "state.json"
        return read_json(p) if p.exists() else {}

    def set_state(self, **values):
        state = self.state()
        state.update(values)
        state["updated"] = now()
        write_json(self.folder / "state.json", state)
        return state

    def record_run(self, kind, summary, metrics=None):
        state = self.state()
        runs = state.setdefault("runs", {})
        runs[kind] = dict(summary, at=now())
        if metrics:
            state.setdefault("metrics", {}).update(metrics)
        state["updated"] = now()
        write_json(self.folder / "state.json", state)

    def describe(self):
        spec, state = self.spec(), self.state()
        return {"id": self.id, "name": self.name, "space": self.space, "status": state.get("status", "created"),
                "parent": spec.get("parent"), "why": spec.get("why", ""), "folder": str(self.folder),
                "metrics": state.get("metrics", {})}


# -----------------------------
# Copies of designs
# -----------------------------

def derive(study, parent, name, sets=(), why=""):
    """A new design from `parent` with parameters (name=value) or body properties (Body.prop=value) changed."""
    spec = copy.deepcopy(parent.spec())
    params = spec.setdefault("parameters", {}) or {}
    spec["parameters"] = params
    for key, value in sets:
        if key.startswith(("solver.", "study.")):
            continue
        if "." in key:
            body_name, prop = key.split(".", 1)
            if parent.space == "full":
                spec.setdefault("set", {})[key] = value
                continue
            body = next((b for b in spec.get("bodies", []) if b.get("name") == body_name), None)
            if body is None:
                raise ApiError(f"{parent.id} has no body {body_name!r}",
                               "Bodies: " + ", ".join(b.get("name", "?") for b in spec.get("bodies", [])))
            if prop == "role":
                body["role"] = value
            else:
                body.setdefault("props", {})[prop] = value
        elif key in ("solver", "study"):
            raise ApiError(f"Set solver/study settings as {key}.<field>=value")
        else:
            if key not in params:
                raise ApiError(f"{parent.id} has no parameter {key!r}",
                               f"Parameters: {', '.join(params) or 'none'}. Use Body.prop=value for body properties, "
                               "solver.<field>=value or study.<field>=value for settings.")
            params[key] = value
    for key, value in sets:
        if key.startswith(("solver.", "study.")):
            group, field = key.split(".", 1)
            spec.setdefault(group, {})[field] = value
    evaluate_parameters(params)  # fails early on a bad expression
    design = study.new_design(parent.space, name, spec, why=why, parent=parent.id)
    for extra in ("base.step",):  # files the design file refers to
        if (parent.folder / extra).exists():
            shutil.copy2(parent.folder / extra, design.folder / extra)
    return design


def spec_number(value, names, what):
    return as_number(value, names, what)


def dump_yaml(data):
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=110, default_flow_style=None)


def load_yaml_text(text):
    return yaml.safe_load(text)


def json_or_yaml(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
