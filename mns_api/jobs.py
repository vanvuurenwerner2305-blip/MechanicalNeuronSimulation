"""
Background jobs: a long command started with --background runs as its own process; `mns wait <job>` blocks until
it is done (or a time limit passes) and prints the command's output. Each job keeps
<study>/.mns/jobs/<job>/status.json (running / done / failed / cancelled), output.json and log.txt.
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .util import ApiError, read_json, write_json

CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def jobs_dir(study):
    folder = study.root / ".mns" / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def start(study, argv, design=None, why=""):
    job = datetime.now().strftime("J%m%d-%H%M%S")
    folder = jobs_dir(study) / job
    n = 1
    while folder.exists():
        n += 1
        folder = jobs_dir(study) / f"{job}-{n}"
    job = folder.name
    folder.mkdir()
    env = dict(os.environ, MNS_JOB=job, MNS_STUDY=str(study.root), PYTHONIOENCODING="utf-8")
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", "mns_api.cli", *argv, "--job-output", str(folder / "output.json")]
    log = open(folder / "log.txt", "wb")
    flags = CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    try:
        proc = subprocess.Popen(cmd, cwd=str(study.root), env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                creationflags=flags | (CREATE_BREAKAWAY_FROM_JOB if os.name == "nt" else 0))
    except OSError:  # the caller's job object does not allow breaking away
        proc = subprocess.Popen(cmd, cwd=str(study.root), env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                                creationflags=flags)
    write_json(folder / "status.json", {"job": job, "state": "running", "pid": proc.pid, "design": design,
                                        "command": "mns " + " ".join(argv), "why": why,
                                        "started": datetime.now().isoformat(timespec="seconds")})
    return job


def status(study, job):
    folder = jobs_dir(study) / job
    if not (folder / "status.json").exists():
        known = [p.name for p in sorted(jobs_dir(study).iterdir())][-10:]
        raise ApiError(f"No job {job!r}", f"Recent jobs: {', '.join(known) or 'none'}")
    st = read_json(folder / "status.json")
    if st["state"] == "running" and not _alive(st.get("pid")):
        if (folder / "output.json").exists():
            st["state"] = "done"
        else:
            st["state"] = "failed"
            st["error"] = "the job's process ended without output; see log.txt"
            st["log_tail"] = _tail(folder / "log.txt")
        write_json(folder / "status.json", st)
    progress = study.root / ".mns" / "progress" / f"{job}.json"
    if st["state"] == "running" and progress.exists():
        try:
            st["progress"] = json.loads(progress.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return st


def finish(study, job, state, output=None):
    folder = jobs_dir(study) / job
    st = read_json(folder / "status.json") if (folder / "status.json").exists() else {"job": job}
    st.update(state=state, ended=datetime.now().isoformat(timespec="seconds"))
    write_json(folder / "status.json", st)


def wait(study, job, timeout=540.0, poll=2.0):
    """The job's output once it is done, or its status and progress when `timeout` s pass first."""
    end = time.time() + float(timeout)
    while True:
        st = status(study, job)
        if st["state"] != "running":
            out = jobs_dir(study) / job / "output.json"
            result = read_json(out) if out.exists() else {}
            return {"job": job, "state": st["state"], **({"output": result} if result else {}),
                    **({k: st[k] for k in ("error", "log_tail") if k in st})}
        if time.time() >= end:
            p = st.get("progress", {})
            return {"job": job, "state": "running", "progress": p.get("text"),
                    "fraction": round(p.get("fraction", -1), 3) if p else None,
                    "hint": f"Still running: call mns wait {job} again."}
        time.sleep(poll)


def cancel(study, job):
    st = status(study, job)
    if st["state"] != "running":
        return {"job": job, "state": st["state"], "note": "not running"}
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(st["pid"]), "/T", "/F"], capture_output=True)
    else:
        os.kill(st["pid"], 9)
    finish(study, job, "cancelled")
    study.events.emit("error", f"job {job} cancelled", design=st.get("design"))
    return {"job": job, "state": "cancelled"}


def listing(study, limit=10):
    out = []
    for folder in sorted(jobs_dir(study).iterdir())[-limit:]:
        try:
            st = status(study, folder.name)
        except ApiError:
            continue
        item = {k: st.get(k) for k in ("job", "state", "design", "command", "started", "ended")}
        if st.get("progress"):
            item["progress"] = st["progress"].get("text")
        out.append(item)
    return out


def running(study):
    return [j for j in listing(study, 50) if j["state"] == "running"]


def _alive(pid):
    if not pid:
        return False
    try:
        import psutil
        return psutil.pid_exists(int(pid)) and psutil.Process(int(pid)).status() != "zombie"
    except Exception:
        return True


def _tail(path, lines=15):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(t for t in text[-lines:] if "TripleDES" not in t and "algorithms." not in t)
    except OSError:
        return ""
