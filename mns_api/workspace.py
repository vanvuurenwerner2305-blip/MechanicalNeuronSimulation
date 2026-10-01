"""
A new study folder, ready for the design agent:

  CLAUDE.md, MANUAL.md        the agent's rules and its user manual (from agent/ in the simulator)
  background/                 the research article behind the study, read once for perspective
  .claude/settings.json       what the agent may run and edit (mns, pdflatex; no scripts, no simulator source)
  brief.md                    the user's brief (copied, or a template to fill in)
  feature_requests.md         where the agent writes what the software can not do yet
  start_agent.cmd             opens Claude Code in the folder with `mns` on the PATH
  .mns/bin/mns, mns.cmd       the command line, bound to this Python and this simulator
  inputs/                     copies of the starting files; designs/ the imported starting designs
"""
import json
import os
import shutil
import sys
from pathlib import Path

from .study import Study
from .util import ApiError

SIMULATOR = Path(__file__).resolve().parents[1]
AGENT = SIMULATOR / "agent"

FIRST_PROMPT = ("Start the design study: read CLAUDE.md, background/README.md and background/article.tex, MANUAL.md "
                "and brief.md, check the starting designs with mns status, then plan and begin.")


def write_launchers(root):
    """mns (Git Bash) and mns.cmd (cmd / PowerShell) in <study>/.mns/bin, and start_agent.cmd."""
    root = Path(root)
    bin_dir = root / ".mns" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":  # launched from the GUI
        python = python.with_name("python.exe")
    (bin_dir / "mns.cmd").write_text(
        "@echo off\r\n"
        f'set "PYTHONPATH={SIMULATOR};%PYTHONPATH%"\r\n'
        "set PYTHONIOENCODING=utf-8\r\n"
        f'"{python}" -m mns_api.cli %*\r\n', encoding="utf-8")
    posix = lambda p: "/" + str(p).replace(":", "", 1).replace("\\", "/") if ":" in str(p) else str(p)  # noqa: E731
    sh = bin_dir / "mns"
    sh.write_bytes((
        "#!/bin/sh\n"
        f'export PYTHONPATH="{SIMULATOR}"\n'
        "export PYTHONIOENCODING=utf-8\n"
        f'exec "{posix(python)}" -m mns_api.cli "$@"\n').encode("utf-8"))
    try:
        sh.chmod(0o755)
    except OSError:
        pass
    prompt = FIRST_PROMPT.replace('"', "'")
    # Claude Code's own installer puts claude.exe in ~/.local/bin, which is often only on Git Bash's PATH: look for it
    # in the usual places as well as on the PATH
    found = shutil.which("claude")
    candidates = ([found] if found else []) + [
        str(Path.home() / ".local" / "bin" / "claude.exe"),
        str(Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "npm" / "claude.cmd")]
    lines = ["@echo off",
             "rem Opens the design agent (Claude Code) in this study folder, with the mns command on the PATH.",
             'cd /d "%~dp0"',
             'set "PATH=%~dp0.mns\\bin;%USERPROFILE%\\.local\\bin;%APPDATA%\\npm;%PATH%"',
             "set MNS_STUDY=%~dp0.",
             'set "CLAUDE="']
    lines += [f'if not defined CLAUDE if exist "{c}" set "CLAUDE={c}"' for c in dict.fromkeys(candidates)]
    lines += ['if not defined CLAUDE for /f "delims=" %%c in (\'where claude 2^>nul\') do if not defined CLAUDE '
              'set "CLAUDE=%%c"',
              "if not defined CLAUDE (echo Claude Code was not found: install it from https://claude.com/claude-code "
              "& pause & exit /b 1)",
              f'"%CLAUDE%" "{prompt}"']
    (root / "start_agent.cmd").write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    return bin_dir


def write_agent_files(root):
    root = Path(root)
    for name in ("CLAUDE.md", "MANUAL.md"):
        src = AGENT / name
        if src.exists():
            shutil.copy2(src, root / name)
    background = AGENT / "background"
    if background.is_dir():  # the research the study belongs to (the article), read once before starting
        shutil.copytree(background, root / "background", dirs_exist_ok=True)
    settings = AGENT / "settings.json"
    if settings.exists():
        (root / ".claude").mkdir(exist_ok=True)
        text = settings.read_text(encoding="utf-8").replace("{SIMULATOR}", str(SIMULATOR).replace("\\", "/"))
        json.loads(text)  # must stay valid JSON
        (root / ".claude" / "settings.json").write_text(text, encoding="utf-8")
    requests = root / "feature_requests.md"
    if not requests.exists():
        requests.write_text("# Feature requests\n\nWhat the software could not do during this study (written by the "
                            "agent; one entry per need: what, why, the workaround used).\n", encoding="utf-8")


def create_study(folder, brief=None, sources=(), name=None):
    folder = Path(folder).resolve()
    study = Study.create(folder, name=name)
    write_agent_files(folder)
    write_launchers(folder)
    if brief:
        brief_path = Path(brief)
        if not brief_path.exists():
            raise ApiError(f"Brief not found: {brief}")
        shutil.copy2(brief_path, folder / "brief.md")
    elif not (folder / "brief.md").exists():
        template = AGENT / "brief_template.md"
        (folder / "brief.md").write_text(template.read_text(encoding="utf-8") if template.exists() else
                                         "# Brief\n\nGoal:\n\nFree parameters and ranges:\n\nBudget:\n",
                                         encoding="utf-8")
    imported, errors = [], []
    if sources:
        from .model import import_file
        for src in sources:
            try:
                result = import_file(study, src, why="starting design from the user")
                imported += result if isinstance(result, list) else [result]
            except Exception as exc:  # the study stays; the file can be imported again with mns design import
                errors.append(f"{Path(src).name}: {exc}")
    study.events.emit("note", f"study {study.info['name']} created", files=[str(folder)])
    return {"study": str(folder), "imported": imported, **({"import_errors": errors} if errors else {}),
            "start_agent": str(folder / "start_agent.cmd")}


def refresh_agent_files(folder):
    """Update CLAUDE.md, MANUAL.md, settings and launchers of an existing study (after a software update)."""
    write_agent_files(folder)
    write_launchers(folder)
