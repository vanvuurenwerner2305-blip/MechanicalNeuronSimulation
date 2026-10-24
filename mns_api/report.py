"""
LaTeX reports. The facts are generated, the story is written by the agent:

  designs/<ID>/report/design.tex      main file (generated; do not edit)
                      narrative.tex   the agent's text, created once as a skeleton with \\input{auto_design} and
                                      \\input{auto_results} where the generated parts go; never overwritten
                      auto_design.tex, auto_results.tex   generated from the design file and the runs (overwritten)
  report/study.tex                    main file of the study report (generated)
         narrative.tex                the agent's text (skeleton created once)
         auto_overview.tex            table of all designs and their metrics (generated)
         auto_designs.tex             one short section per design with its main figure (generated)
         figures/                     comparison plots (mns compare --x --y)

`--build` compiles with pdflatex (twice) and returns the PDF and the first LaTeX errors.
"""
import os
import re
import subprocess
from pathlib import Path

from .util import ApiError, read_json, rounded

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\setlength{\parskip}{0.4em}
\setlength{\parindent}{0pt}
"""


def tex(text):
    """Text safe for LaTeX."""
    text = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                 ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}"),
                 ("Δ", r"$\Delta$"), ("³", r"$^3$"), ("²", r"$^2$"), ("↔", r"$\leftrightarrow$"),
                 ("μ", r"$\mu$"), ("ṁ", r"$\dot m$")):
        text = text.replace(a, b)
    return text


def number(x):
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        v = rounded(float(x), 4)
        return "--" if v is None else f"{v:g}"
    if isinstance(x, list):
        return " .. ".join(number(v) for v in x[:2]) if len(x) == 2 else ", ".join(number(v) for v in x)
    return tex(x if x is not None else "--")


def table(rows, header, caption="", align=None):
    align = align or "l" * len(header)
    lines = [r"\begin{longtable}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule",
             r"\endhead"]
    lines += [" & ".join(cells) + r" \\" for cells in rows]
    lines += [r"\bottomrule"]
    if caption:
        lines += [r"\caption{" + caption + "}"]
    lines += [r"\end{longtable}"]
    return "\n".join(lines) + "\n"


def figure(path, caption, width=None, base=None):
    if width is None:  # by the picture's width: small single plots are not blown up to the full line
        try:
            from PIL import Image
            with Image.open(Path(base or ".") / path) as im:
                width = min(0.95, max(0.45, im.size[0] / 1150))
        except Exception:
            width = 0.8
    return (r"\begin{figure}[htbp]\centering" + "\n" + rf"\includegraphics[width={width}\linewidth]{{{path}}}" + "\n"
            + r"\caption{" + caption + "}\n" + r"\end{figure}" + "\n")


def _rel(path, start):
    """path relative to the folder the .tex is compiled in, with forward slashes (spaces stay out of it: the
    study's own folder names have none)."""
    return os.path.relpath(Path(path).resolve(), Path(start).resolve()).replace("\\", "/")


def _exists(design, *parts):
    p = design.folder.joinpath(*parts)
    return p if p.exists() else None


# -----------------------------
# One design
# -----------------------------

def auto_design(design, base):
    """Parameters, bodies and the model picture. base: the folder the .tex is compiled in."""
    spec, state = design.spec(), design.state()
    out = []
    if design.space == "full":
        geo = state.get("geometry") or {}
        out.append("Full neuron: neuron design \\textbf{" + tex(spec.get("neuron")) + "} with activation design "
                   "\\textbf{" + tex(spec.get("activation")) + "} in place of its membrane \\textbf{"
                   + tex((spec.get("link") or {}).get("part") or geo.get("link", {}).get("part")) + "}.\n\n")
        if spec.get("set"):
            out.append(table([[tex(k), number(v)] for k, v in spec["set"].items()], ["Chamber value", "Set to"]))
        return "".join(out)
    params = spec.get("parameters") or {}
    if params:
        values = design.parameters()
        out.append(table([[tex(k), tex(v), number(values[k])] for k, v in params.items()],
                         ["Parameter", "Expression", "Value"], "Design parameters (lengths in mm, moduli in MPa, "
                                                                "pressures in kPa)."))
    geo = state.get("geometry") or {}
    if geo.get("bodies"):
        rows = []
        for b in geo["bodies"]:
            extra = b.get("model") or (f"t = {number(b['thickness_mm'])} mm" if b.get("thickness_mm") else "")
            if b.get("pressure_kPa") not in (None, 0, 0.0):
                extra += f", {number(b['pressure_kPa'])} kPa"
            rows.append([tex(b["name"]), tex(b["role"]), number(b["volume_mm3"]),
                         " x ".join(number(s) for s in b["size_mm"]), tex(extra)])
        out.append(table(rows, ["Body", "Role", r"Volume [mm$^3$]", "Size [mm]", "Notes"], "Bodies of the design."))
    if geo.get("warnings"):
        out.append("\\textbf{Model warnings:}\n\\begin{itemize}\n" + "".join(r"\item " + tex(w) + "\n"
                                                                            for w in geo["warnings"])
                   + "\\end{itemize}\n")
    render = _exists(design, "renders", "model.png")
    if render:
        out.append(figure(_rel(render, base), f"{tex(design.id)}: the bodies coloured by role (fluids see-through).", base=base))
    return "".join(out)


def auto_results(design, base):
    out = []
    runs = design.state().get("runs", {})
    results = design.folder / "results"
    if design.space == "neuron":
        if (results / "solve.json").exists():
            s = read_json(results / "solve.json")
            out.append(r"\subsection*{Static solve}" + "\n")
            out.append(("Converged" if s["converged"] else r"\textbf{Not converged}") + f" ({tex(s['message'])})"
                       + (f", with {tex(', '.join(f'{k} = {v}' for k, v in s['set'].items()))}" if s.get("set") else "")
                       + ".\n\n")
            out.append(table([[tex(n), number(c["P_kPa"]), number(c["dV_mm3"])] for n, c in s["chambers"].items()],
                             ["Chamber", "P [kPa]", r"$\Delta V$ [mm$^3$]"], "Chamber pressures and volume changes."))
            out.append(table([[tex(n), number(v["max_displacement_mm"]), number(v["max_area_stretch"])]
                              for n, v in s["sheets"].items()],
                             ["Membrane", "max. displacement [mm]", "max. area stretch"], "Membrane deformation."))
            if _exists(design, "renders", "solve.png"):
                out.append(figure(_rel(results.parent / "renders" / "solve.png", base),
                                  f"{tex(design.id)}: deformed membranes, coloured by displacement.", base=base))
        if (results / "sweep_rows.json").exists() and "sweep" in runs:
            r = runs["sweep"]
            out.append(r"\subsection*{Sweep}" + "\n")
            out.append(f"{r['points']} points ({r['converged']} converged) over "
                       + tex(", ".join(f"{k} {v[0]:g}..{v[1]:g} kPa ({v[2]} values)" for k, v in r["inputs"].items()))
                       + ".\n\n")
            if (results / "sweep_response.png").exists():
                out.append(figure(_rel(results / "sweep_response.png", base), "Response to the inputs.", base=base))
    elif design.space == "activation":
        if "study" in runs:
            m = design.state().get("metrics", {})
            out.append(r"\subsection*{Activation function}" + "\n")
            rows = [[tex(k), number(v)] for k, v in m.items() if not k.startswith(("solve:", "full:"))]
            out.append(table(rows, ["Quantity", "Value"], "Key numbers of the activation-function study."))
            for name, cap in (("activation.png", "Tube area, mass flow, output pressure and swept volume against the "
                                                 "pressure difference across the membrane."),
                              ("profiles.png", "Cross-section along the tube at several pressure differences.")):
                if (results / name).exists():
                    out.append(figure(_rel(results / name, base), cap, base=base))
            if _exists(design, "renders", "study.png"):
                out.append(figure(_rel(design.folder / "renders" / "study.png", base),
                                  "The device at the largest pressure difference, coloured by displacement.", base=base))
    else:
        for name, title in (("solve.json", "Solve"), ("characterisation.json", "Characterisation")):
            if (results / name).exists():
                s = read_json(results / name)["summary"]
                out.append(r"\subsection*{" + title + "}\n")
                if s.get("values"):
                    out.append(table([[tex(k), number(v)] for k, v in s["values"].items()], ["Quantity", "Value"]))
                if s.get("ranges"):
                    out.append(table([[tex(k), number(v)] for k, v in s["ranges"].items()],
                                     ["Quantity", "Range over the grid"],
                                     tex(", ".join(f"{k}: {v[0]:g}..{v[1]:g} ({v[2]})" for k, v in s["axes"].items()))))
                if s.get("warning"):
                    out.append(r"\textbf{Warning:} " + tex(s["warning"]) + "\n\n")
        if (results / "characterisation.png").exists():
            out.append(figure(_rel(results / "characterisation.png", base), "The characterisation.", base=base))
    return "".join(out) or "No simulation results yet.\n"


NARRATIVE = r"""% Written by the agent. Replace every comment with prose (keep the \input lines where the
% generated tables and figures should appear). Numbers come from the generated parts: refer to them, do not retype.
\section{Aim}
% Why this design was made: the question it answers and what it changes from its parent.

\section{Design}
\input{auto_design}
% The geometry and the choices behind it.

\section{Results}
\input{auto_results}
% What the results show.

\section{Findings}
% Conclusions, comparison with other designs, what to try next.
"""


def report_design(design, build=False):
    folder = design.path("report")
    (folder / "auto_design.tex").write_text(auto_design(design, folder), encoding="utf-8")
    (folder / "auto_results.tex").write_text(auto_results(design, folder), encoding="utf-8")
    narrative = folder / "narrative.tex"
    if not narrative.exists():
        narrative.write_text(NARRATIVE, encoding="utf-8")
    spec = design.spec()
    title = f"{design.id}: {design.name}"
    main = (PREAMBLE + r"\title{" + tex(title) + "}\n" + r"\author{Agentic research study, " + tex(design.study.info.get("name"))
            + "}\n" + r"\date{\today}" + "\n" + r"\begin{document}" + "\n" + r"\maketitle" + "\n"
            + (r"\noindent\textit{" + tex(spec.get("why")) + "}" + "\n\n" if spec.get("why") else "")
            + (f"Derived from {tex(spec['parent'])}.\n\n" if spec.get("parent") else "")
            + r"\input{narrative}" + "\n" + r"\end{document}" + "\n")
    (folder / "design.tex").write_text(main, encoding="utf-8")
    out = {"design": design.id, "folder": str(folder), "narrative": str(narrative),
           "unwritten_sections": _unwritten(narrative)}
    if build:
        out.update(compile_pdf(folder, "design.tex"))
        if out.get("pdf"):
            design.set_state(status="reported" if design.state().get("status") != "created" else "created")
            design.study.events.emit("file", f"report {design.id}", design=design.id, files=[out["pdf"]])
    return out


def _unwritten(narrative):
    """Sections of the narrative that still hold only the skeleton's comments."""
    text = narrative.read_text(encoding="utf-8")
    out = []
    for m in re.finditer(r"\\section\*?\{([^}]*)\}(.*?)(?=\\section|\\appendix|\Z)", text, re.S):
        lines = [line.strip() for line in m.group(2).splitlines() if line.strip()]
        prose = [line for line in lines if not line.startswith("%") and "\\input{" not in line]
        asked = any(line.startswith("%") for line in lines)  # the skeleton asks for text here
        if asked and not prose:
            out.append(m.group(1))
    return out


# -----------------------------
# The study
# -----------------------------

STUDY_NARRATIVE = r"""% Written by the agent. Replace every comment with prose; keep the \input lines.
\section{Introduction}
% The brief: what the study set out to find, and the starting design(s).

\section{Method}
% How the design space was explored: which parameters, ranges, in what order, and why.

\section{Designs}
\input{auto_overview}
% Walk through the designs and what each one tested.

\section{Findings}
% The main results, with comparison figures (report/figures, from mns compare --x --y):
% \begin{figure}[htbp]\centering\includegraphics[width=0.8\linewidth]{figures/<file>.png}\caption{...}\end{figure}

\section{Conclusions and recommendations}
% Answer the brief. What design to use, what is still uncertain, what to study next.

\appendix
\section{Design summaries}
\input{auto_designs}
"""


def report_study(study, build=False):
    from .compare import design_row
    folder = study.root / "report"
    folder.mkdir(exist_ok=True)
    designs = study.designs()
    rows = []
    for d in designs:
        r = design_row(d)
        metrics = {k[2:]: v for k, v in r.items() if k.startswith("m:")}
        key = ", ".join(f"{tex(k)} = {number(v)}" for k, v in list(metrics.items())[:4])
        rows.append([tex(d.id), tex(d.name), tex(d.space), tex(r.get("parent") or "--"), tex(r.get("status")),
                     key or "--"])
    overview = table(rows, ["ID", "Name", "Space", "Parent", "Status", "Key metrics"],
                     "All designs of the study.", "lllllp{6.5cm}")
    (folder / "auto_overview.tex").write_text(overview, encoding="utf-8")
    parts = []
    for d in designs:
        spec = d.spec()
        parts.append(r"\subsection{" + tex(f"{d.id}: {d.name}") + "}\n")
        if spec.get("why"):
            parts.append(r"\textit{" + tex(spec["why"]) + "}\n\n")
        pdf = d.folder / "report" / "design.pdf"
        if pdf.exists():
            parts.append("Full report: \\texttt{" + tex(_rel(pdf, folder)) + "}.\n\n")
        for sub, name in (("results", "sweep_response.png"), ("results", "activation.png"),
                          ("results", "characterisation.png"), ("renders", "solve.png"), ("renders", "model.png")):
            p = d.folder / sub / name
            if p.exists():
                parts.append(figure(_rel(p, folder), tex(f"{d.id}: {name[:-4].replace('_', ' ')}"), 0.75))
                break
    (folder / "auto_designs.tex").write_text("".join(parts) or "No designs.\n", encoding="utf-8")
    narrative = folder / "narrative.tex"
    if not narrative.exists():
        narrative.write_text(STUDY_NARRATIVE, encoding="utf-8")
    brief = study.root / "brief.md"
    main = (PREAMBLE + r"\title{" + tex(study.info.get("name", "Agentic research study")) + "}\n"
            + r"\author{Agentic research study with the Membrane Neuron Simulator}" + "\n" + r"\date{\today}" + "\n"
            + r"\begin{document}" + "\n" + r"\maketitle" + "\n" + r"\tableofcontents" + "\n" + r"\input{narrative}"
            + "\n" + r"\end{document}" + "\n")
    (folder / "study.tex").write_text(main, encoding="utf-8")
    figures = sorted(str(p.relative_to(folder).as_posix()) for p in (folder / "figures").glob("*.png")) \
        if (folder / "figures").exists() else []
    out = {"folder": str(folder), "narrative": str(narrative), "unwritten_sections": _unwritten(narrative),
           "comparison_figures": figures, "brief": str(brief) if brief.exists() else None}
    if build:
        out.update(compile_pdf(folder, "study.tex"))
        if out.get("pdf"):
            study.events.emit("file", "study report", files=[out["pdf"]])
    return out


def compile_pdf(folder, main):
    folder = Path(folder)
    errors = []
    for _ in range(2):
        try:
            proc = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", main], cwd=str(folder),
                                  capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=240)
        except FileNotFoundError:
            raise ApiError("pdflatex not found", "Install MiKTeX (or TeX Live) and make sure pdflatex is on the PATH.")
        except subprocess.TimeoutExpired:
            raise ApiError("pdflatex took too long (over 4 minutes)", "Look at the .log file in the report folder.")
        if proc.returncode != 0:
            log = (folder / Path(main).with_suffix(".log").name)
            text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else proc.stdout
            lines = text.splitlines()
            for k, line in enumerate(lines):
                if line.startswith("!"):
                    errors.append(" ".join(l.strip() for l in lines[k:k + 3]))
            break
    pdf = folder / Path(main).with_suffix(".pdf").name
    if errors or not pdf.exists():
        return {"pdf": None, "latex_errors": errors[:5] or ["pdflatex failed; see the .log file"]}
    return {"pdf": str(pdf)}
