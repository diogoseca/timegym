"""Shared plotted coordinates for SVG, PNG, and inline TikZ figures.

Coordinates are typographic big points, with the origin at the top left.
No raster images are embedded. TikZ is self-contained for the document editor.
"""
from __future__ import annotations

from itertools import groupby
from pathlib import Path
import os
import re
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "timegym-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from figure_common import INK, GRAY, BLUE


def tex_text(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
                    "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
                    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
                    "−": "-", "–": "--", "—": "---", "×": r"\ensuremath{\times}",
                    "ρ": r"\ensuremath{\rho}", "Δ": r"\ensuremath{\Delta}",
                    "≠": r"\ensuremath{\ne}", "≤": r"\ensuremath{\leq}",
                    "≥": r"\ensuremath{\geq}", "→": r"\ensuremath{\rightarrow}",
                    "·": r"\ensuremath{\cdot}", "✓": r"\ensuremath{\checkmark}",
                    "✗": r"\ensuremath{\times}"}
    superscripts = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    pieces = re.split(r"([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", str(value))
    return "".join(r"\textsuperscript{" + piece.translate(superscripts) + "}"
                   if re.fullmatch(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+", piece)
                   else "".join(replacements.get(char, char) for char in piece)
                   for piece in pieces)


class Canvas:
    def __init__(self, width=448, height=320):
        self.width, self.height = float(width), float(height)
        self.items = []

    def text(self, x, y, text, size=7, color=INK, weight="normal", align="left"):
        lines = str(text).split("\n")
        for index, value in enumerate(lines):
            self.items.append(dict(kind="text", x=float(x), y=float(y)+(index-(len(lines)-1)/2)*size*1.2,
                                   text=value, size=size, color=color, weight=weight, align=align))

    def line(self, x1, y1, x2, y2, color=GRAY, width=.6, dash=False):
        self.polyline([(x1, y1), (x2, y2)], color=color, width=width, dash=dash)

    def polyline(self, points, color=GRAY, width=.6, dash=False):
        self.items.append(dict(kind="path", points=[(float(x), float(y)) for x, y in points],
                               color=color, width=width, dash=dash))

    def circle(self, x, y, r=.8, color=GRAY):
        self.items.append(dict(kind="circle", x=float(x), y=float(y), r=r, color=color))

    def cross(self, x, y, r=1.4, color=BLUE, width=.65):
        self.line(x-r, y-r, x+r, y+r, color=color, width=width)
        self.line(x-r, y+r, x+r, y-r, color=color, width=width)

    def rect(self, x, y, w, h, fill="#ffffff", stroke=None, width=.5):
        self.items.append(dict(kind="rect", x=x, y=y, w=w, h=h, fill=fill, stroke=stroke, width=width))

    def _groups(self):
        def key(pair):
            index, item = pair
            return ("circle", item["color"], item["r"]) if item["kind"] == "circle" else (index,)
        for _, group in groupby(enumerate(self.items), key=key):
            yield [item for _, item in group]

    def render(self, directory: Path, name: str):
        directory.mkdir(parents=True, exist_ok=True)
        with plt.rc_context({"svg.fonttype": "none", "svg.hashsalt": "timegym-paper-v2",
                             "font.family": "DejaVu Sans", "lines.scale_dashes": False}):
            fig = plt.figure(figsize=(self.width/72, self.height/72), facecolor="white")
            ax = fig.add_axes([0, 0, 1, 1]); ax.set(xlim=(0, self.width), ylim=(self.height, 0))
            ax.axis("off")
            for layer, group in enumerate(self._groups(), 1):
                obj = group[0]
                kind = obj["kind"]
                if kind == "text":
                    ax.text(obj["x"], obj["y"], obj["text"], fontsize=obj["size"], color=obj["color"],
                            ha=obj["align"], va="center", fontweight=obj["weight"], zorder=layer)
                elif kind == "path":
                    xs, ys = zip(*obj["points"])
                    ax.plot(xs, ys, color=obj["color"], linewidth=obj["width"],
                            linestyle=(0, (2, 2)) if obj["dash"] else "-", solid_capstyle="round", zorder=layer)
                elif kind == "circle":
                    ax.scatter([item["x"] for item in group], [item["y"] for item in group],
                               s=(2*obj["r"])**2, c=obj["color"], marker="o", linewidths=0,
                               edgecolors="none", zorder=layer)
                elif kind == "rect":
                    ax.add_patch(Rectangle((obj["x"], obj["y"]), obj["w"], obj["h"],
                                           facecolor=obj["fill"] or "none", edgecolor=obj["stroke"] or "none",
                                           linewidth=obj["width"], zorder=layer))
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for label in ax.texts:
                bounds = label.get_window_extent(renderer=renderer)
                if bounds.x0 < -1 or bounds.y0 < -1 or bounds.x1 > fig.bbox.width+1 or bounds.y1 > fig.bbox.height+1:
                    raise ValueError(f"Figure {name} clips text: {label.get_text()!r}")
            svg_path = directory / f"{name}.svg"
            fig.savefig(svg_path, metadata={"Date": None})
            svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())+"\n",
                                encoding="utf-8")
            fig.savefig(directory / f"{name}.png", dpi=220)
            plt.close(fig)
        (directory / f"{name}.tikz").write_text(self.tikz(), encoding="utf-8")

    def tikz(self):
        colors = set()
        for obj in self.items:
            for key in ("color", "fill", "stroke"):
                if obj.get(key): colors.add(obj[key])
        def c(value):
            if not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
                raise ValueError(f"Expected a six-digit hex color: {value}")
            return "tg" + value[1:]
        lines = ["% Generated vector scene; SVG and PNG use the same plotted geometry.", r"\begingroup"]
        lines += [f"\\definecolor{{{c(value)}}}{{HTML}}{{{value[1:]}}}" for value in sorted(colors)]
        lines += [r"\resizebox{\linewidth}{!}{%", r"\begin{tikzpicture}[x=1bp,y=-1bp]",
                  f"\\path[use as bounding box] (0,0) rectangle ({self.width:g},{self.height:g});"]
        for group in self._groups():
            obj = group[0]
            kind = obj["kind"]
            if kind == "text":
                anchor = {"left": "west", "right": "east", "center": "center"}[obj["align"]]
                bold = r"\bfseries" if obj["weight"] == "bold" else ""
                font = rf"\sffamily\fontsize{{{obj['size']:g}bp}}{{{obj['size']*1.2:g}bp}}\selectfont{bold}"
                lines.append(f"\\node[anchor={anchor},inner sep=0pt,outer sep=0pt,text={c(obj['color'])},font={{{font}}}] "
                             f"at ({obj['x']:.3f},{obj['y']:.3f}) {{{tex_text(obj['text'])}}};")
            elif kind == "path":
                points = " -- ".join(f"({x:.3f},{y:.3f})" for x, y in obj["points"])
                dash = ",dash pattern=on 2bp off 2bp" if obj["dash"] else ""
                lines.append(f"\\draw[color={c(obj['color'])},line width={obj['width']:g}bp,line cap=round{dash}] {points};")
            elif kind == "circle":
                circles = " ".join(f"({item['x']:.3f},{item['y']:.3f}) circle[radius={item['r']:g}bp]" for item in group)
                lines.append(f"\\path[fill={c(obj['color'])},draw=none] {circles};")
            elif kind == "rect":
                options = [f"fill={c(obj['fill'])}" if obj["fill"] else "fill=none",
                           f"draw={c(obj['stroke'])}" if obj["stroke"] else "draw=none",
                           f"line width={obj['width']:g}bp"]
                lines.append(f"\\path[{','.join(options)}] ({obj['x']:.3f},{obj['y']:.3f}) rectangle "
                             f"({obj['x']+obj['w']:.3f},{obj['y']+obj['h']:.3f});")
        lines += [r"\end{tikzpicture}%", "}", r"\endgroup", ""]
        return "\n".join(lines)
