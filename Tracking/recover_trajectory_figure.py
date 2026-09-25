#!/usr/bin/env python3
"""Recover Figure 13's embedded draw.io sources and typeset readable layouts.

This preserves the published raster plots exactly. It does not infer 3D samples
from pixels or regenerate measured trajectories. See figures/trajectory_comparison.
Requires numpy, matplotlib, and Pillow; run with --help for options.
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import urllib.parse
import xml.etree.ElementTree as ET
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image


ROWS = (
    ("Monocular pose fusion", "3 cameras"),
    ("Monocular pose fusion", "5 cameras"),
    ("Triangulation", "3 cameras"),
    ("Triangulation", "5 cameras"),
)
COLS = ("Basic", "Kalman", "Adaptive Kalman")


def recover(source, output):
    """Decode draw.io metadata and copy embedded image bytes without re-encoding."""
    with Image.open(source) as image:
        metadata = image.info.get("mxfile")
    if not metadata:
        raise ValueError("Source PNG has no embedded mxfile/draw.io document.")
    xml = urllib.parse.unquote(metadata) if metadata.startswith("%") else metadata
    document = ET.fromstring(xml)
    diagrams = document.findall("diagram")
    if len(diagrams) != 1:
        raise ValueError("Expected Figure 13's single draw.io page.")
    diagram = diagrams[0]
    model = diagram.find("mxGraphModel")
    if model is None:
        model = ET.fromstring(urllib.parse.unquote(
            zlib.decompress(base64.b64decode(diagram.text), -15).decode()))
    cells = {cell.get("id"): cell for cell in model.iter("mxCell")}

    def position(cell):
        geometry = cell.find("mxGeometry")
        x = float(geometry.get("x", 0)) if geometry is not None else 0
        y = float(geometry.get("y", 0)) if geometry is not None else 0
        parent = cells.get(cell.get("parent"))
        if parent is not None:
            px, py = position(parent)
            x, y = x + px, y + py
        return x, y

    panels = []
    for cell in cells.values():
        style = cell.get("style", "")
        if "image=data:image/png," not in style:
            continue
        encoded = style.split("image=data:image/png,", 1)[1].split(";", 1)[0]
        data = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(data)) as image:
            if image.size != (1200, 857):
                raise ValueError("Unexpected panel size; inspect the source layout first.")
        x, y = position(cell)
        panels.append((y, x, cell.get("id"), data))
    panels.sort(key=lambda panel: panel[:2])
    if len(panels) != 12 or len({p[0] for p in panels}) != 4:
        raise ValueError("Expected Figure 13's four rows of three embedded plots.")

    recovered = output / "recovered"
    recovered.mkdir(parents=True, exist_ok=True)
    (recovered / "original.drawio").write_text(xml, encoding="utf-8")
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "description": "Published raster plots; 3D sample coordinates are not included.",
        "panels": [],
    }
    for i, (y, x, cell_id, data) in enumerate(panels):
        row, col = divmod(i, 3)
        filename = f"panel_{i + 1:02d}.png"
        (recovered / filename).write_bytes(data)
        manifest["panels"].append({
            "file": filename, "cell_id": cell_id,
            "row": row, "column": col, "original_position": [x, y],
            "method": ROWS[row][0], "cameras": ROWS[row][1],
            "filter": COLS[col], "sha256": hashlib.sha256(data).hexdigest(),
        })
    (recovered / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return recovered


def add_panel(fig, path, rectangle):
    """Place a plot using a viewport that excludes only the old title and legend.

    The retained pixel rectangle contains the complete plotting box, curves,
    tick labels, and axis labels. Originals on disk remain byte-for-byte intact.
    """
    ax = fig.add_axes(rectangle)
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"))
    ax.imshow(pixels, interpolation="none")
    ax.set_xlim(115, 1150)
    ax.set_ylim(825, 70)
    ax.set_axis_off()
    return ax


def legend(fig, y, fontsize=9):
    handles = [
        Line2D([0], [0], color="#E74C3C", lw=1.8, label="Ground truth (GT)"),
        Line2D([0], [0], color="#3498DB", lw=1.2, ls="--", marker="o",
               markersize=3, label="Measured trajectory"),
    ]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, y),
               ncol=2, frameon=False, fontsize=fontsize, handlelength=2.5)


def overview(recovered, rows=(0, 1, 2, 3)):
    """A paper-width figure with row labels, column labels, and a shared legend."""
    width, height = 7.2, 0.8 + len(rows) * 1.96
    fig = plt.figure(figsize=(width, height), facecolor="white")
    legend(fig, 1 - 0.19 / height)
    left, right, gap = 0.38, 0.08, 0.08
    panel_width = (width - left - right - 2 * gap) / 3
    for col, heading in enumerate(COLS):
        x = left + col * (panel_width + gap) + panel_width / 2
        fig.text(x / width, 1 - 0.48 / height, heading,
                 ha="center", va="center", fontsize=10, weight="bold")
    for index, row in enumerate(rows):
        top = 0.74 + index * 1.96
        method, cameras = ROWS[row]
        fig.text(left / width, 1 - top / height,
                 f"{method}  |  {cameras}", fontsize=9,
                 weight="bold", va="center", color="#17212B")
        for col in range(3):
            x = left + col * (panel_width + gap)
            rectangle = [x / width, 1 - (top + 1.78) / height,
                         panel_width / width, 1.63 / height]
            add_panel(fig, recovered / f"panel_{row * 3 + col + 1:02d}.png", rectangle)
            fig.text((x + 0.02) / width, 1 - (top + 0.20) / height,
                     f"({chr(97 + row * 3 + col)})", fontsize=8, va="top")
        if index < len(rows) - 1:
            y = 1 - (top + 1.87) / height
            fig.add_artist(Line2D([left / width, 1 - right / width], [y, y],
                                 transform=fig.transFigure, color="#D8DDE3", lw=0.5))
    return fig


def save_figure(fig, output, stem):
    # Preserve the physical page size: do not use bbox_inches='tight'.
    fig.savefig(output / f"{stem}.pdf", metadata={
        "Title": "LARK Figure 13: recovered trajectory comparison",
        "Subject": "New typesetting; original raster trajectory panels preserved",
    })
    fig.savefig(output / f"{stem}.svg")
    fig.savefig(output / f"{stem}.png", dpi=240)
    plt.close(fig)


def build(recovered, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "pdf.fonttype": 42})
    save_figure(overview(recovered), output, "trajectory_comparison_readable")
    save_figure(overview(recovered, (0, 1)), output, "monocular_comparison")
    save_figure(overview(recovered, (2, 3)), output, "triangulation_comparison")
    with PdfPages(output / "trajectory_panels_large.pdf") as pdf:
        for i in range(12):
            row, col = divmod(i, 3)
            fig = plt.figure(figsize=(9, 7), facecolor="white")
            fig.text(0.5, 0.96,
                     f"({chr(97+i)}) {ROWS[row][0]} · {ROWS[row][1]} · {COLS[col]}",
                     ha="center", fontsize=14, weight="bold")
            legend(fig, 0.905, fontsize=12)
            add_panel(fig, recovered / f"panel_{i + 1:02d}.png", [0.015, 0.025, 0.97, 0.83])
            pdf.savefig(fig)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original 3dplots_v2.png with draw.io metadata")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "figures/trajectory_comparison")
    args = parser.parse_args()
    try:
        recovered = recover(args.source, args.output)
        build(recovered, args.output)
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(f"Recovered 12 original panels and draw.io source; figures saved to {args.output}")


if __name__ == "__main__":
    main()
