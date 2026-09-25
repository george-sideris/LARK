#!/usr/bin/env python3
"""Render Figure 13 with a shaded STL and the published trajectory pixel layers.

The measured XYZ samples are unavailable. This is a presentation reconstruction:
only the head's display projection is fitted to the known GT and published red
pixels. Curve locations are never fitted, resampled into XYZ, or interpolated.
The STL retains the legacy visualization transform (not anatomical registration).
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib import font_manager
from mpl_toolkits.mplot3d import proj3d  # Registers the 3D projection on older MPL.
import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import trimesh

from recover_trajectory_figure import ROWS, COLS

GT_COLOR = "#BE3025"
MEASURED_COLOR = "#0067AC"
HEAD_VIEWS = {
    "full": dict(bottom=820, row_height=2.10, panel_height=1.76,
                 panel_bottom=1.92),
    "half": dict(bottom=750, row_height=1.95, panel_height=1.61,
                 panel_bottom=1.77),
}


def configure_fonts(style):
    """Match the Computer Modern-style MathJax labels in the draw.io figures."""
    family = "DejaVu Sans"
    if style == "mathlike":
        for name, weight, slant in (
            ("cmunrm.otf", 400, "normal"), ("cmunbx.otf", 700, "normal"),
            ("cmunti.otf", 400, "italic"), ("cmunbi.otf", 700, "italic"),
        ):
            try:
                path = subprocess.check_output(["kpsewhich", name], text=True).strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ValueError("Mathlike text requires the CM Unicode fonts from TeX Live.") from exc
            if not Path(path).is_file():
                raise ValueError(f"Font not found: {name}")
            # Explicit weights also handle Matplotlib 3.1 misidentifying cmunbx.
            font_manager.fontManager.ttflist.append(font_manager.FontEntry(
                fname=path, name="CMU Serif", style=slant, weight=weight,
            ))
        family = "CMU Serif"
    plt.rcParams.update({"font.family": family, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "mathtext.fontset": "cm"})
    return family


class Scene:
    def __init__(self, root, head_view="full"):
        self.head_view = head_view
        # Read the saved literal coordinates without executing the legacy script.
        tree = ast.parse((root / "Tracking/data_processing.py").read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "landmark_points" for t in node.targets
            ):
                points = ast.literal_eval(node.value)
                if "R1" in points:
                    break
        else:
            raise ValueError("Head landmark coordinate dictionary not found.")
        landmarks = np.array(list(points.values()))
        self.mesh_path = root / "IBIS/GS Head Landmark Shell v2.stl"
        mesh = trimesh.load(self.mesh_path)
        vertices = mesh.vertices.copy()
        scale = (np.ptp(landmarks, axis=0) / np.ptp(vertices, axis=0)).max() * 1.1
        self.vertices = (vertices - vertices.mean(0)) * scale + landmarks.mean(0)
        self.gt = np.concatenate([
            np.load(root / f"Landmarks/Ground Truth/{name}_trajectory.npy")
            for name in ("back", "front")
        ])
        self.offset = -np.concatenate([landmarks, self.gt]).min(0) + 10
        self.limits = np.concatenate([landmarks, self.gt, self.vertices]).max(0) + self.offset + 10
        self.source_face_count = len(mesh.faces)
        # The optional half-head view restores only the earlier lower cut (Z).
        # Never clip at the Y=0 plane: that plane passes through the nose tip.
        if head_view == "half":
            centers = (self.vertices[mesh.faces] + self.offset).mean(1)
            self.faces = mesh.faces[centers[:, 2] >= 0]
        else:
            self.faces = mesh.faces
        triangles = self.vertices[self.faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.array([-0.3, -0.6, 1.0])
        light /= np.linalg.norm(light)
        intensity = 0.66 + 0.24 * np.abs(normals @ light)
        self.colors = np.column_stack([
            intensity * 0.95, intensity * 0.975, intensity, np.ones(len(normals))
        ])
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(projection="3d")
        self.axes.view_init(elev=26, azim=-56)
        self.axes.set_xlim(0, self.limits[0])
        self.axes.set_zlim(0, self.limits[2])

    def project(self, points, parameters):
        ymax, sx, sy, tx, ty = parameters
        self.axes.set_ylim(0, ymax)
        homogeneous = np.column_stack([points + self.offset, np.ones(len(points))])
        projected = homogeneous @ self.axes.get_proj().T
        projected = projected[:, :3] / projected[:, 3, None]
        projected[:, :2] = projected[:, :2] * [sx, -sy] + [tx, ty]
        return projected

    def fit_view(self, pixels):
        a = pixels.astype(int)
        red = (a[:, :, 0] - a[:, :, 1] > 40) & (a[:, :, 0] - a[:, :, 2] > 40)
        red[:75] = False  # Exclude the old legend.
        y, x = np.where(red)
        points = np.column_stack([x, y])[::3]
        if len(points) < 200:
            raise ValueError("Insufficient published GT pixels to match the head view.")

        def residual(parameters):
            q = self.project(self.gt, parameters)[:, :2]
            _, nearest = cKDTree(q).query(points)
            return (q[nearest] - points).ravel()

        candidates = [least_squares(
            residual, [ymax, 6347, 4502, 615, 416], diff_step=1e-5,
            max_nfev=250, bounds=([215, 6100, 4250, 595, 400],
                                 [300, 6650, 4750, 635, 432])
        ) for ymax in (231.94, 267.52)]
        best = min(candidates, key=lambda fit: np.mean(fit.fun ** 2))
        distances = cKDTree(self.project(self.gt, best.x)[:, :2]).query(points)[0]
        if np.percentile(distances, 95) > 3:
            raise ValueError("Head display alignment exceeds the 3-pixel acceptance limit.")
        return best.x, {
            "parameters_ymax_sx_sy_tx_ty": best.x.tolist(),
            "reference_pixels_sampled": len(points),
            "median_reference_distance_px": float(np.median(distances)),
            "p95_reference_distance_px": float(np.percentile(distances, 95)),
        }


def curve_layer(pixels, bottom=820):
    """Change only hue/opacity of existing colored pixels, without spatial filters."""
    values = pixels.astype(float) / 255
    chroma = np.ptp(values, axis=2)
    gt = (values[:, :, 0] > values[:, :, 1]) & (values[:, :, 0] > values[:, :, 2])
    measured = (values[:, :, 2] > values[:, :, 0]) & (values[:, :, 1] > values[:, :, 0])
    support = (chroma > 0) & (gt | measured)
    support[:75] = False
    layer = np.zeros((*pixels.shape[:2], 4))
    layer[gt, :3] = matplotlib.colors.to_rgb(GT_COLOR)
    layer[measured, :3] = matplotlib.colors.to_rgb(MEASURED_COLOR)
    layer[:, :, 3] = np.where(support, np.clip(chroma / 0.45, 0, 1), 0)
    assert np.array_equal(layer[:, :, 3] > 0, support)
    # The crop used in the clean figure must not remove any colored curve pixel.
    outside = support.copy()
    outside[180:bottom + 1, 270:1061] = False
    if outside.any():
        raise ValueError("Clean panel viewport would clip a trajectory.")
    return layer, int(support.sum())


def render_panel(scene, pixels, parameters, output, number):
    bottom = HEAD_VIEWS[scene.head_view]["bottom"]
    projected = scene.project(scene.vertices, parameters)
    # Every selected face must fit; the half-head cut is intentional geometry,
    # not an accidental crop of the image or of the intact nose.
    rendered_vertices = projected[np.unique(scene.faces), :2]
    lower = rendered_vertices.min(0)
    upper = rendered_vertices.max(0)
    if np.any(lower < [270, 180]) or np.any(upper > [1060, bottom]):
        raise ValueError("Clean panel viewport would clip the head mesh.")
    triangles = projected[scene.faces]
    order = np.argsort(triangles[:, :, 2].mean(1))[::-1]
    layer, count = curve_layer(pixels, bottom=bottom)
    fig = plt.figure(figsize=(12, 8.57), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    # Preserve source axes for a separate inspection version.
    original = ax.imshow(pixels, interpolation="none", zorder=0)
    ax.add_collection(PolyCollection(
        triangles[order, :, :2], facecolors=scene.colors[order],
        edgecolors="none", antialiased=False, zorder=1,
    ))
    ax.imshow(layer, interpolation="none", zorder=2)
    ax.set_xlim(-0.5, 1199.5)
    ax.set_ylim(856.5, -0.5)
    ax.axis("off")
    fig.savefig(output / f"panel_{number:02d}_axes.png", dpi=100)
    original.set_visible(False)
    fig.savefig(output / f"panel_{number:02d}.png", dpi=100)
    plt.close(fig)
    return count


def add_panel(fig, path, rectangle, axes=False, head_view="full"):
    ax = fig.add_axes(rectangle)
    with Image.open(path) as im:
        ax.imshow(np.asarray(im.convert("RGB")), interpolation="none")
    if axes:
        ax.set_xlim(115, 1150)
        ax.set_ylim(825, 70)
    else:
        ax.set_xlim(270, 1060)
        ax.set_ylim(HEAD_VIEWS[head_view]["bottom"], 180)
    ax.axis("off")
    return ax


def shared_orientation(fig):
    """One direction-only coordinate triad for all panels, in the lower right."""
    ax = fig.add_axes([0.905, 0.012, 0.08, 0.075])
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 90)
    ax.set_aspect("equal")
    origin = np.array([15, 26])
    for label, direction in (("X", [40, -11]), ("Y", [31, 18]), ("Z", [0, 41])):
        endpoint = origin + direction
        ax.annotate("", xy=endpoint, xytext=origin,
                    arrowprops=dict(arrowstyle="->", color="#414A53", lw=0.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=7))
        ax.text(*(origin + np.array(direction) * 1.35), f"${label}$",
                fontsize=9, ha="center", va="center", color="#303840")
    ax.axis("off")


def legend(fig, y, fontsize=9):
    handles = [Line2D([0], [0], color=GT_COLOR, lw=1.8, label="Ground truth (GT)"),
               Line2D([0], [0], color=MEASURED_COLOR, lw=1.4, ls="--", marker="o",
                      markersize=3, label="Measured trajectory")]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, y), ncol=2,
               frameon=False, fontsize=fontsize, handlelength=2.4)


def overview(panels, rows=(0, 1, 2, 3), head_view="full"):
    view = HEAD_VIEWS[head_view]
    width, height = 7.2, 0.74 + len(rows) * view["row_height"]
    fig = plt.figure(figsize=(width, height), facecolor="white")
    legend(fig, 1 - 0.18 / height, fontsize=9.5)
    left, right, gap = 0.15, 0.10, 0.08
    panel_width = (width - left - right - gap * 2) / 3
    for col, label in enumerate(COLS):
        x = left + col * (panel_width + gap) + panel_width / 2
        fig.text(x / width, 1 - 0.48 / height, label, ha="center", va="center",
                 fontsize=11, weight="bold")
    for index, row in enumerate(rows):
        top = 0.77 + index * view["row_height"]
        fig.text(left / width, 1 - top / height,
                 f"{ROWS[row][0]}  |  {ROWS[row][1]}",
                 fontsize=10, weight="bold", va="center", color="#17212B")
        for col in range(3):
            x = left + col * (panel_width + gap)
            add_panel(fig, panels / f"panel_{row * 3 + col + 1:02d}.png",
                      [x / width, 1 - (top + view["panel_bottom"]) / height,
                       panel_width / width, view["panel_height"] / height], head_view=head_view)
            fig.text((x + 0.02) / width, 1 - (top + 0.16) / height,
                     f"({chr(97 + row * 3 + col)})", fontsize=8, va="top")
        if index < len(rows) - 1:
            # Leave clear space above the following row's subtitle.
            y = 1 - (top + view["row_height"] - 0.16) / height
            fig.add_artist(Line2D([left / width, 1 - right / width], [y, y],
                                 transform=fig.transFigure, color="#D8DDE3", lw=0.5))
    shared_orientation(fig)
    return fig


def export(fig, output, name):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300)
    plt.close(fig)


def build(panels, recovered, output, head_view="full", font_style="mathlike"):
    family = configure_fonts(font_style)
    export(overview(panels, head_view=head_view), output, "trajectory_comparison_shaded")
    export(overview(panels, (0, 1), head_view=head_view), output, "monocular_shaded")
    export(overview(panels, (2, 3), head_view=head_view), output, "triangulation_shaded")
    with PdfPages(output / "trajectory_panels_shaded_large.pdf") as pdf:
        for i in range(12):
            row, col = divmod(i, 3)
            fig = plt.figure(figsize=(9, 7), facecolor="white")
            fig.text(0.5, 0.96,
                     f"({chr(97 + i)}) {ROWS[row][0]} · {ROWS[row][1]} · {COLS[col]}",
                     ha="center", fontsize=14, weight="bold")
            legend(fig, 0.905, 12)
            add_panel(fig, panels / f"panel_{i + 1:02d}_axes.png",
                      [0.015, 0.025, 0.97, 0.83], axes=True)
            pdf.savefig(fig)
            plt.close(fig)
    fig = plt.figure(figsize=(12, 5.4), facecolor="white")
    for index, (label, path) in enumerate((
        ("Original wireframe", recovered / "panel_01.png"),
        ("Shaded head + stronger curves", panels / "panel_01_axes.png"),
    )):
        fig.text(0.25 + 0.5 * index, 0.965, label, ha="center", fontsize=14, weight="bold")
        add_panel(fig, path, [0.005 + 0.5 * index, 0.015, 0.49, 0.9], axes=True)
    export(fig, output, "before_after")
    (output / "layout_manifest.json").write_text(json.dumps({
        "head_view": head_view, "font_family": family, "font_style": font_style,
        "shared_orientation_frames_per_overview": 1,
        "orientation_frame_location": "lower right; direction only, not a metric scale",
        "panel_directory": str(panels.resolve()),
        "panel_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(panels.glob("panel_??.png"))},
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font-style", choices=("mathlike", "sans"), default="mathlike")
    parser.add_argument("--panels-dir", type=Path,
                        help="Existing rendered panels; requires --reuse-panels")
    parser.add_argument("--head-view", choices=HEAD_VIEWS, default="full",
                        help="Full STL or the earlier lower-head cut with an intact nose")
    parser.add_argument("--reuse-panels", action="store_true", help="Retypeset existing shaded panel renders")
    args = parser.parse_args()
    if args.panels_dir and not args.reuse_panels:
        parser.error("--panels-dir requires --reuse-panels")
    root = Path(__file__).resolve().parents[1]
    base = root / "figures/trajectory_comparison"
    output = args.output or base / ("shaded_half_head" if args.head_view == "half" else "shaded")
    output.mkdir(parents=True, exist_ok=True)
    panels = args.panels_dir or output / "panels"
    if args.reuse_panels:
        if not panels.is_dir():
            parser.error(f"Rendered panel directory not found: {panels}")
    else:
        panels.mkdir(parents=True, exist_ok=True)
    recovered = base / "recovered"
    if not args.reuse_panels:
        scene = Scene(root, head_view=args.head_view)
        report = {
            "method": "STL display fit to known GT; published colored curve pixel positions retained",
            "mesh_sha256": hashlib.sha256(scene.mesh_path.read_bytes()).hexdigest(),
            "mesh_faces_rendered": len(scene.faces),
            "mesh_source_faces": scene.source_face_count,
            "head_view": args.head_view,
            "mesh_clipping": ("Lower cut at Z=0 only; no X/Y clipping; nose intact"
                              if args.head_view == "half" else "None; all source STL faces retained"),
            "head_transform": "Legacy landmark-bounds fit; contextual display, not anatomical registration",
            "orientation_triads": "Direction only, not a metric scale",
            "panels": [],
        }
        for i in range(1, 13):
            path = recovered / f"panel_{i:02d}.png"
            with Image.open(path) as im:
                pixels = np.asarray(im.convert("RGB"))
            parameters, fit = scene.fit_view(pixels)
            count = render_panel(scene, pixels, parameters, panels, i)
            fit.update(panel=i, source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       preserved_colored_pixel_positions=count)
            report["panels"].append(fit)
            print(f"Panel {i:02d}: median GT alignment {fit['median_reference_distance_px']:.2f} px; "
                  f"retained {count} curve pixel positions", flush=True)
        plt.close(scene.figure)
        (output / "render_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    build(panels, recovered, output, head_view=args.head_view, font_style=args.font_style)
    print(f"Saved shaded figures to {output}")


if __name__ == "__main__":
    main()
