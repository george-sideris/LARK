# LARK Figure 13: trajectory comparison recovery

The 12-panel GT-versus-measured figure is Figure 13 in both the local
`LARK_MedIA_Submission.pdf` (PDF page 16) and `LARK_TRO_Submission (1).pdf`
(PDF page 14). The original manuscript included `Definitions/3dplots_v2.png` under
the label `fig:trajectory_comparison_left_occlusion`.

The repository includes the recovered source panels and draw.io layout, cropped
shaded panels with their rendering manifest, and the final publication figure
in PDF, PNG, and SVG formats. Other layouts described below are generated outputs;
local manuscript previews and temporary rendering files are excluded from Git.

## Manuscript typography revision

`publication/trajectory_comparison_shaded.pdf` keeps the cropped head and intact
nose, uses Computer Modern-style CMU Serif lettering to match the draw.io MathJax
labels, and has one shared XYZ orientation frame in the lower-right corner.
Its PDF embeds regular/bold CMU Serif and Computer Modern math italic fonts.
The shaded panel images are reused unchanged from `shaded_half_head/panels`.

Rebuild this layout without rerendering geometry:

```bash
python Tracking/shade_trajectory_figure.py --head-view half --reuse-panels \
  --panels-dir figures/trajectory_comparison/shaded_half_head/panels \
  --output figures/trajectory_comparison/publication
```

Mathlike lettering is now the default (`--font-style sans` selects the earlier
sans-serif family). TeX Live's CM Unicode fonts must be available to `kpsewhich`.
The manuscript checkout `/home/george/LARK-manuscript` also contains an independent
layout script and the twelve shaded inputs under
`figure_sources/trajectory_comparison/`, and includes the figure through
`Definitions/trajectory_comparison.pdf`.

## Shaded head version

The `shaded/` directory contains the subsequent visibility revision:

- `trajectory_comparison_shaded.pdf` (also SVG/PNG): a shaded head surface,
  darker red/blue curves, tighter framing, shared headings and legend, and
  small X/Y/Z orientation arrows instead of repeated grid boxes.
- `trajectory_panels_shaded_large.pdf`: twelve enlarged panels, retaining the
  original numeric axes and grid for inspection.
- `before_after.pdf` (also SVG/PNG): the first panel before and after shading.
- `monocular_shaded.pdf` and `triangulation_shaded.pdf`: separate six-panel
  layouts, also in SVG/PNG.
- `render_manifest.json`: mesh and source-panel hashes, display-fit residuals,
  and the number of original colored pixel positions retained per panel.

Regenerate with:

```bash
python Tracking/shade_trajectory_figure.py
```

This script needs NumPy, Matplotlib, Pillow, SciPy, and trimesh. To change only
the page layout after the panels have been rendered, use `--reuse-panels`.

The surface is rendered from the actual `IBIS/GS Head Landmark Shell v2.stl`,
using the legacy landmark-bounds visualization transform. The original camera
elevation/azimuth are retained. Five display parameters (vertical world extent,
horizontal/vertical image scale, and image translation) are fitted to the known
GT curves and visible red reference pixels in each source panel. The fitted
display is accepted only if 95% of the sampled reference pixels are within three
pixels of the projected GT. **This is a display alignment, not recovered camera
calibration or an anatomical surface registration.** The shaded head provides
spatial context; distances between the curves and its surface should not be
interpreted quantitatively. The residuals are in image pixels, not TRE units.

The renderer retains all 449,448 STL faces, including the nose, chin, and base.
An earlier shaded draft incorrectly discarded faces extending beyond the old
plot's zero-coordinate planes, which cut the nose and lower head. That clipping
has been removed. The full projected mesh is checked against the expanded panel
viewport before export.

An additional `shaded_half_head/` version restores the earlier lower-head cut and
compact layout while preserving the nose tip. It applies only the old lower
`Z=0` display cut; the erroneous `Y=0` cut through the nose stays disabled. Its
file names match those in `shaded/`, including `trajectory_comparison_shaded.pdf`
and `trajectory_panels_shaded_large.pdf`. The full-head version remains in
`shaded/`. Generate the additional version with:

```bash
python Tracking/shade_trajectory_figure.py --head-view half
```

Both published trajectory layers retain their original colored pixel positions.
Only hue and opacity are changed to increase contrast; there is no curve
interpolation, smoothing, thickening, or reconstruction of missing measured XYZ
samples. The compact figure omits numeric axes and uses direction-only X/Y/Z
arrows; those arrows are not scale bars. The enlarged PDF retains the original
numeric axes. The original files and the first formatting revision remain intact.

## Ready-to-review files

- `trajectory_comparison_readable.pdf`: all 12 published plots, with shared
  column headings, readable method/camera row headings, panel letters, and
  one legend. Page width is 7.2 inches; PDF/SVG headings are vector text.
- `trajectory_comparison_readable.svg` and `.png`: editable text layout and preview.
- `monocular_comparison.pdf` and `triangulation_comparison.pdf`: separate
  six-panel figures, also exported to SVG and PNG.
- `trajectory_panels_large.pdf`: one plot per page (12 pages) for reading the
  original axes and examining trajectory differences.
- `recovered/original.drawio`: the complete original layout, recovered from
  PNG metadata. Open it in diagrams.net/draw.io.
- `recovered/panel_01.png` through `panel_12.png`: original embedded plot images,
  copied byte-for-byte without recompression.
- `recovered/manifest.json`: original layout positions, row/column assignments,
  source path, and SHA-256 checksums.

The original PNG is a draw.io export with an embedded `mxfile` document. That
document holds twelve 1200-by-857-pixel plot images. Its headings, legends, axes,
and curves are already rasterized inside each plot: draw.io contains the assembly,
not the Matplotlib figure objects or underlying 3D coordinates.

The new layout preserves the original plotted data. A viewport excludes the old
per-panel title/legend and blank margins; it retains the full plot box, trajectory
curves, axes, and ticks. New headings and a shared legend are typeset separately.
The axis text inside each panel remains raster text at the original resolution;
it is still small in the all-panel paper layout. Use the enlarged PDF to inspect
it. Increasing its font size properly requires regenerating the plots from data.

## Repeat the recovery and typesetting

From the LARK repository root, with NumPy, Matplotlib, and Pillow installed:

```bash
python Tracking/recover_trajectory_figure.py \
  /home/george/Downloads/LARK_TRO_Submission__Copy_/Definitions/3dplots_v2.png
```

Use `--output /path/to/output` to write elsewhere. The script validates the
expected panel count and dimensions, sorts images by their absolute draw.io
positions, and creates all the files listed above. It does not modify the source
manuscript or original PNG.

## Plotting code that survived

The GitHub checkout at `/home/george/LARK` contains:

- `Tracking/data_processing.py:513`: `plot_3d_trajectories`, which produces
  the individual GT/measurement plots, including the head mesh and the view
  angle `elev=26, azim=-56`.
- `Tracking/data_processing.py:918`: `results_table`, which registers measured
  fiducials and applies that transformation to the measured trajectories.
- `Tracking/data_processing.py:1440`: the call to `plot_3d_trajectories` and
  export of each `<key>_trajectory_plot.png`.
- `Tracking/util/landmark_registration.py`: the rigid registration algorithm.
- `Landmarks/Ground Truth/{back,front}_trajectory.npy`: reference curves.
- `IBIS/GS Head Landmark Shell v2.stl`: the background head model.

The plotting file was added in commit `05102e3`; no separate 12-panel assembly
script was found in the seven local GitHub commits. The Codeberg checkout and
the supplementary software archive also retain the plotting file. The Codeberg
copy is byte-identical to the GitHub copy.

The original script is not currently runnable as configured: `base_folder`
points at `/home/george/MultiCameraTracking/Registration/Landmark_Registration_Trials`,
which is absent on this machine, and its GT file paths still use the old
`../Registration/Landmark/Landmark Trajectories` hierarchy. It defaults to
`GRID=False`, `TRIALS=[4]`, `MEAN=False`, and `SHOW_3D_MODEL=True`.

## What is needed for regeneration with larger axis fonts

Likely trial: `HT04`, left occlusion scenario. This is supported by the saved
script configuration and the results CSV: trial 4 has the largest three-camera,
left-scenario, basic monocular back-trajectory RMSE (16.181444 mm). The image itself
does not encode a trial number, so confirm it against the original measured data.

Look for the processed recording folder:

```text
Landmark_Registration_Trials/HT04/_HT04_LEFT/
```

For a standalone redraw, the required JSON recordings are the nine fiducials
`R1`–`R9` and the two trajectories `RB` and `RF`. They need the `poses` arrays for
3/5 cameras and each of `Mono`, `Mono Kalman`, `Mono Kalman Adaptive`, `Stereo`,
`Stereo Kalman`, and `Stereo Kalman Adaptive`. Example key: `3 Cam Mono`.
The legacy `results_table` additionally expects the ten target recordings
`01`–`10`, although those points are not needed to register the trajectory curves.

The local `LARK_Datasets.zip` contains error-only CSV tables. They have no XYZ
samples, so they cannot reproduce the blue measured curves. Neither the original
processed JSON recordings nor saved transformed trajectory arrays were found
in the inspected GitHub/Codeberg checkouts or supplementary software archive.
The NIST data page links the raw video release; rerunning tracking from those
videos is another route if the processed JSONs are lost.

For a fresh data-based figure, use one shared coordinate translation, identical
axis bounds across all panels, larger/fewer tick labels, and the same camera
view. The legacy plotting function computes a separate offset and axis limits
for each panel, which slightly changes the displayed coordinate frame when an
outlier changes the data bounds. Its head-mesh positioning is a visualization
fit to landmark bounds, not an anatomical surface registration.
