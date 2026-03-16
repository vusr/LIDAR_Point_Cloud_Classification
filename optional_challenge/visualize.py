"""
Visualization helpers for the optional-challenge pipeline.

Outputs produced (all written to results/visualization/):
  frame_XX.png          — per-frame BEV with trails, velocity arrows,
                          semi-transparent filled bboxes, confidence labels
  trajectories.png      — 3-panel trajectory plot (one panel per class)
  dashboard.png         — summary stats: track counts, confidence, lifetimes, mix
  class_distribution.png — point counts per semantic class per frame
  tracking.gif          — animated BEV across all 10 frames  (requires Pillow)

All plots use matplotlib with the non-interactive "Agg" backend.
"""

import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import numpy as np

from tracker import TrackedObject

# ---------------------------------------------------------------------------
# Colour scheme
# ---------------------------------------------------------------------------

CLASS_COLORS: dict[str, str] = {
    "ground":     "#555566",
    "background": "#999999",
    "car":        "#e74c3c",
    "pedestrian": "#3498db",
    "bicyclist":  "#2ecc71",
}

TRACK_CLASS_COLORS: dict[str, str] = {
    "car":        "#ff6b6b",
    "pedestrian": "#74b9ff",
    "bicyclist":  "#55efc4",
}

LABEL_TO_CLASS = {0: "ground", 1: "background", 2: "car", 3: "pedestrian", 4: "bicyclist"}

MAX_RENDER_POINTS: int = 60_000
TRAIL_ALPHA_MIN:   float = 0.10   # oldest ghost
TRAIL_ALPHA_MAX:   float = 0.55   # most recent ghost


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _dark_axis(ax, xlabel="X (m)", ylabel="Y (m)"):
    ax.set_facecolor("#12121f")
    ax.tick_params(colors="#aaaaaa", labelsize=7)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=8)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")


def _sensor_origin(ax):
    """Mark the LiDAR sensor position at the origin."""
    ax.scatter(0, 0, s=60, c="#f1c40f", marker="*", zorder=10)
    ax.text(0.5, -1.5, "sensor", fontsize=5, color="#f1c40f", ha="center", zorder=10)


def _subsample(points, labels, cap):
    n = len(points)
    if n <= cap:
        return points, labels
    idx = np.random.choice(n, cap, replace=False)
    return points[idx], labels[idx]


def _track_color_map(all_tracked_per_frame: list[list[TrackedObject]]) -> dict[int, str]:
    """
    Assign a unique colour to each track ID within its class bucket so that
    trajectories for the same class are distinguishable from each other.
    """
    class_tracks: dict[str, list[int]] = defaultdict(list)
    seen = set()
    for frame in all_tracked_per_frame:
        for obj in frame:
            if obj.track_id not in seen:
                class_tracks[obj.class_name].append(obj.track_id)
                seen.add(obj.track_id)

    colormaps = {"car": "Reds", "pedestrian": "Blues", "bicyclist": "Greens"}
    color_map: dict[int, str] = {}
    for cls, track_ids in class_tracks.items():
        cmap = plt.get_cmap(colormaps.get(cls, "gray"))
        for i, tid in enumerate(track_ids):
            frac = 0.4 + 0.55 * (i / max(len(track_ids) - 1, 1))
            color_map[tid] = mcolors.to_hex(cmap(frac))
    return color_map


# ---------------------------------------------------------------------------
# 1. Per-frame BEV
# ---------------------------------------------------------------------------

def save_bev_frame(
    points: np.ndarray,
    labels: np.ndarray,
    tracked_objects: list[TrackedObject],
    frame_idx: int,
    output_path: str,
    trail_history: dict[int, list[tuple[float, float]]] = None,
    xlim: tuple[float, float] = (-50, 50),
    ylim: tuple[float, float] = (-50, 50),
    point_size: float = 0.5,
):
    """
    BEV frame with:
      - Points coloured by semantic class
      - Ghost trails showing recent positions of each track
      - Velocity arrow (last displacement direction)
      - Filled semi-transparent bounding boxes
      - Track ID + class + confidence label
      - Sensor origin marker
      - Frame stats HUD

    Args:
        trail_history: {track_id: [(x0,y0), (x1,y1), ...]} — positions from
                       previous frames in chronological order. The last entry is
                       the most recent.  Pass None or {} if not available.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    trail_history = trail_history or {}

    fig, ax = plt.subplots(figsize=(11, 11))
    fig.patch.set_facecolor("#0d0d1a")
    _dark_axis(ax)

    # --- Point cloud scatter ---
    pts_r, lbl_r = _subsample(points, labels, MAX_RENDER_POINTS)
    for lbl_idx, cls_name in LABEL_TO_CLASS.items():
        mask = lbl_r == lbl_idx
        if not mask.any():
            continue
        # Reduce alpha for uninformative classes to keep objects prominent
        alpha = 0.25 if cls_name in ("ground", "background") else 0.75
        ps    = point_size * 0.6 if cls_name in ("ground", "background") else point_size * 1.4
        ax.scatter(
            pts_r[mask, 0], pts_r[mask, 1],
            s=ps, c=CLASS_COLORS[cls_name], alpha=alpha,
            linewidths=0, label=cls_name, rasterized=True,
        )

    # --- Ghost trails ---
    if trail_history:
        for obj in tracked_objects:
            past = trail_history.get(obj.track_id, [])
            if len(past) < 2:
                continue
            color = TRACK_CLASS_COLORS.get(obj.class_name, "#ffffff")
            n_trail = len(past)
            for i in range(len(past) - 1):
                # Fade alpha along the trail (older = more transparent)
                alpha = TRAIL_ALPHA_MIN + (TRAIL_ALPHA_MAX - TRAIL_ALPHA_MIN) * (i / (n_trail - 1))
                ax.plot(
                    [past[i][0], past[i + 1][0]],
                    [past[i][1], past[i + 1][1]],
                    color=color, alpha=alpha, linewidth=1.0, zorder=4,
                )
            # Ghost dot at each past position
            for i, (px, py) in enumerate(past[:-1]):
                a = TRAIL_ALPHA_MIN + (TRAIL_ALPHA_MAX - TRAIL_ALPHA_MIN) * (i / n_trail)
                ax.scatter(px, py, s=6, c=color, alpha=a, linewidths=0, zorder=4)

    # --- Tracked objects: bbox + velocity arrow + label ---
    for obj in tracked_objects:
        color = TRACK_CLASS_COLORS.get(obj.class_name, "#ffffff")
        cx, cy = float(obj.centroid[0]), float(obj.centroid[1])
        bx1 = float(obj.bbox_min[0]);  by1 = float(obj.bbox_min[1])
        bx2 = float(obj.bbox_max[0]);  by2 = float(obj.bbox_max[1])
        bw  = bx2 - bx1;              bh  = by2 - by1

        # Filled semi-transparent bbox
        ax.add_patch(patches.FancyBboxPatch(
            (bx1, by1), bw, bh,
            boxstyle="square,pad=0",
            linewidth=1.4, edgecolor=color,
            facecolor=color, alpha=0.18, zorder=5,
        ))
        # Solid outline on top
        ax.add_patch(patches.Rectangle(
            (bx1, by1), bw, bh,
            linewidth=1.4, edgecolor=color, facecolor="none", zorder=6,
        ))

        # Velocity arrow: direction from previous trail position
        past = trail_history.get(obj.track_id, [])
        if len(past) >= 2:
            dx = past[-1][0] - past[-2][0]
            dy = past[-1][1] - past[-2][1]
            speed = np.sqrt(dx ** 2 + dy ** 2)
            if speed > 0.05:
                scale = min(speed * 2.5, 3.0)
                ax.annotate(
                    "", xy=(cx + dx / speed * scale, cy + dy / speed * scale),
                    xytext=(cx, cy),
                    arrowprops=dict(
                        arrowstyle="->,head_width=0.3,head_length=0.4",
                        color=color, lw=1.3,
                    ),
                    zorder=7,
                )

        # Label box: "#{id} cls\nconf%"
        label_text = f"#{obj.track_id} {obj.class_name[:3]}\n{obj.confidence:.0%}"
        ax.text(
            cx, by2 + 0.3, label_text,
            fontsize=5.5, color="white", ha="center", va="bottom",
            zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=color, alpha=0.70, edgecolor="none",
            ),
        )

    _sensor_origin(ax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")

    # Legend
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(
        by_label.values(), by_label.keys(),
        loc="upper right", fontsize=7,
        facecolor="#1e1e30", labelcolor="white",
        markerscale=6, framealpha=0.8,
    )

    # HUD: active track counts
    from collections import Counter
    cnt = Counter(o.class_name for o in tracked_objects)
    hud_lines = [f"Frame {frame_idx:02d}"]
    for cls in ("car", "pedestrian", "bicyclist"):
        c = cnt.get(cls, 0)
        if c:
            hud_lines.append(f"  {cls[:3].upper()} {c}")
    ax.text(
        0.02, 0.98, "\n".join(hud_lines),
        transform=ax.transAxes, fontsize=8, color="white",
        va="top", ha="left", zorder=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#1e1e30", alpha=0.75),
    )

    ax.set_title(
        f"Frame {frame_idx:02d} — BEV Segmentation & Tracking",
        color="white", fontsize=12, pad=10,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(output_path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Trajectory plot — 3-panel, one per trackable class
# ---------------------------------------------------------------------------

def save_trajectory_plot(
    all_tracked: list[list[TrackedObject]],
    output_path: str,
    xlim: tuple[float, float] = (-50, 50),
    ylim: tuple[float, float] = (-50, 50),
):
    """
    3-panel trajectory plot (car / pedestrian / bicyclist).

    Each track gets a unique shade within its class colourmap.
    Line thickness encodes average speed; start = circle, end = triangle.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    color_map = _track_color_map(all_tracked)

    # Accumulate per-track positions
    track_info: dict[int, dict] = {}
    for frame_idx, frame_tracks in enumerate(all_tracked):
        for obj in frame_tracks:
            if obj.track_id not in track_info:
                track_info[obj.track_id] = {
                    "class_name": obj.class_name,
                    "positions":  [],
                }
            track_info[obj.track_id]["positions"].append(
                (frame_idx, float(obj.centroid[0]), float(obj.centroid[1]))
            )

    trackable = ("car", "pedestrian", "bicyclist")
    fig, axes = plt.subplots(1, 3, figsize=(21, 8), facecolor="#0d0d1a")

    for ax, cls_name in zip(axes, trackable):
        _dark_axis(ax)
        _sensor_origin(ax)

        cls_tracks = {
            tid: info for tid, info in track_info.items()
            if info["class_name"] == cls_name and len(info["positions"]) >= 2
        }

        for tid, info in cls_tracks.items():
            pos  = info["positions"]
            xs   = [p[1] for p in pos]
            ys   = [p[2] for p in pos]
            col  = color_map.get(tid, TRACK_CLASS_COLORS[cls_name])

            # Speed = mean step distance (m per frame)
            dists = [
                np.sqrt((xs[i+1]-xs[i])**2 + (ys[i+1]-ys[i])**2)
                for i in range(len(xs)-1)
            ]
            avg_speed = float(np.mean(dists)) if dists else 0.0
            lw = 0.8 + min(avg_speed * 1.2, 3.0)

            ax.plot(xs, ys, color=col, linewidth=lw, alpha=0.85, zorder=3)
            ax.scatter(xs[0],  ys[0],  s=30, c=col, marker="o", zorder=5)
            ax.scatter(xs[-1], ys[-1], s=45, c=col, marker="^", zorder=5)
            ax.text(
                xs[-1] + 0.4, ys[-1] + 0.4,
                f"#{tid}", fontsize=5.5, color=col, zorder=6,
            )

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(
            f"{cls_name.capitalize()}  ({len(cls_tracks)} tracks)",
            color="white", fontsize=11, pad=8,
        )
        # Speed legend
        ax.text(
            0.03, 0.03,
            "Line width ∝ speed",
            transform=ax.transAxes, fontsize=6, color="#aaaaaa",
        )

    fig.suptitle("Object Trajectories across 10 Frames", color="white", fontsize=14, y=1.01)
    fig.tight_layout(pad=1.0)
    fig.savefig(output_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Summary dashboard
# ---------------------------------------------------------------------------

def save_dashboard(
    all_tracked: list[list[TrackedObject]],
    per_frame_counts: list[dict[str, int]],
    output_path: str,
):
    """
    2×2 summary dashboard:
      [TL] Track counts per frame (stacked area)
      [TR] Confidence distribution per class (violin)
      [BL] Track lifetime histogram
      [BR] Overall point-class mix (pie)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(16, 10), facecolor="#0d0d1a")
    gs  = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.30)
    ax_counts  = fig.add_subplot(gs[0, 0])
    ax_conf    = fig.add_subplot(gs[0, 1])
    ax_life    = fig.add_subplot(gs[1, 0])
    ax_pie     = fig.add_subplot(gs[1, 1])

    trackable = ("car", "pedestrian", "bicyclist")
    n_frames  = len(all_tracked)
    frames    = list(range(n_frames))

    # --- TL: stacked area — track counts per frame -------------------------
    counts_by_class = {cls: [] for cls in trackable}
    for frame_tracks in all_tracked:
        from collections import Counter
        cnt = Counter(o.class_name for o in frame_tracks)
        for cls in trackable:
            counts_by_class[cls].append(cnt.get(cls, 0))

    ax_counts.set_facecolor("#12121f")
    bottoms = np.zeros(n_frames)
    for cls in trackable:
        vals = np.array(counts_by_class[cls], dtype=float)
        ax_counts.fill_between(
            frames, bottoms, bottoms + vals,
            alpha=0.75, label=cls,
            color=TRACK_CLASS_COLORS[cls],
        )
        ax_counts.plot(frames, bottoms + vals, color=TRACK_CLASS_COLORS[cls], lw=0.8)
        bottoms += vals
    ax_counts.set_xlim(0, n_frames - 1)
    ax_counts.set_xticks(frames)
    ax_counts.set_xticklabels([f"f{i:02d}" for i in frames], color="#aaaaaa", fontsize=7)
    ax_counts.set_ylabel("Active tracks", color="#aaaaaa", fontsize=8)
    ax_counts.set_title("Active Track Counts per Frame", color="white", fontsize=10)
    ax_counts.tick_params(colors="#aaaaaa")
    ax_counts.legend(fontsize=7, facecolor="#1e1e30", labelcolor="white")
    ax_counts.grid(axis="y", alpha=0.2, color="white")
    for spine in ax_counts.spines.values():
        spine.set_edgecolor("#333344")

    # --- TR: violin — confidence per class ---------------------------------
    conf_by_class: dict[str, list[float]] = {cls: [] for cls in trackable}
    for frame_tracks in all_tracked:
        for obj in frame_tracks:
            if obj.class_name in conf_by_class:
                conf_by_class[obj.class_name].append(obj.confidence)

    ax_conf.set_facecolor("#12121f")

    # Only plot classes that have at least 2 data points (violin needs variance)
    violin_classes   = [cls for cls in trackable if len(conf_by_class[cls]) >= 2]
    violin_data      = [conf_by_class[cls] for cls in violin_classes]
    violin_positions = list(range(1, len(violin_classes) + 1))

    if violin_data:
        parts = ax_conf.violinplot(
            violin_data, positions=violin_positions,
            showmedians=True, showextrema=True,
        )
        for i, (pc, cls) in enumerate(zip(parts["bodies"], violin_classes)):
            pc.set_facecolor(TRACK_CLASS_COLORS[cls])
            pc.set_alpha(0.65)
        for key in ("cmedians", "cbars", "cmins", "cmaxes"):
            if key in parts:
                parts[key].set_color("#dddddd")
                parts[key].set_linewidth(1.0)
    else:
        ax_conf.text(0.5, 0.5, "No tracking data", transform=ax_conf.transAxes,
                     color="#aaaaaa", ha="center", va="center", fontsize=9)

    # Show all class labels on x-axis; mark absent ones with "(no data)"
    all_positions = list(range(1, len(trackable) + 1))
    all_labels    = [
        cls if len(conf_by_class[cls]) >= 2 else f"{cls}\n(no data)"
        for cls in trackable
    ]
    ax_conf.set_xticks(all_positions)
    ax_conf.set_xticklabels(all_labels, color="#aaaaaa", fontsize=8)
    ax_conf.set_xlim(0, len(trackable) + 1)
    ax_conf.set_ylabel("Confidence", color="#aaaaaa", fontsize=8)
    ax_conf.set_ylim(0, 1.05)
    ax_conf.set_title("Classifier Confidence Distribution", color="white", fontsize=10)
    ax_conf.tick_params(colors="#aaaaaa")
    ax_conf.grid(axis="y", alpha=0.2, color="white")
    for spine in ax_conf.spines.values():
        spine.set_edgecolor("#333344")

    # --- BL: histogram — track lifetimes -----------------------------------
    lifetime: dict[int, dict] = {}
    for frame_tracks in all_tracked:
        for obj in frame_tracks:
            if obj.track_id not in lifetime:
                lifetime[obj.track_id] = {"class": obj.class_name, "count": 0}
            lifetime[obj.track_id]["count"] += 1

    ax_life.set_facecolor("#12121f")
    for cls in trackable:
        lives = [v["count"] for v in lifetime.values() if v["class"] == cls]
        if not lives:
            continue
        ax_life.hist(
            lives, bins=range(1, n_frames + 2), alpha=0.7,
            label=cls, color=TRACK_CLASS_COLORS[cls],
            edgecolor="#12121f", linewidth=0.5,
        )
    ax_life.set_xlabel("Frames alive", color="#aaaaaa", fontsize=8)
    ax_life.set_ylabel("Number of tracks", color="#aaaaaa", fontsize=8)
    ax_life.set_title("Track Lifetime Distribution", color="white", fontsize=10)
    ax_life.tick_params(colors="#aaaaaa")
    ax_life.legend(fontsize=7, facecolor="#1e1e30", labelcolor="white")
    ax_life.grid(axis="y", alpha=0.2, color="white")
    for spine in ax_life.spines.values():
        spine.set_edgecolor("#333344")

    # --- BR: pie — total point-class mix across all frames -----------------
    ax_pie.set_facecolor("#12121f")
    totals: dict[str, int] = {cls: 0 for cls in CLASS_COLORS}
    for frame_cnt in per_frame_counts:
        for cls, n in frame_cnt.items():
            totals[cls] = totals.get(cls, 0) + n

    # Exclude zero-count classes
    pie_labels = [c for c, v in totals.items() if v > 0]
    pie_vals   = [totals[c] for c in pie_labels]
    pie_colors = [CLASS_COLORS[c] for c in pie_labels]

    wedges, texts, autotexts = ax_pie.pie(
        pie_vals, labels=pie_labels, colors=pie_colors,
        autopct="%1.1f%%", startangle=140,
        pctdistance=0.78,
        textprops={"color": "white", "fontsize": 8},
        wedgeprops={"linewidth": 0.5, "edgecolor": "#0d0d1a"},
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax_pie.set_title("Point-class Mix (all frames)", color="white", fontsize=10)

    fig.suptitle("Pipeline Summary Dashboard", color="white", fontsize=15, y=1.01)
    fig.savefig(output_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Animated GIF
# ---------------------------------------------------------------------------

def save_animated_gif(
    frame_png_paths: list[str],
    output_path: str,
    duration_ms: int = 600,
):
    """
    Stitch per-frame BEV PNGs into an animated GIF using Pillow.

    Args:
        frame_png_paths: Ordered list of PNG paths (frame_00.png … frame_09.png).
        output_path:     Path to write the GIF.
        duration_ms:     Display duration per frame in milliseconds.
    """
    try:
        from PIL import Image
    except ImportError:
        print("[warn] Pillow not installed — skipping animated GIF. Run: pip install Pillow")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    frames_pil = []
    for path in frame_png_paths:
        if not os.path.exists(path):
            print(f"[warn] Missing frame for GIF: {path}")
            continue
        img = Image.open(path).convert("RGB")
        # Downscale for a reasonable GIF file size
        w, h = img.size
        img = img.resize((w // 2, h // 2), Image.LANCZOS)
        frames_pil.append(img)

    if not frames_pil:
        print("[warn] No frames found — skipping animated GIF.")
        return

    frames_pil[0].save(
        output_path,
        save_all=True,
        append_images=frames_pil[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    print(f"  GIF saved → {output_path}  ({len(frames_pil)} frames)")


# ---------------------------------------------------------------------------
# 5. Class distribution bar chart (unchanged from v1)
# ---------------------------------------------------------------------------

def save_class_distribution_plot(
    per_frame_counts: list[dict[str, int]],
    output_path: str,
):
    """Bar chart — point counts per semantic class per frame."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    classes   = ["ground", "background", "car", "pedestrian", "bicyclist"]
    n_frames  = len(per_frame_counts)
    x         = np.arange(n_frames)
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(14, 5))
    for i, cls_name in enumerate(classes):
        counts = [per_frame_counts[f].get(cls_name, 0) for f in range(n_frames)]
        ax.bar(
            x + i * bar_width, counts, bar_width,
            label=cls_name, color=CLASS_COLORS[cls_name], alpha=0.85,
        )

    ax.set_xlabel("Frame", fontsize=10)
    ax.set_ylabel("Point count", fontsize=10)
    ax.set_title("Point distribution per semantic class per frame", fontsize=11)
    ax.set_xticks(x + bar_width * (len(classes) - 1) / 2)
    ax.set_xticklabels([f"f{i:02d}" for i in range(n_frames)])
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
