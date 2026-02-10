import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle
from scipy.interpolate import griddata


def plot_scalp(signal, channel_locations, channel_names=None, title=None):
    figsize = 7
    head_radius=1.0
    grid_res=300
    robust_pct=98.0
    sensor_size=35
    sensor_margin=0.90
    boundary_value=0.0
    boundary_n=128

    sensor_xy_raw = np.asarray(channel_locations, dtype=float)
    values = np.asarray(signal, dtype=float)
    if sensor_xy_raw.ndim != 2 or sensor_xy_raw.shape[1] != 2:
        raise ValueError(f"channel_locations must be (n_channels, 2); got {sensor_xy_raw.shape}")
    if values.ndim != 1 or values.shape[0] != sensor_xy_raw.shape[0]:
        raise ValueError(f"signal must be (n_channels,) matching locations; got {values.shape} vs {sensor_xy_raw.shape[0]}")
    if channel_names is not None:
        if not hasattr(channel_names, "__len__") or len(channel_names) != sensor_xy_raw.shape[0]:
            raise ValueError(f"channel_names must have length {sensor_xy_raw.shape[0]}; got {len(channel_names)}")

    sensor_xy = sensor_xy_raw.copy()
    sensor_xy -= sensor_xy.mean(axis=0, keepdims=True)
    r = np.linalg.norm(sensor_xy, axis=1)
    rmax = float(np.max(r)) if np.max(r) > 0 else 1.0
    sensor_xy = (sensor_xy / rmax) * (sensor_margin * head_radius)

    vmax = np.percentile(np.abs(values), robust_pct)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = np.max(np.abs(values)) if np.max(np.abs(values)) > 0 else 1.0
    vmin = -vmax

    theta = np.linspace(0.0, 2.0 * np.pi, boundary_n, endpoint=False)
    boundary_r = head_radius * 0.999
    boundary_xy = np.c_[boundary_r * np.cos(theta), boundary_r * np.sin(theta)]
    points = np.vstack([sensor_xy, boundary_xy])
    point_values = np.concatenate([values, np.full(boundary_n, boundary_value, dtype=float)])

    lin = np.linspace(-head_radius, head_radius, grid_res)
    gx, gy = np.meshgrid(lin, lin)
    zi = griddata(points, point_values, (gx, gy), method="cubic")

    if np.isnan(zi).any():
        zi_nn = griddata(points, point_values, (gx, gy), method="nearest")
        zi = np.where(np.isnan(zi), zi_nn, zi)

    mask = (gx**2 + gy**2) > (head_radius**2)
    zi = np.ma.array(zi, mask=mask)

    fig, ax = plt.subplots(figsize=(figsize, figsize))
    ax.set_aspect("equal")
    ax.axis("off")
    im = ax.imshow(
        zi,
        origin="lower",
        extent=[-head_radius, head_radius, -head_radius, head_radius],
        cmap="jet",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )
    ax.add_patch(Circle((0, 0), head_radius, fill=False, linewidth=2, edgecolor="k"))
    nose_size = 0.07 * head_radius
    ax.plot(
        [-nose_size, 0.0, nose_size, -nose_size],
        [head_radius, head_radius + nose_size, head_radius, head_radius],
        color="k",
        linewidth=2,
    )
    if channel_names is not None:
        for (x, y), name in zip(sensor_xy, channel_names):
            ax.text(
                x, y, str(name),
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white",
                zorder=5,
                path_effects=[pe.Stroke(linewidth=1.5, foreground="black"), pe.Normal()]
            )
    else:
        ax.scatter(sensor_xy[:, 0], sensor_xy[:, 1], c="k", s=sensor_size, linewidths=0.0, alpha=0.75)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Amplitude (µV)")
    if title:
        ax.set_title(title)
    ax.set_xlim(-1.12 * head_radius, 1.12 * head_radius)
    ax.set_ylim(-1.12 * head_radius, 1.12 * head_radius)
    plt.tight_layout()
    plt.show()


def plot_eeg_channel_joint(t, channel_data, label_names=None, title=None):
    if channel_data.ndim == 1:
        channel_data = channel_data[None, :]
    if channel_data.ndim != 2:
        raise ValueError(f"`channel_data` must be 2D (n_traces, n_samples); got {channel_data.shape}")
    n_traces, n_samples = channel_data.shape
    if t.ndim != 1 or t.shape[0] != n_samples:
        raise ValueError(f"`t` must be 1D with length n_samples={n_samples}; got t.shape={t.shape}")
    if label_names is None:
        label_names = [f"Class {i}" for i in range(n_traces)]
    else:
        if len(label_names) != n_traces:
            raise ValueError(f"label_names must have length {n_traces}; got {len(label_names)}")

    cmap = plt.get_cmap("rainbow")
    colors = [cmap(i / n_traces) for i in range(n_traces)]
    plt.figure(figsize=(10, 4))
    for i in range(n_traces):
        plt.plot(t, channel_data[i], label=str(label_names[i]), linewidth=1.25, color=colors[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    if title:
        plt.title(title)
    plt.legend(frameon=False, fontsize="small")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_eeg_heatmap(t, sample, channels=None, channel_names=None, title=None):
    block_h_in = 0.1
    robust_pct=98.0
    
    sample = np.asarray(sample, dtype=float)
    if sample.ndim != 2:
        raise ValueError(f"sample must be (n_channels, n_samples); got {sample.shape}")

    n_channels = sample.shape[0]
    if channels is None:
        channels = list(range(n_channels))
    channels = sorted(list(channels))
    if any((ch < 0 or ch >= n_channels) for ch in channels):
        bad = [ch for ch in channels if ch < 0 or ch >= n_channels]
        raise IndexError(f"channels out of range 0..{n_channels-1}: {bad}")
    if channel_names is not None:
        if len(channel_names) != n_channels:
            raise ValueError(f"channel_names must have length {n_channels}; got {len(channel_names)}")

    img = sample[channels, :]
    vmax = np.percentile(np.abs(img), robust_pct)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = np.max(np.abs(img)) if np.max(np.abs(img)) > 0 else 1.0
    fig_h_in = block_h_in * len(channels)
    fig, ax = plt.subplots(figsize=(10, fig_h_in))
    im = ax.imshow(
        img,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, len(channels)],
        vmin=-vmax,
        vmax=vmax,
        cmap="plasma",
        interpolation="nearest",
    )
    # ax.hlines(np.arange(1, len(channels)), xmin=t[0], xmax=t[-1], colors="k", linewidth=0.3, alpha=0.1)
    ax.set_xlabel("Time (s)")
    if channel_names is not None:
        ax.set_yticks(np.arange(len(channels)) + 0.5)
        ax.set_yticklabels([channel_names[ch] for ch in channels], fontsize=6)
    ax.set_ylabel("Channel")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Amplitude (µV)")
    plt.show()


def plot_subject_accuracies(
    scores: dict,
    summary: dict | None = None,
    show_mean_line: bool = True,
    show_ci_band: bool = True,
    ci_key: str = "ci95_half"
):
    color_above_ucl = "teal"
    color_below_ucl = "orangered"
    ucl_color = "dodgerblue"

    if not isinstance(scores, dict) or len(scores) == 0:
        raise ValueError("scores must be a non-empty dict from evaluation.evaluate_all_subjects().")

    subj_ids = list(scores.keys())
    means, stds, ucls = [], [], []
    for sid in subj_ids:
        r = scores[sid]
        if not hasattr(r, "mean"):
            raise ValueError("Each scores[subject] must have attribute .mean")
        means.append(float(r.mean))
        if hasattr(r, "std") and r.std is not None:
            stds.append(float(r.std))
        else:
            stds.append(np.nan)
        if hasattr(r, 'ucl_accuracy') and getattr(r, 'ucl_accuracy') is not None:
            ucls.append(float(getattr(r, 'ucl_accuracy')))
        else:
            ucls.append(np.nan)

    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)
    ucls = np.asarray(ucls, dtype=float)
    is_fraction = np.nanmax(means) <= 1.2
    y_label = "Accuracy" + ("" if is_fraction else " (%)")
    order = np.argsort(means)
    means_s = means[order]
    stds_s = stds[order]
    ucls_s = ucls[order]
    x = np.arange(len(means_s))
    has_ucl = np.isfinite(ucls_s)
    above = has_ucl & (means_s >= ucls_s)
    n_above = int(np.sum(above))
    n_total = int(np.sum(has_ucl)) if np.any(has_ucl) else len(means_s)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    # Main accuracy line
    above_mask = above.copy()
    above_mask[~has_ucl] = False
    colors = np.where(above_mask, color_above_ucl, color_below_ucl)
    label_below = f'accuracy (below UCL)'
    label_above = f'accuracy (at or above UCL)'
    line_labeled = {"below_ucl": False, "above_ucl": False}
    for i in range(len(means_s) - 1):
        c = colors[i+1]
        above_ucl = c == color_above_ucl
        ax.plot(
            x[i:i+2], means_s[i:i+2], linewidth=2.8, color=c, zorder=3, 
            label=(label_above if above_ucl and not line_labeled["above_ucl"] else label_below if not line_labeled["below_ucl"] else None)
        )
        if above_ucl:
            line_labeled["above_ucl"] = True
        else:
            line_labeled["below_ucl"] = True

    # Markers colored by chunk color
    for c in (color_below_ucl, color_above_ucl):
        idx = np.where(colors == c)[0]
        if idx.size:
            ax.scatter(x[idx], means_s[idx], s=18, color=c, zorder=4)

    # Per-subject repeat variability colored by chunk color
    if np.isfinite(stds_s).any():
        for c in (color_below_ucl, color_above_ucl):
            idx = np.where((colors == c) & np.isfinite(stds_s))[0]
            if idx.size:
                ax.errorbar(x[idx], means_s[idx], yerr=stds_s[idx], fmt="none", ecolor=c, elinewidth=1.0, capsize=2, alpha=0.6, zorder=2)

    # UCL thresholds
    if np.isfinite(ucls_s).all():
        ax.plot(x, ucls_s, linestyle=':', linewidth=1.3, color=ucl_color, zorder=2, label="UCL threshold (α=0.05)")
    elif np.isfinite(ucls_s).any():
        finite = np.isfinite(ucls_s)
        idx = np.where(finite)[0]
        runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for j, run in enumerate(runs):
            ax.plot(x[run], ucls_s[run], linestyle=':', linewidth=1.3, color=ucl_color, zorder=2, label="UCL threshold (α=0.05)" if j == 0 else None)

    # Mean + CI across subjects
    mean_all = None
    if summary is not None and "mean" in summary:
        mean_all = float(summary["mean"])
        if show_mean_line:
            ax.axhline(mean_all, linestyle="--", linewidth=1.2, alpha=0.9, color='purple', label=f'{mean_all*100:.2f}% mean accuracy')
        if show_ci_band:
            if ci_key not in summary:
                raise ValueError(f"summary must contain '{ci_key}' to draw CI band.")
            ci_half = float(summary[ci_key])
            ax.axhspan(mean_all - ci_half, mean_all + ci_half, alpha=0.15, zorder=0, color='purple', label="mean accuracy ± 95% CI")

    # Title
    title = f'Classification Accuracy'
    if np.isfinite(ucls_s).any():
        title += f'. Subjects ≥ UCL: {n_above}/{n_total} ({n_above/n_total*100:.2f}%)'
    ax.set_title(title)
    ax.set_xlabel("Subjects (sorted by accuracy)")
    ax.set_ylabel(y_label)

    n = len(means_s)
    if n > 25:
        ticks = np.linspace(0, n - 1, num=10, dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t + 1) for t in ticks])
    else:
        ax.set_xticks(x)
        ax.set_xticklabels([str(i + 1) for i in x])
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim((0.0, 1.0) if is_fraction else (0.0, 100.0))
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(frameon=False, loc="lower right")
    plt.tight_layout()
    plt.show()
