import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle
from scipy.interpolate import griddata
from scipy.signal import hilbert

from data.preprocess import filter_band_new
# from evaluation.result import DatasetEvalResult, Score


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


def plot_model_accuracies(subject_means: np.array, subject_stds: np.array, subject_ucls: np.array):
    color_above_ucl = "teal"
    color_below_ucl = "orangered"
    ucl_color = "dodgerblue"

    if not isinstance(subject_means, np.ndarray) or subject_means.size == 0:
        raise ValueError("subject_means must be a non-empty np array")
    if not isinstance(subject_stds, np.ndarray) or subject_stds.size == 0:
        raise ValueError("subject_stds must be a non-empty np array")
    if not isinstance(subject_ucls, np.ndarray) or subject_ucls.size == 0:
        raise ValueError("subject_ucls must be a non-empty np array")

    is_fraction = np.nanmax(subject_means) <= 1.2
    y_label = "Accuracy" + ("" if is_fraction else " (%)")
    order = np.argsort(subject_means)
    means_s = subject_means[order]
    stds_s = subject_stds[order]
    ucls_s = subject_ucls[order]
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
        prev_above = (colors[i] == color_above_ucl) if i > 0 else None
        entering = (prev_above is None) or (above_ucl != prev_above)
        label = None
        if entering:
            if above_ucl and not line_labeled["above_ucl"]:
                label = label_above
            elif (not above_ucl) and not line_labeled["below_ucl"]:
                label = label_below
        ax.plot(
            x[i:i+2], means_s[i:i+2],
            linewidth=2.8, color=c, zorder=3,
            label=label
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

    # Mean across subject means
    mean_all = means_s.mean()
    ax.axhline(mean_all, linestyle="--", linewidth=1.2, alpha=0.9, color='purple', label=f'{mean_all*100:.2f}% mean accuracy')

    # CI across subjects
    ci_half = float(1.96 * np.std(means_s, ddof=0) / np.sqrt(means_s.size))
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


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str] | None = None,
    labels: np.ndarray | list | None = None,
    normalize: str | None = "true",   # None | "true" | "pred" | "all"
    title: str | None = None,
    show_values: bool = True,
    value_fmt: str | None = None
):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1D; got {y_true.shape} and {y_pred.shape}")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred must have same length; got {y_true.shape[0]} vs {y_pred.shape[0]}")
    if normalize not in (None, "true", "pred", "all"):
        raise ValueError(f"normalize must be one of None|'true'|'pred'|'all'; got {normalize!r}")

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred], axis=0))
        try:
            labels = np.sort(labels)
        except Exception:
            pass
    else:
        labels = np.asarray(labels)

    n_classes = int(labels.shape[0])
    if n_classes == 0:
        raise ValueError("No labels to plot (empty `labels`).")

    if label_names is None:
        label_names = [str(x) for x in labels.tolist()]
    else:
        if not hasattr(label_names, "__len__") or len(label_names) != n_classes:
            raise ValueError(f"label_names must have length {n_classes}; got {len(label_names)}")

    idx = {labels[i]: i for i in range(n_classes)}
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        if yt not in idx or yp not in idx:
            continue
        cm[idx[yt], idx[yp]] += 1

    cm_plot = cm.astype(float)
    if normalize is not None:
        if normalize == "true":
            denom = cm_plot.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            denom = cm_plot.sum(axis=0, keepdims=True)
        else:  # "all"
            denom = cm_plot.sum(keepdims=True)
        cm_plot = np.divide(cm_plot, denom, out=np.zeros_like(cm_plot), where=(denom != 0))

    if value_fmt is None:
        value_fmt = ".2f" if normalize is not None else "d"
    fig, ax = plt.subplots(figsize=(1.0 + 0.65 * n_classes, 1.0 + 0.55 * n_classes))
    im = ax.imshow(cm_plot, interpolation="nearest", aspect="equal")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Proportion" if normalize is not None else "Count")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    if title is None:
        if normalize is None:
            title = "Confusion matrix (counts)"
        else:
            title = f"Confusion matrix (normalized: {normalize})"
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.25)
    ax.tick_params(which="minor", bottom=False, left=False)

    if show_values:
        vmax = float(np.max(cm_plot)) if np.isfinite(cm_plot).any() else 0.0
        thresh = 0.5 * vmax if vmax > 0 else 0.0
        for i in range(n_classes):
            for j in range(n_classes):
                v = cm_plot[i, j]
                if normalize is None:
                    s = format(int(cm[i, j]), value_fmt)
                else:
                    s = format(float(v), value_fmt)
                ax.text(
                    j, i, s,
                    ha="center", va="center",
                    fontsize=9,
                    color="white" if v > thresh else "black",
                )
    plt.tight_layout()
    plt.show()


def plot_eeg_rhythm_power(
    t,
    sample,
    title=None,
    smoothing_s=0.25,
    include_bands=None,
    min_cycles=3.0,
    use_hilbert=True,
):
    sample = np.asarray(sample, dtype=float)
    t = np.asarray(t, dtype=float)

    if sample.ndim != 2:
        raise ValueError(f"sample must be (n_channels, n_samples); got {sample.shape}")
    if t.ndim != 1 or t.shape[0] != sample.shape[1]:
        raise ValueError(
            f"t must be 1D with length n_samples={sample.shape[1]}; got t.shape={t.shape}"
        )
    if t.shape[0] < 2:
        raise ValueError("Need at least 2 time points to infer sampling rate")

    dt = np.diff(t)
    if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
        raise ValueError("t must be strictly increasing and finite")
    if not np.allclose(dt, dt[0], rtol=1e-3, atol=1e-6):
        raise ValueError("t must be uniformly sampled")

    sampling_rate = 1.0 / float(dt[0])
    duration_s = float(t[-1] - t[0])
    nyq = sampling_rate / 2.0

    band_defs = {
        "delta":        (0.5, 4.0),
        "theta":        (4.0, 8.0),
        "alpha":        (8.0, 13.0),
        "beta":         (13.0, 30.0),
        "gamma":        (30.0, 45.0),
        "low-hg":       (55.0, 80.0),
        "mid-hg":       (80.0, 100.0),
        "high-hg":      (100.0, 125.0),
    }
    colors = {
        "delta":        "tab:cyan",
        "theta":        "tab:blue",
        "alpha":        "tab:purple",
        "beta":         "tab:red",
        "gamma":        "tab:orange",
        "low-hg":       "tab:pink",
        "mid-hg":       "tab:olive",
        "high-hg":      "tab:brown",
    }

    if include_bands is None:
        include_bands = list(band_defs.keys())
    else:
        unknown = [b for b in include_bands if b not in band_defs]
        if unknown:
            raise ValueError(f"Unknown band names: {unknown}")

    # Combine all channels into one regional signal
    x = np.mean(sample, axis=0)

    # Prepare smoothing
    if smoothing_s < 0:
        raise ValueError(f"smoothing_s must be >= 0; got {smoothing_s}")
    smooth_n = max(1, int(round(smoothing_s * sampling_rate)))
    smooth_kernel = np.ones(smooth_n, dtype=float) / smooth_n

    valid_bands = []
    skipped = {}

    for name in include_bands:
        lo, hi = band_defs[name]

        # Band must fit below Nyquist
        if hi >= nyq:
            skipped[name] = f"hi={hi:g} Hz exceeds Nyquist={nyq:.2f} Hz"
            continue

        # Sample should contain enough cycles of the low cutoff
        if lo > 0 and duration_s * lo < min_cycles:
            skipped[name] = (
                f"duration too short for {min_cycles:g} cycles at {lo:g} Hz"
            )
            continue

        valid_bands.append((name, lo, hi))

    if not valid_bands:
        raise ValueError(
            "No rhythm bands are valid for this sample. "
            f"sampling_rate={sampling_rate:.3f} Hz, duration={duration_s:.3f} s"
        )

    # Use your preprocessing bandpass helper, which expects (N, Ch, Time)
    x3 = x[None, None, :]  # (1, 1, T)

    plt.figure(figsize=(10, 4.5))

    for name, lo, hi in valid_bands:
        x_band = filter_band_new(x3, sampling_rate=sampling_rate, lo=lo, hi=hi, order=4)[0, 0]

        if use_hilbert:
            power = np.abs(hilbert(x_band)) ** 2
        else:
            power = x_band ** 2

        if smooth_n > 1:
            power = np.convolve(power, smooth_kernel, mode="same")

        plt.plot(
            t,
            power,
            linewidth=1.6,
            label=f"{name} ({lo:g}–{hi:g} Hz)",
            color=colors.get(name, None),
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Power (µV²)")
    if title:
        plt.title(title)
    plt.legend(frameon=False, fontsize="small", ncol=2)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()
