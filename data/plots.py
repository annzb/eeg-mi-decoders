import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle
from scipy.interpolate import griddata


COLOR_PURPLE = '#bf00ff'
COLOR_TEAL = "#00bfbf"


def plot_scalp(signal, channel_locations, title=None):
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

    fig, ax = plt.subplots(figsize=(6, 6))
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

    cmap = plt.get_cmap("tab10" if n_traces <= 10 else "tab20")
    colors = [cmap(i % cmap.N) for i in range(n_traces)]
    plt.figure(figsize=(10, 4))
    for i in range(n_traces):
        plt.plot(t, channel_data[i], label=str(label_names[i]), linewidth=1.25, color=colors[i])
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    if title:
        plt.title(title)
    plt.legend(frameon=False)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_eeg_heatmap(t, sample, channels=None, title=None):
    block_h_in = 0.1
    robust_pct=98.0
    
    sample = np.asarray(sample, dtype=float)
    if sample.ndim != 2:
        raise ValueError(f"sample must be (n_channels, n_samples); got {sample.shape}")

    n_channels, _ = sample.shape
    if channels is None:
        channels = list(range(n_channels))
    channels = sorted(list(channels))
    if any((ch < 0 or ch >= n_channels) for ch in channels):
        bad = [ch for ch in channels if ch < 0 or ch >= n_channels]
        raise IndexError(f"channels out of range 0..{n_channels-1}: {bad}")

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
    ax.set_ylabel("Channel Index")
    if title:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Amplitude (µV)")
    plt.show()
