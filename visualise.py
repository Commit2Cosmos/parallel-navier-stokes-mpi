import io
import h5py
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

filename = "fields.h5"
gif_name = "fields_evolution.gif"

with h5py.File(filename, "r") as f:
    x = f["/geometry/x_coords"][:]
    y = f["/geometry/y_coords"][:]

    step_names = sorted(name for name in f.keys() if name.startswith("step_"))

    extent = [x.min(), x.max(), y.min(), y.max()]

    x_plot = x[1:-1]
    y_plot = y[1:-1]
    X, Y = np.meshgrid(x_plot, y_plot)

    arrow_stride = max(1, min(len(x), len(y)) // 24)

    smin, smax = np.inf, -np.inf
    max_speed = 0.0

    for step in step_names:
        u = np.asarray(f[f"{step}/u"][:], dtype=float)
        v = np.asarray(f[f"{step}/v"][:], dtype=float)
        speed = np.sqrt(u**2 + v**2)

        finite_speed = speed[np.isfinite(speed)]
        if finite_speed.size:
            smin = min(smin, finite_speed.min())
            smax = max(smax, finite_speed.max())
            max_speed = max(max_speed, finite_speed.max())

    if not np.isfinite(smin) or not np.isfinite(smax):
        smin, smax = 0.0, 1.0
    elif np.isclose(smin, smax):
        eps = max(1e-12, 1e-6 * max(1.0, abs(smin)))
        smin -= eps
        smax += eps

    norm = colors.Normalize(vmin=smin, vmax=smax)
    quiver_scale = max(max_speed, 1e-12) / 0.008

    frames = []

    for step in step_names:
        grp = f[step]
        u = np.asarray(grp["u"][:], dtype=float)
        v = np.asarray(grp["v"][:], dtype=float)
        speed = np.sqrt(u**2 + v**2)
        time = grp.attrs["time"]

        fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)

        im = ax.imshow(speed, origin="lower", extent=extent, norm=norm, cmap="viridis")
        
        uu = np.where(np.isfinite(u), u, 0.0)
        vv = np.where(np.isfinite(v), v, 0.0)
        mask = np.isfinite(u) & np.isfinite(v)

        plot_stride = arrow_stride * 2

        ax.quiver(
            X[::plot_stride, ::plot_stride],
            Y[::plot_stride, ::plot_stride],
            uu[::plot_stride, ::plot_stride],
            vv[::plot_stride, ::plot_stride],
            mask[::plot_stride, ::plot_stride],
            cmap="gray",
            clim=(0, 1),
            pivot="mid",
            angles="xy",
            scale_units="xy",
            scale=quiver_scale,
            width=0.003,
            minlength=0,
            alpha=0.6,
        )

        ax.set_title(f"Velocity field \nt = {time:.3f}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, label="Velocity magnitude (m/s)")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

imageio.mimsave(gif_name, frames, duration=0.12)
print(f"Saved {gif_name}")