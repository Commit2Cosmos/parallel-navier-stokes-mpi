import io
import h5py
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt

filename = "fields.h5"
gif_name = "fields_evolution.gif"

with h5py.File(filename, "r") as f:
    x = f["/geometry/x_coords"][:]
    y = f["/geometry/y_coords"][:]

    x_in = x[1:-1]
    y_in = y[1:-1]

    step_names = sorted([name for name in f.keys() if name.startswith("step_")])

    extent = [x_in.min(), x_in.max(), y_in.min(), y_in.max()]

    # Pre-scan for stable color limits across all frames
    pmin, pmax = np.inf, -np.inf
    umin, umax = np.inf, -np.inf
    smin, smax = np.inf, -np.inf

    for step in step_names:
        P = f[f"{step}/P"][:]
        u = f[f"{step}/u"][:]
        v = f[f"{step}/v"][:]
        speed = np.sqrt(u**2 + v**2)

        pmin = min(pmin, np.nanmin(P))
        pmax = max(pmax, np.nanmax(P))
        umin = min(umin, np.nanmin(u))
        umax = max(umax, np.nanmax(u))
        smin = min(smin, np.nanmin(speed))
        smax = max(smax, np.nanmax(speed))

    frames = []

    for step in step_names:
        grp = f[step]
        P = grp["P"][:]
        u = grp["u"][:]
        v = grp["v"][:]
        speed = np.sqrt(u**2 + v**2)
        time = grp.attrs["time"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)

        im0 = axes[0].imshow(P, origin="lower", extent=extent,
                             vmin=pmin, vmax=pmax, cmap="viridis")
        axes[0].set_title(f"Pressure P\nt = {time:.3f}")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(speed, origin="lower", extent=extent,
                             vmin=smin, vmax=smax, cmap="magma")
        axes[1].set_title("Speed |v|")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(im1, ax=axes[1])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

imageio.mimsave(gif_name, frames, duration=0.12)
print(f"Saved {gif_name}")