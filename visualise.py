import io
import h5py
import imageio.v2 as imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


filename = "fields.h5"
gif_name = "fields_evolution.gif"


# Open the HDF5 file in read-only mode
with h5py.File(filename, "r") as f:
    # Read global x- and y-coordinate arrays
    x = f["/geometry/x_coords"][:]
    y = f["/geometry/y_coords"][:]

    # Collect all timestep groups in sorted order
    step_names = sorted(name for name in f.keys() if name.startswith("step_"))

    # Spatial extent used by imshow so axes match physical coordinates
    extent = [x.min(), x.max(), y.min(), y.max()]

    # Use interior coordinates only (exclude ghost cells)
    x_plot = x[1:-1]
    y_plot = y[1:-1]

    # Build coordinate grids for quiver arrows
    X, Y = np.meshgrid(x_plot, y_plot)
    arrow_stride = max(1, min(len(x), len(y)) // 24)

    smin, smax = np.inf, -np.inf
    max_speed = 0.0

    # Pass over all timesteps to determine consistent colour limits and quiver scaling
    for step in step_names:
        # Read velocity components for this timestep
        u = np.asarray(f[f"{step}/u"][:], dtype=float)
        v = np.asarray(f[f"{step}/v"][:], dtype=float)
        # Compute velocity magnitude
        speed = np.sqrt(u**2 + v**2)

        # Ignore non-finite values when computing plotting limits
        finite_speed = speed[np.isfinite(speed)]
        if finite_speed.size:
            smin = min(smin, finite_speed.min())
            smax = max(smax, finite_speed.max())
            max_speed = max(max_speed, finite_speed.max())

    # Provide safe fallback limits if the data are invalid or constant
    if not np.isfinite(smin) or not np.isfinite(smax):
        smin, smax = 0.0, 1.0
    elif np.isclose(smin, smax):
        eps = max(1e-12, 1e-6 * max(1.0, abs(smin)))
        smin -= eps
        smax += eps

    # Use one shared colour normalization for all frames
    norm = colors.Normalize(vmin=smin, vmax=smax)
    # Scale quiver arrows relative to the maximum speed
    quiver_scale = max(max_speed, 1e-12) / 0.008

    frames = []

    # Render one figure per timestep
    for step in step_names:
        # Access current timestep group
        grp = f[step]
        # Read velocity components
        u = np.asarray(grp["u"][:], dtype=float)
        v = np.asarray(grp["v"][:], dtype=float)
        speed = np.sqrt(u**2 + v**2)
        time = grp.attrs["time"]

        # Create a figure for the current frame
        fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)

        # Plot velocity magnitude as a coloured image
        im = ax.imshow(speed, origin="lower", extent=extent, norm=norm, cmap="viridis")
        
        # Replace non-finite values before plotting arrows
        uu = np.where(np.isfinite(u), u, 0.0)
        vv = np.where(np.isfinite(v), v, 0.0)
        # Mask used to colour valid arrow locations
        mask = np.isfinite(u) & np.isfinite(v)

        # Plot arrows more sparsely than the scalar field to avoid clutter
        plot_stride = arrow_stride * 2

        # Overlay velocity vectors on top of the magnitude field
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

        # Add title, axis labels, and colourbar
        ax.set_title(f"Velocity field \nt = {time:.3f}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, label="Velocity magnitude (m/s)")

        # Render the figure to an in-memory PNG buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120)
        plt.close(fig)
        buf.seek(0)
        # Read the PNG buffer back as an image and store it as a GIF frame
        frames.append(imageio.imread(buf))


# Assemble all frames into an animated GIF
imageio.mimsave(gif_name, frames, duration=0.12)
print(f"Saved {gif_name}")