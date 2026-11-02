"""
Plots: preprocessing checks, retrieval monitoring, retrieved field and
caustic (D4σ along the fixed x / y axes of the sensor).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.restoration import unwrap_phase

from utils import caustic_curve, fit_caustic


def _z_label(zs, i):
    """' at z = ... mm' if the distances are known."""
    return f"\nat z = {zs[i] * 1e3:.2f} mm" if zs is not None else ""


def _colorbar(ax, im):
    """Colorbar on the right of ax."""
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)


# ---------------------------------------------------------------------- #
# Preprocessing checks (ImageSetTrt with plot_bkg / plot_resize)
# ---------------------------------------------------------------------- #

def plot_bkg_images(ims_raw, cdms_raw, ims_bck, bck_cdms, z=None, cmap='gist_ncar'):
    """
    Raw images (top) and images after background removal (bottom), with
    their centroids. Colour scale: lowest 10 % of the intensity range, to
    show the background.
    """
    fig, axes = plt.subplots(2, len(ims_bck), figsize=(15, 7), squeeze=False)
    for i, (im_raw, cdm_raw, im_bck, bck_cdm) in enumerate(zip(ims_raw, cdms_raw,
                                                               ims_bck, bck_cdms)):
        for ax, im, cdm, title in ((axes[0, i], im_raw, cdm_raw, "Raw image"),
                                   (axes[1, i], im_bck, bck_cdm, "Background removed")):
            ax.imshow(im, cmap=cmap, vmin=im.min(), vmax=im.min() + (im.max() - im.min()) * 0.1)
            ax.plot(cdm[0], cdm[1], '1', color='red')
            ax.set_title(f"{title} #{i}{_z_label(z, i)}")
        axes[1, i].axis('off')
    fig.tight_layout()
    return fig


def plot_crop_lims(ims, cdms, crop_ims, crop_cdms, bounds, zs=None, cmap='gist_ncar'):
    """
    Crop limits on the images (top) and padded images with the crop box
    around their centroid (bottom).
    """
    fig, axes = plt.subplots(2, len(ims), figsize=(15, 7), squeeze=False)
    for i, (im, cdm, crop_im, crop_cdm, (x0, x1, y0, y1)) in enumerate(
            zip(ims, cdms, crop_ims, crop_cdms, bounds)):
        axes[0, i].imshow(im, cmap=cmap, vmin=im.min(),
                          vmax=im.min() + (im.max() - im.min()) * 0.1)
        axes[0, i].plot(cdm[0], cdm[1], '1', color='red')
        axes[0, i].plot([x0, x1 - 1, x1 - 1, x0, x0], [y0, y0, y1 - 1, y1 - 1, y0], '--y')
        axes[0, i].set_title(f"Crop limits, image #{i}{_z_label(zs, i)}")

        half_w, half_h = (x1 - x0) // 2, (y1 - y0) // 2
        xc, yc = crop_cdm
        axes[1, i].imshow(crop_im, cmap=cmap, vmin=crop_im.min())
        axes[1, i].plot(xc, yc, '1r')
        axes[1, i].plot([xc - half_w, xc + half_w, xc + half_w, xc - half_w, xc - half_w],
                        [yc - half_h, yc - half_h, yc + half_h, yc + half_h, yc - half_h], '--y')
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- #
# Retrieval
# ---------------------------------------------------------------------- #

def plot_comp_I(ucalc, umeas, title: str = '', iter: int = None, z: float = None,
                cmap: str = 'gist_ncar'):
    """
    Measured and calculated intensities, calculated phase (unwrapped and
    wrapped) at one plane, during the retrieval.

    Args:
        ucalc: calculated complex field
        umeas: measured amplitude (or field)
        title, iter, z: figure title, iteration number, plane position (m)
    """
    fig, ax = plt.subplots(2, 2, figsize=(13, 7))
    panels = ((ax[0, 0], 'Measured intensity', np.abs(umeas) ** 2, cmap),
              (ax[0, 1], 'Calculated intensity', np.abs(ucalc) ** 2, cmap),
              (ax[1, 0], 'Calculated phase (unwrapped)', unwrap_phase(np.angle(ucalc)), 'twilight'),
              (ax[1, 1], 'Calculated phase', np.angle(ucalc), 'twilight'))
    for a, name, data, cm in panels:
        a.set_title(name)
        _colorbar(a, a.imshow(data, cmap=cm))

    parts = [title] if title else []
    if z is not None:
        parts.append(f"z = {z * 1e3:.1f} mm")
    if iter is not None:
        parts.append(f"iteration {iter}")
    fig.suptitle(", ".join(parts))
    fig.tight_layout()
    return fig


def plot_convergence(errors, ax=None):
    """RMSE at the reference plane versus iteration (log scale)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(np.ravel(errors), marker='x', mfc='red', mec='red', ls='--', lw=1, color='black')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('RMSE (reference plane)')
    ax.figure.tight_layout()
    return ax


def plot_field(field, pixel_size=None, title: str = None, unwrap: bool = True,
               cmap='gist_ncar'):
    """
    Amplitude, intensity and phase of a complex field. Axes in mm if
    pixel_size (m) is given, in pixels otherwise.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    phase = unwrap_phase(np.angle(field)) if unwrap else np.angle(field)
    extent = None
    if pixel_size is not None:
        h, w = field.shape
        extent = (0, w * pixel_size * 1e3, h * pixel_size * 1e3, 0)
    for ax, name, data, cm in ((axes[0], "Amplitude", np.abs(field), cmap),
                               (axes[1], "Intensity", np.abs(field) ** 2, cmap),
                               (axes[2], "Phase (rad)" + (", unwrapped" if unwrap else ""),
                                phase, 'twilight')):
        ax.set_title(name)
        _colorbar(ax, ax.imshow(data, cmap=cm, extent=extent))
        ax.set_xlabel("x (mm)" if extent else "x (px)")
    axes[0].set_ylabel("y (mm)" if extent else "y (px)")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_caustic(z, dx, dy, z_meas=None, dx_meas=None, dy_meas=None,
                 wavelength=343e-9, z_planes=None):
    """
    Caustic of the retrieved field (red, circles, dashed fit) and of the
    measurement (blue, squares, solid fit), D4σ along x and y, with the
    fitted M² (fit_caustic). All distances and diameters in m.

    Args:
        z, dx, dy: planes and diameters of the retrieved field
        z_meas, dx_meas, dy_meas: measured planes and diameters (optional)
        z_planes: retrieval planes, drawn as vertical dotted lines (optional)

    Returns:
        fig, fits: fits = {'Ret': fit_caustic(...), 'Meas': ...}
    """
    curves = [("Ret", z, dx, dy, 'red', dict(marker='o', ms=4, zorder=4), dict(ls='--', lw=1, zorder=3))]
    if z_meas is not None:
        curves.insert(0, ("Meas", z_meas, dx_meas, dy_meas, 'tab:blue',
                          dict(marker='s', ms=6, zorder=2), dict(ls='-', lw=1.6, zorder=2)))
    fits = {name: fit_caustic(zz, ddx, ddy, wavelength) for name, zz, ddx, ddy, *_ in curves}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, a in zip(axes, ("x", "y")):
        handles = []
        for name, zz, ddx, ddy, color, mk, ln in curves:
            d = ddx if a == "x" else ddy
            p = fits[name]
            z_fit = np.linspace(np.min(zz), np.max(zz), 300)
            ax.plot(np.asarray(zz) * 1e3, np.asarray(d) * 1e6, ls='none', color=color, **mk)
            ax.plot(z_fit * 1e3, caustic_curve(z_fit, p[f"d0{a}"], p[f"z0{a}"], p[f"theta{a}"]) * 1e6,
                    color=color, **ln)
            handles.append((Line2D([], [], color=color, marker=mk['marker'], ms=mk['ms'],
                                   ls=ln['ls'], lw=ln['lw']),
                            f"{name}: $M^2_{a}$ = {p[f'M2{a}']:.3f}"))
        if z_planes is not None:
            for zp in z_planes:
                ax.axvline(zp * 1e3, color='gray', ls=':', lw=0.8)
            handles.append((Line2D([], [], color='gray', ls=':', lw=0.8), "Retrieval planes"))
        ax.set_xlabel("z (mm)")
        ax.set_ylabel(f"D4σ$_{a}$ diameter (µm)")
        ax.legend(*zip(*handles))
    fig.tight_layout()
    return fig, fits
