"""
Utility functions: centroid and normalisation, error metric, support,
angular spectrum propagation, and beam size / M² along fixed axes.

Beam sizes are second-moment (D4σ) diameters computed along the fixed x
(columns) and y (rows) axes of the sensor, following ISO 11146: background
from the image corners, rectangular integration area of mask_diameters x
the diameter around the centroid, iterated until convergence.
laserbeamsize.beam_size is not used for the diameters: it returns the
(major, minor) axes, whose orientation changes along an asymmetric caustic.
"""
import logging

import laserbeamsize as lbs
import numpy as np
from scipy.fft import fft2, fftfreq, ifft2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Centroid, normalisation, error
# ---------------------------------------------------------------------- #

def get_centroid_xy(im):
    """Beam centroid [x, y] in pixels (integers), from laserbeamsize."""
    xc, yc, _, _, _ = lbs.beam_size(im)
    return [int(xc), int(yc)]


def norm_sum(image):
    """Image normalised to unit integrated power."""
    return image / np.sum(image)


def err_rmse(calc_intensity: np.ndarray, meas_intensity: np.ndarray) -> float:
    """Root mean square error between two intensity maps."""
    return np.sqrt(np.mean((calc_intensity - meas_intensity) ** 2))


# ---------------------------------------------------------------------- #
# Support
# ---------------------------------------------------------------------- #

def calc_r99(intensity, percentile=0.99, cdm=None, nbins=600):
    """
    Radius (pixels) of the disc containing `percentile` of the energy.

    Args:
        intensity: intensity image (negative values clipped to 0)
        percentile: energy fraction (0.99 for r99)
        cdm: centre (x, y) in pixels; centroid if None
        nbins: number of radial bins
    """
    I = np.clip(intensity, 0, None).astype(np.float64)
    H, W = I.shape
    if cdm is None:
        cx, cy = get_centroid_xy(I)
    else:
        cx, cy = cdm

    yy, xx = np.ogrid[:H, :W]
    r = np.hypot(yy - cy, xx - cx)

    # cumulated energy over radial bins
    bins = np.linspace(0, r.max(), nbins + 1)
    idx = np.digitize(r.ravel(), bins)
    sums = np.bincount(idx, weights=I.ravel(), minlength=nbins + 2)
    cumE = np.cumsum(sums)[1:-1]
    cumE /= cumE[-1]

    rc = 0.5 * (bins[1:] + bins[:-1])
    return float(np.interp(percentile, cumE, rc))


def circular_mask(image, radius, cdm=None, margin=1.2):
    """Boolean disc of radius margin x radius around cdm (centroid if None)."""
    H, W = image.shape
    y, x = np.ogrid[:H, :W]
    if cdm is None:
        cx, cy = get_centroid_xy(image)
    else:
        cx, cy = cdm
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return r <= margin * radius


# ---------------------------------------------------------------------- #
# Propagation
# ---------------------------------------------------------------------- #

def H(z, field, lbd=343e-9, px=3.75e-6):
    """
    Angular spectrum transfer function over a distance z (m) for the grid of
    field. Same transfer function as
    phase_retriever.PhaseRetriever.transfer_functions, written differently
    (results equal up to the last bits).
    """
    y_res, x_res = field.shape
    nuX, nuY = np.meshgrid(fftfreq(x_res, px), fftfreq(y_res, px))
    return np.exp(2 * np.pi * 1j * np.sqrt(1 / lbd ** 2 - nuX ** 2 - nuY ** 2) * z)


def propagate(field, z, wavelength=343e-9, pixel_size=3.75e-6):
    """Field propagated over z (m, negative = backward), angular spectrum method."""
    return ifft2(fft2(field) * H(z=z, field=field, lbd=wavelength, px=pixel_size))


# ---------------------------------------------------------------------- #
# Beam size (D4σ, fixed axes) and M²
# ---------------------------------------------------------------------- #

def _moments(im: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    """Centroid and D4σ along x / y (pixels) of im in the area [y0:y1, x0:x1]."""
    sub = im[y0:y1, x0:x1]
    px, py, tot = sub.sum(axis=0), sub.sum(axis=1), sub.sum()
    xs, ys = np.arange(x0, x1), np.arange(y0, y1)
    xc, yc = (px * xs).sum() / tot, (py * ys).sum() / tot
    dx = 4 * np.sqrt(max((px * (xs - xc) ** 2).sum() / tot, 0))
    dy = 4 * np.sqrt(max((py * (ys - yc) ** 2).sum() / tot, 0))
    return xc, yc, dx, dy


def get_beam_size(image: np.ndarray, mask_diameters: float = 3, corner_fraction: float = 0.035,
            n_t: float = 3, max_iter: int = 50, tol: float = 0.05):
    """
    Second-moment (D4σ) diameters along the fixed x (columns) and y (rows) axes.

    The background, mean of the four corners (corner_fraction of the image
    size), is subtracted. First estimate on the pixels above n_t x the
    corner noise, then integration over a rectangle of mask_diameters x
    (dx, dy) centred on the centroid, iterated on the image without
    threshold (negative values kept) until dx and dy change by less than
    tol pixels.

    Returns:
        xc, yc, dx, dy in pixels
    """
    im = np.asarray(image, dtype=float)
    h, w = im.shape
    ny, nx = max(1, int(corner_fraction * h)), max(1, int(corner_fraction * w))
    corners = np.concatenate([im[:ny, :nx].ravel(), im[:ny, -nx:].ravel(),
                              im[-ny:, :nx].ravel(), im[-ny:, -nx:].ravel()])
    im = im - corners.mean()

    xc, yc, dx, dy = _moments(np.where(im > n_t * corners.std(), im, 0), 0, w, 0, h)
    for _ in range(max_iter):
        hx, hy = mask_diameters * dx / 2, mask_diameters * dy / 2
        x0, x1 = max(0, int(round(xc - hx))), min(w, int(round(xc + hx)) + 1)
        y0, y1 = max(0, int(round(yc - hy))), min(h, int(round(yc + hy)) + 1)
        xc, yc, dx_new, dy_new = _moments(im, x0, x1, y0, y1)
        converged = abs(dx_new - dx) < tol and abs(dy_new - dy) < tol
        dx, dy = dx_new, dy_new
        if converged:
            break
    else:
        logger.warning(f"get_beam_size: no convergence in {max_iter} iterations")
    return xc, yc, dx, dy


def caustic(field, distances, z_ref=0.0, wavelength=343e-9, pixel_size=3.75e-6, **beam_size_kw):
    """
    D4σ diameters (m) along x and y of field (given at z_ref) propagated to
    each of the distances (m).

    Returns:
        dx, dy: arrays of diameters in m, one value per distance
    """
    d = np.array([get_beam_size(np.abs(propagate(field, z - z_ref, wavelength, pixel_size)) ** 2,
                                **beam_size_kw)[2:] for z in distances]) * pixel_size
    d = d.reshape(-1, 2)
    return d[:, 0], d[:, 1]


def fit_caustic(z, dx, dy, wavelength=343e-9) -> dict:
    """
    Hyperbolic fit d(z)² = d0² + Θ² (z - z0)² of the caustic along x and y
    (laserbeamsize.M2_fit), all in m.

    Returns a dict with, for each axis a in ('x', 'y'): M2a, z0a, d0a, thetaa
    and their 1σ fit uncertainties s_M2a, s_z0a, s_d0a, s_thetaa; plus
    M2eff = sqrt(M2x M2y) and s_M2eff (first-order propagation).
    """
    p = {}
    for ax, d in (("x", dx), ("y", dy)):
        (d0, z0, theta, m2, _), (s_d0, s_z0, s_theta, s_m2, _), _ = \
            lbs.M2_fit(np.asarray(z), np.asarray(d), lambda0=wavelength)
        for k, v, e in zip(("M2", "z0", "d0", "theta"), (m2, z0, d0, theta),
                           (s_m2, s_z0, s_d0, s_theta)):
            p[f"{k}{ax}"], p[f"s_{k}{ax}"] = v, e
    p["M2eff"] = np.sqrt(p["M2x"] * p["M2y"])
    p["s_M2eff"] = 0.5 * p["M2eff"] * np.hypot(p["s_M2x"] / p["M2x"],
                                               p["s_M2y"] / p["M2y"])
    return p


def caustic_curve(z, d0, z0, theta):
    """Fitted diameter d(z) = sqrt(d0² + Θ² (z - z0)²)."""
    return np.sqrt(d0 ** 2 + theta ** 2 * (np.asarray(z) - z0) ** 2)
