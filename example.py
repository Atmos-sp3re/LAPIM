"""
Example: multi-plane phase retrieval on the two data sets, reference (clean
area of the sample) and perturbed (contaminated area).

The images are single-layer float32 TIFF files named after their distance to
the reference plane, z_<distance in mm>mm.tiff (z_0000.000mm.tiff is the
reference plane, where the field is retrieved).

Outputs (in Results/<data set>/):
    <data set>_field.npy, _calc_intensity.npy, _errors.npy, _config.json
    caustic.png, convergence.png, field.png
and Results/retrieve.log.
"""
import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from phase_retriever import PhaseRetriever, configure_logging
from plot_functions import plot_caustic, plot_convergence, plot_field
from utils import caustic, get_beam_size

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE / "Datas"
OUTPUT_ROOT = HERE / "Results"

# Planes used for the retrieval (mm from the reference plane), per data set:
# the reference plane, both ends of the caustic and three planes around the waist
DATASETS = {
    "reference_set": [0.0, 354.300, 569.861, 587.055, 621.444, 899.014],
    "perturbed_set": [0.0, 354.300, 543.170, 572.955, 588.492, 946.300],
}

# Image preprocessing (image_functions.ImageSetTrt)
IMAGE_CONFIG = {
    "remove_background": True,  # subtract the background (mean over an edge frame)
    "crop_before": 2,           # border removed before processing (px)
    "bkg_zone": 30,             # width of the edge frame used for the background (px)
    "gaussian_filter": True,    # Gaussian filter after background subtraction
    "sigma": 2,                 # width of the Gaussian filter (px)
    "threshold": None,          # beam width threshold (None: 10 % of the std of the 1st image)
    "min_width": 20,            # minimum beam width (px)
    "square_size": 4,           # crop size / beam width
    "pad_frame": 250,           # zero-padding on each side of the largest crop (px)
    "pad_apodisation": True,    # Tukey window before padding
    "pad_alpha": 0.3,           # Tukey parameter
}

# Propagation and retrieval (phase_retriever.PhaseRetriever)
RETRIEVE_CONFIG = {
    "wavelength": 343e-9,       # m
    "pixel_size": 3.75e-6,      # m
    "iterations": 31,
    "init_phase_type": "random",  # initial phase at the reference plane
    "seed": 0,                  # seed of the random phase (None: different at each run)
    "support_mode": "soft",     # relaxed support constraint at the reference plane
    "support_beta": 0.2,        # relaxation parameter β
    "support_eps": 1e-6,        # residual amplitude outside the support (fraction of max)
    "support_margin": 1.3,      # support radius / r99
    "apodisation": True,        # Tukey window at the reference plane
    "kind_apodisation": "tukey",
    "alpha_apo": 0.3,
    "normalize_energy": False,
}


def load_dataset(folder):
    """{distance (m): image} for the z_<mm>mm.tiff files of folder."""
    images = {}
    for path in sorted(Path(folder).glob("z_*mm.tiff")):
        z_mm = float(re.search(r"z_([\d.]+)mm", path.name).group(1))
        images[z_mm * 1e-3] = tifffile.imread(path).astype(float)
    return images


def run_retrieval(name):
    """Retrieval, saving and validation plots for the data set `name`; returns the field."""
    log = logging.getLogger("example")
    output_dir = OUTPUT_ROOT / name
    images = load_dataset(DATA_ROOT / name)
    distances = [z * 1e-3 for z in DATASETS[name]]
    raw_images = [images[min(images, key=lambda zk: abs(zk - z))] for z in distances]

    # ---------- retrieval ----------
    retriever = PhaseRetriever(raw_images=raw_images, distances=distances,
                               image_config=IMAGE_CONFIG, file_name=name, **RETRIEVE_CONFIG)
    field, errors = retriever.retrieve_phase()
    retriever.save(output_dir)

    # ---------- validation: caustic on all the measured planes ----------
    # measured D4σ on the raw images, retrieved D4σ on the propagated field
    px = RETRIEVE_CONFIG["pixel_size"]
    z_meas = np.array([z for z in images if z > 0])
    d_meas = np.array([get_beam_size(images[z])[2:] for z in z_meas]).T * px
    dx_ret, dy_ret = caustic(field, z_meas, wavelength=RETRIEVE_CONFIG["wavelength"],
                             pixel_size=px)
    fig, fits = plot_caustic(z_meas, dx_ret, dy_ret, z_meas, d_meas[0], d_meas[1],
                             wavelength=RETRIEVE_CONFIG["wavelength"], z_planes=distances[1:])
    fig.savefig(output_dir / "caustic.png", dpi=150)

    for label, p in fits.items():
        log.info(f"{name} {label}: M²x = {p['M2x']:.3f} ± {p['s_M2x']:.3f}, "
                 f"M²y = {p['M2y']:.3f} ± {p['s_M2y']:.3f}, M²eff = {p['M2eff']:.3f}, "
                 f"z0 = ({p['z0x'] * 1e3:.1f}, {p['z0y'] * 1e3:.1f}) mm, "
                 f"d0 = ({p['d0x'] * 1e6:.1f}, {p['d0y'] * 1e6:.1f}) µm")

    plot_convergence(errors).figure.savefig(output_dir / "convergence.png", dpi=150)
    plot_field(field, pixel_size=px, title=f"Retrieved field at the reference plane, {name}").savefig(
        output_dir / "field.png", dpi=150)
    return field


if __name__ == "__main__":
    configure_logging(OUTPUT_ROOT / "retrieve.log")
    for dataset in DATASETS:
        run_retrieval(dataset)
    plt.show()
