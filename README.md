# LAPIM: LAser Phase retrieval from Intensity Measurements

LAPIM retrieves the complex field of a laser beam from intensity images
recorded at several planes along its caustic. The retrieved field is then
propagated to every measured plane, and its beam size and M² are compared to
the measurement.

This is the code used in:

> J. Herbuvaux et al., *TITLE*, JOURNAL (YEAR). DOI: XXXX

The data were recorded at 343 nm through a CaF₂ sample. The repository
contains two data sets: one through a clean area and one through an area with
laser-induced contamination.

The implementation builds on the following works:

- L. J. Allen and M. P. Oxley, "Phase retrieval from series of images obtained
  by defocus variation," *Optics Communications* **199**, 65–75 (2001).
  [doi:10.1016/S0030-4018(01)01556-5](https://doi.org/10.1016/S0030-4018(01)01556-5)
- Y. Rivenson, Y. Wu, H. Wang, Y. Zhang, A. Feizi and A. Ozcan,
  "Sparsity-based multi-height phase recovery in holographic microscopy,"
  *Scientific Reports* **6**, 37862 (2016).
  [doi:10.1038/srep37862](https://doi.org/10.1038/srep37862)
- L. Lehmann, S. Bouillet, C. Leymarie, C. Ameil-Schuh, M. Benoit and
  C. Rouyer, "Characterization of surface defects using a phase retrieval
  technique in a high-power laser system," *Applied Optics* **61**, 1545
  (2022). [doi:10.1364/AO.451256](https://doi.org/10.1364/AO.451256)

## Method

- **Algorithm.** A multi-plane Gerchberg–Saxton algorithm with angular
  spectrum propagation.
- **Iteration.** Each iteration propagates the field from the reference plane
  through all the retrieval planes and back. On the forward pass, the measured
  amplitude is imposed at every plane. On the way back, it is imposed only at
  the reference plane.
- **Constraints at the reference plane.** A relaxed (soft) support of radius
  1.3 × r99 is applied (β = 0.2), together with a Tukey apodisation (α = 0.3).
- **Convergence.** Monitored by the RMSE between the computed and measured
  intensities at the reference plane.
- **Preprocessing** (`ImageSetTrt`), in this order:
  1. background subtraction (mean over a frame at the image edges);
  2. Gaussian filter (σ = 2 px), then negative values set to zero;
  3. normalisation;
  4. crop to 4 × the beam width;
  5. Tukey window;
  6. zero-padding (250 px, rounded up to a fast FFT size), centred on the
     centroid.
- **Beam size.** Beam sizes are second-moment (D4σ) diameters along the fixed
  x and y axes of the sensor, computed following ISO 11146. The M² is fitted
  separately on each axis, and M²eff = √(M²x · M²y).

## Repository content

| File | Content |
|---|---|
| `phase_retriever.py` | `PhaseRetriever` class (retrieval, saving) and `configure_logging` |
| `image_functions.py` | Image preprocessing: background removal, crop, padding (`ImageSetTrt`) |
| `utils.py` | Propagation, support, D4σ beam size, caustic and M² fit |
| `plot_functions.py` | Preprocessing checks, convergence, retrieved field, caustic |
| `example.py` | Retrieval and M² comparison on both data sets |
| `Datas/reference_set/` | Images through the clean area |
| `Datas/perturbed_set/` | Images through the contaminated area |

### Data format

Each data set has 19 images, stored as single-layer float32 TIFF files named
after their distance to the reference plane: `z_<distance in mm>mm.tiff`.
`z_0000.000mm.tiff` is the reference plane, where the field is retrieved. The
pixel size is 3.75 µm.

`measured_diameters.csv` gives the measured D4σ diameters for each plane
(`z_mm, d4sigma_x_um, d4sigma_y_um`). The `used_in_fit` column flags the
planes used in the M² fit. The example does not read this file: it recomputes
the diameters from the images.

## Installation

Developed and tested with Python 3.14 (it should also work with Python ≥ 3.11):

```bash
pip install -r requirements.txt
```

## Usage

```bash
python example.py
```

For each data set, the script:

1. retrieves the field at the reference plane from 6 planes: the reference
   plane, both ends of the caustic and three planes around the waist;
2. propagates the field to all the measured planes;
3. compares the retrieved caustic and M² to the ones measured on the raw
   images.

Results are written to `Results/<data set>/`:

- `<data set>_field.npy`: the retrieved complex field at the reference plane;
- `_calc_intensity.npy`, `_errors.npy`: the computed intensity and the RMSE per
  iteration;
- `_config.json`: every preprocessing and retrieval parameter, the date and the
  duration;
- `caustic.png`, `convergence.png`, `reference_plane.png` (measured and calculated intensities, retrieved phase): the figures.

The log goes to `Results/retrieve.log`.

All parameters are set in the `IMAGE_CONFIG` and `RETRIEVE_CONFIG`
dictionaries at the top of `example.py`. The random initial phase uses a fixed
seed (`seed = 0`), so results are reproducible. Set `seed` to `None` to draw a
new initial phase at each run.

### Using your own data

```python
from phase_retriever import PhaseRetriever, configure_logging

configure_logging("retrieve.log")  # not necessary 
retriever = PhaseRetriever(raw_images=images,       # list of 2D arrays, reference plane first
                           distances=distances,     # m, from the reference plane (first = 0)
                           wavelength=343e-9, pixel_size=3.75e-6,
                           iterations=81, seed=0)
field, errors = retriever.retrieve_phase()
retriever.save("Results/my_beam")
```

## Citation

If you use this code, please cite the article above.

## License

MIT License, see `LICENSE`. The data sets in `Datas/` are distributed under the same terms.
