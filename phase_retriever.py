"""
Multi-plane phase retrieval (Gerchberg-Saxton type) from intensity
measurements acquired in several planes along the propagation axis.

The field is estimated in the reference plane z_0 (first distance). One
iteration:
    forward  : z_0 -> z_1 -> ... -> z_N-1, the measured amplitude is imposed
               at each plane (the calculated phase is kept)
    backward : z_N-1 -> z_0, the measured amplitude is imposed at z_0 only
    then, at z_0 : relaxed ("soft") support constraint and apodisation
Propagation uses the angular spectrum method.
"""
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.fft import fft2, fftfreq, ifft2
from scipy.signal.windows import tukey
from tqdm import tqdm

from image_functions import ImageSetTrt
from plot_functions import plot_comp_I
from utils import calc_r99, circular_mask, err_rmse

logger = logging.getLogger(__name__)


def configure_logging(log_file=None, level=logging.INFO):
    """
    Print the logs in the console and, if log_file is given, also append
    them to this file.
    """
    fmt = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    root = logging.getLogger()
    root.setLevel(level)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


class PhaseRetriever:
    """
    Retrieve the complex field in the reference plane from intensity images.

    Args:
        raw_images: list of 2D intensity images, one per plane
        distances: plane positions (m), strictly increasing; the first one is
            the reference plane z_0 where the field is retrieved
        wavelength, pixel_size: in m
        iterations: number of iterations
        init_phase_type: initial phase at z_0, 'random' or 'zeros'
        seed: seed of the random initial phase (same seed -> same result)
        support_mode: support constraint at z_0, 'soft', 'hard' or 'none'
        support_beta: relaxation parameter of the soft support, 0 < β <= 1
        support_eps: residual amplitude outside the support, as a fraction of
            the maximum amplitude (zero phase outside the support)
        support_margin: support radius = support_margin x r99 of the measured
            intensity at z_0 (r99: radius containing 99 % of the energy)
        apodisation, kind_apodisation, alpha_apo: Tukey window applied at
            z_0 after the support constraint
        normalize_energy: normalise the field energy to 1 at each iteration
        image_config: preprocessing parameters passed to
            image_functions.ImageSetTrt (its defaults are the article values)
        plot, plot_stamp: plot measured / calculated intensities at z_0
            every plot_stamp iterations
        file_name: default name of the saved files

    After retrieve_phase():
        ret_field: complex field at z_0 (last iteration, after constraints)
        last_calc_intensity: |u_calc|² at z_0, last iteration, before the
            amplitude constraint
        errors: intensity RMSE at z_0 for each iteration
    """

    def __init__(self,
                 raw_images,
                 distances,
                 wavelength: float = 343e-9,
                 pixel_size: float = 3.75e-6,
                 iterations: int = 30,
                 init_phase_type: Literal['random', 'zeros'] = 'random',
                 seed: Optional[int] = None,
                 support_mode: Literal['soft', 'hard', 'none'] = 'soft',
                 support_beta: float = 0.2,
                 support_eps: float = 1e-6,
                 support_margin: float = 1.3,
                 apodisation: bool = True,
                 kind_apodisation: Literal['tukey'] = 'tukey',
                 alpha_apo: float = 0.3,
                 normalize_energy: bool = False,
                 image_config: Optional[Dict] = None,
                 plot: bool = False,
                 plot_stamp: Optional[int] = None,
                 file_name: str = 'ret_field'):

        if raw_images is None or len(raw_images) < 2:
            raise ValueError("At least 2 images are required")
        if distances is None or len(distances) != len(raw_images):
            raise ValueError(f"One distance per image is required "
                             f"({len(raw_images)} images)")
        if np.any(np.diff(distances) <= 0):
            raise ValueError("distances must be strictly increasing, the first "
                             "one being the reference plane")
        if init_phase_type not in ('random', 'zeros'):
            raise ValueError(f"Unknown init_phase_type: {init_phase_type!r}")
        support_mode = str(support_mode).lower()
        if support_mode not in ('soft', 'hard', 'none'):
            raise ValueError(f"Unknown support_mode: {support_mode!r}")
        if not 0 < support_beta <= 1:
            raise ValueError("support_beta must be in ]0, 1]")
        if kind_apodisation != 'tukey':
            raise ValueError(f"Unknown kind_apodisation: {kind_apodisation!r}")

        self.raw_images = raw_images
        self.zs = distances
        self.lbd = wavelength
        self.px = pixel_size
        self.iters = iterations
        self.init_phase_type = init_phase_type
        self.seed = seed
        # RandomState(seed) gives the same sequence as np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        self.support_mode = support_mode
        self.support_beta = support_beta
        self.support_eps = support_eps
        self.support_margin = support_margin
        self.apo = apodisation
        self.kind_apo = kind_apodisation
        self.alpha_apo = alpha_apo
        self.normalize_energy = normalize_energy
        self.image_config = image_config or {}
        self.plot = plot
        self.plot_stamp = plot_stamp
        self.fname = file_name

        # preprocessing, filled by _preprocess_image_set()
        self.ims_processed = None
        self.cdms_processed = None
        self.process_config = None
        self._apo_window = None

        # results, filled by retrieve_phase()
        self.ret_field = None
        self.last_calc_intensity = None
        self.errors = None
        self.duration = None
        self.date = None

    # ------------------------------------------------------------------ #
    # Preprocessing and initialisation
    # ------------------------------------------------------------------ #

    def _preprocess_image_set(self):
        """Background removal, filtering, centring and padding (ImageSetTrt)."""
        im_processor = ImageSetTrt(image_set=self.raw_images, distances=self.zs,
                                   **self.image_config)
        # configuration actually applied (defaults included), for save()
        self.process_config = im_processor.config
        self.ims_processed, self.cdms_processed = im_processor.process_image_set()

    def transfer_functions(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Angular spectrum transfer functions between consecutive planes."""
        h, w = self.ims_processed[-1].shape
        nuX, nuY = np.meshgrid(fftfreq(w, self.px), fftfreq(h, self.px))

        k = 2 * np.pi / self.lbd
        radicand = 1 - (self.lbd * nuX) ** 2 - (self.lbd * nuY) ** 2
        gamma = np.sqrt(np.maximum(0.0, radicand))
        mask = radicand > 0          # evanescent waves removed

        Hs_fwd, Hs_bck = [], []
        for dz in np.abs(np.diff(self.zs)):
            Hs_fwd.append(np.exp(1j * k * dz * gamma) * mask)
            Hs_bck.append(np.exp(-1j * k * dz * gamma) * mask)
        return Hs_fwd, Hs_bck

    def _init_field(self) -> np.ndarray:
        """Measured amplitude at z_0 with a random or zero initial phase."""
        amp0 = np.sqrt(self.ims_processed[0])
        if self.init_phase_type == 'random':
            logger.info("Initial phase: random")
            phase0 = 2 * np.pi * self.rng.random_sample(amp0.shape)
        else:
            logger.info("Initial phase: zeros")
            phase0 = np.zeros(amp0.shape)
        return amp0 * np.exp(1j * phase0)

    def _measured_amplitudes(self) -> List[np.ndarray]:
        """sqrt of the processed intensities (small offset against zeros)."""
        return [np.sqrt(im + 1e-12) for im in self.ims_processed]

    # ------------------------------------------------------------------ #
    # Constraints at the reference plane
    # ------------------------------------------------------------------ #

    def _support_target(self, field, support_mask):
        """Field kept inside the support, eps amplitude and zero phase outside."""
        A = np.abs(field)
        phi = np.angle(field)
        eps_A = self.support_eps * (np.max(A) + 1e-18)
        A_t = np.where(support_mask, A, eps_A)
        phi_t = np.where(support_mask, phi, 0.0)
        return A_t * np.exp(1j * phi_t)

    def _apply_support(self, field, support_mask):
        """'hard': field = target ; 'soft': field + β (target - field)."""
        if self.support_mode == 'none':
            return field
        target = self._support_target(field, support_mask)
        if self.support_mode == 'hard':
            return target
        return field + self.support_beta * (target - field)

    def _apodisation(self, field):
        """Tukey window applied to the field."""
        if not self.apo:
            return field
        n_rows, n_cols = field.shape
        if self._apo_window is None or self._apo_window.shape != field.shape:
            self._apo_window = np.outer(tukey(n_rows, alpha=self.alpha_apo),
                                        tukey(n_cols, alpha=self.alpha_apo))
        return field * self._apo_window

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def retrieve_phase(self):
        """
        Run the retrieval.

        Returns:
            ret_field: complex field at the reference plane
            errors: list of the intensity RMSE at z_0, one per iteration
        """
        t_start = time.time()
        self.date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"Retrieval: {len(self.zs)} planes, "
                    f"z = {np.round(np.asarray(self.zs) * 1e3, 3).tolist()} mm, "
                    f"{self.iters} iterations, seed = {self.seed}")
        if self.ims_processed is None:
            self._preprocess_image_set()

        if self.plot_stamp is None:
            self.plot_stamp = max(1, self.iters // 5)
        elif self.plot_stamp > self.iters:
            self.plot_stamp = 1 if self.iters == 1 else self.iters

        Hs_fwd, Hs_bck = self.transfer_functions()
        n_steps = len(self.zs) - 1
        meas_amps = self._measured_amplitudes()
        u_ref = self._init_field()

        # support at z_0: disc of radius margin x r99 around the centroid
        I0 = self.ims_processed[0]
        cdm0 = self.cdms_processed[0]
        r99 = int(calc_r99(intensity=I0, cdm=cdm0))
        support_mask = circular_mask(I0, radius=r99, cdm=cdm0,
                                     margin=self.support_margin)

        error_list = []
        for i in tqdm(range(self.iters), desc='Retrieval'):
            u_now = u_ref.copy()

            # forward: z_0 -> z_N-1, measured amplitude imposed at each plane
            for k in range(n_steps):
                u_now = ifft2(fft2(u_now) * Hs_fwd[k])
                u_now = meas_amps[k + 1] * np.exp(1j * np.angle(u_now))

            # backward: z_N-1 -> z_0, measured amplitude imposed at z_0 only
            for k in range(n_steps):
                u_now = ifft2(fft2(u_now) * Hs_bck[-(k + 1)])
            u_calc0 = u_now.copy()
            u_now = meas_amps[0] * np.exp(1j * np.angle(u_now))

            # constraints at z_0
            if self.normalize_energy:
                u_now /= np.sqrt(np.sum(np.abs(u_now) ** 2)) + 1e-18
            u_now = self._apply_support(u_now, support_mask)
            u_ref = self._apodisation(u_now)

            # error: calculated intensity at z_0, before the amplitude constraint
            calc_intensity = np.abs(u_calc0) ** 2
            error_list.append(err_rmse(calc_intensity=calc_intensity,
                                       meas_intensity=I0))

            if self.plot and i % self.plot_stamp == 0:
                plot_comp_I(ucalc=u_calc0, umeas=meas_amps[0], iter=i, z=self.zs[0])

        self.ret_field = u_ref.copy()
        self.last_calc_intensity = calc_intensity
        self.errors = np.asarray(error_list)
        self.duration = time.time() - t_start
        minutes, seconds = divmod(int(round(self.duration)), 60)
        logger.info(f"Retrieval done in {minutes} min {seconds:02d} s, "
                    f"final RMSE at the reference plane = {self.errors[-1]:.3e}")
        return self.ret_field, error_list

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #

    def save(self, folder, name: Optional[str] = None) -> Path:
        """
        Write in folder (created if needed):
            <name>_field.npy           complex field at the reference plane
            <name>_calc_intensity.npy  calculated intensity at the reference plane
            <name>_errors.npy          RMSE at the reference plane per iteration
            <name>_config.json         all the retrieval parameters
        name defaults to file_name. Returns the path of the config.json.
        """
        if self.ret_field is None:
            raise RuntimeError("Run retrieve_phase() before save()")
        name = name or self.fname
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{name}_field.npy", self.ret_field)
        np.save(folder / f"{name}_calc_intensity.npy", self.last_calc_intensity)
        np.save(folder / f"{name}_errors.npy", self.errors)
        config_path = folder / f"{name}_config.json"
        config_path.write_text(json.dumps(self.get_config_dict(), indent=2,
                                          default=str, ensure_ascii=False),
                               encoding="utf-8")
        logger.info(f"Results saved in {folder} ({name}_*)")
        return config_path

    def get_config_dict(self) -> dict:
        """All the retrieval parameters, preprocessing and distances included."""
        return {
            "date": self.date,
            "duration_s": self.duration,
            "file_name": self.fname,
            "wavelength": self.lbd,
            "pixel_size": self.px,
            "distances_m": np.round(np.asarray(self.zs, dtype=float), 9).tolist(),
            "iterations": self.iters,
            "init_phase_type": self.init_phase_type,
            "seed": self.seed,
            "support_mode": self.support_mode,
            "support_beta": self.support_beta,
            "support_eps": self.support_eps,
            "support_margin": self.support_margin,
            "apodisation": self.apo,
            "kind_apodisation": self.kind_apo,
            "alpha_apodisation": self.alpha_apo,
            "normalize_energy": self.normalize_energy,
            "plot_stamp": self.plot_stamp,
            "image_config": self.process_config or self.image_config,
            "final_rmse": (float(self.errors[-1])
                           if self.errors is not None else None),
        }
