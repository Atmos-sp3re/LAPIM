"""
Preprocessing of the measured intensity images before phase retrieval.

For each image (BackgroundRemover):
    1. a border of crop_before pixels is removed
    2. the background, mean intensity over a frame of bkg_zone pixels along
       the edges, is subtracted
    3. Gaussian filter (sigma, in pixels), then negative pixels set to zero
    4. normalisation to unit integrated power
Then for the whole set (Resizer):
    5. each image is cropped around its centroid (crop size = square_size x
       beam width at the threshold, threshold computed on the first image)
    6. Tukey apodisation of the cropped image
    7. zero-padding on a common grid (largest crop + 2 x pad_frame, rounded
       to a fast FFT size), the centroid placed at the grid centre; the
       padded region is filled with a small constant level (1e-6 of the
       image maximum)
    8. normalisation to unit integrated power
ImageSetTrt chains both steps.
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len
from scipy.ndimage import gaussian_filter
from scipy.signal.windows import tukey

from plot_functions import plot_bkg_images, plot_crop_lims
from utils import get_centroid_xy, norm_sum

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """
    Background removal, filtering and normalisation of each image (steps 1-4).

    Args:
        image_set: list of 2D images
        crop_before: border (pixels) removed before processing
        bkg_zone: width (pixels) of the edge frame used for the background
        gaussian_filter, sigma: Gaussian filter and its width (pixels)
        background_removal: if False, images are returned unchanged
        distances: plane positions (m), only used for the plot titles
        plot: plot raw and processed images
    """

    def __init__(self,
                 image_set,
                 crop_before=2,
                 bkg_zone=30,
                 gaussian_filter=True,
                 sigma=2,
                 background_removal=True,
                 distances=None,
                 plot=False):
        self.image_set = image_set
        self.crop_before = crop_before
        self.bkg_zone = bkg_zone
        self.gfilter = gaussian_filter
        self.sigma = sigma
        self.background_removal = background_removal
        self.plot = plot
        self.zs = distances

    def _remove_bkg_frame(self, image):
        """Steps 1-4 for one image; returns the image and its centroid."""
        h, w = image.shape
        c = self.crop_before
        image_trt = image[c:h - c, c:w - c]

        # edge frame of the background; its limits are computed with the
        # size before the border crop (frame 2 x crop_before pixels thinner
        # at the bottom and on the right)
        mask = np.ones(image_trt.shape, dtype=bool)
        mask[self.bkg_zone:h - self.bkg_zone, self.bkg_zone:w - self.bkg_zone] = False
        image_bkg = image_trt - image_trt[mask].mean()

        if self.gfilter:
            image_bkg = gaussian_filter(image_bkg, sigma=self.sigma)
        image_bkg[image_bkg < 0] = 0

        image_bkg = image_bkg / np.sum(image_bkg)
        return image_bkg, get_centroid_xy(image_bkg)

    def process_bkg_image_set(self):
        """Processed images and their centroids."""
        images_trt, cdms = [], []
        for im in self.image_set:
            if self.background_removal:
                image, cdm = self._remove_bkg_frame(im)
            else:
                image, cdm = im, get_centroid_xy(im)
            images_trt.append(image)
            cdms.append(cdm)

        if self.plot:
            plot_bkg_images(ims_raw=self.image_set,
                            cdms_raw=[get_centroid_xy(im) for im in self.image_set],
                            ims_bck=images_trt, bck_cdms=cdms, z=self.zs)
            plt.show()
        return images_trt, cdms


class Resizer:
    """
    Cropping, apodisation, padding and centring on a common grid (steps 5-8).

    Args:
        image_set: list of 2D images (background already removed)
        threshold: intensity threshold defining the beam width; if None,
            10 % of the standard deviation of the first image
        min_width: minimum beam width (pixels)
        square_size: crop size / beam width
        pad_frame: padding (pixels) added on each side of the largest crop
        pad_apodisation, pad_alpha: Tukey window applied before padding
        distances: plane positions (m), only used for the plot titles
        plot: plot the crop limits and the padded images
    """

    def __init__(self,
                 image_set,
                 threshold=None,
                 min_width=20,
                 square_size=4,
                 pad_frame=250,
                 pad_apodisation: bool = True,
                 pad_alpha: float = 0.3,
                 distances=None,
                 plot=False):
        self.image_set = image_set
        self.min_width = min_width
        self.square_size = square_size
        self.pad_frame = pad_frame
        self.pad_apodisation = pad_apodisation
        self.pad_alpha = pad_alpha
        self.plot = plot
        self.zs = distances
        # same threshold for every image, computed on the first one
        self.thr = threshold if threshold is not None else 10 * np.std(image_set[0]) / 100
        self._cached_crop = None

    def _get_crop_lims(self, image):
        """Square crop [x0, x1, y0, y1] around the centroid of the image."""
        h, w = image.shape
        xc, yc = get_centroid_xy(image)
        xc, yc = int(xc), int(yc)

        # beam limits along the row and the column through the centroid
        row = image[yc, :]
        left = xc
        while left > 0 and row[left] > self.thr:
            left -= 1
        right = xc
        while right < w - 1 and row[right] > self.thr:
            right += 1

        col = image[:, xc]
        top = yc
        while top > 0 and col[top] > self.thr:
            top -= 1
        bottom = yc
        while bottom < h - 1 and col[bottom] > self.thr:
            bottom += 1

        base_size = max(right - left, bottom - top, self.min_width)
        half = int(base_size * self.square_size) // 2

        return [max(0, xc - half), min(w, xc + half),
                max(0, yc - half), min(h, yc + half)], (xc, yc)

    def _crop_image_set(self):
        """Cropped images, their centroids and crop limits (computed once)."""
        if self._cached_crop is None:
            crop_ims, crop_cdms, bounds = [], [], []
            for im in self.image_set:
                bound, _ = self._get_crop_lims(im)
                x0, x1, y0, y1 = bound
                crop_im = im[y0:y1, x0:x1]
                crop_ims.append(crop_im)
                crop_cdms.append(get_centroid_xy(crop_im))
                bounds.append(bound)
            self._cached_crop = (crop_ims, crop_cdms, bounds)
        return self._cached_crop

    def _ref_image_size(self):
        """Largest cropped image (by area): it sets the grid size."""
        crop_ims, _, _ = self._crop_image_set()
        return crop_ims[int(np.argmax([im.shape[0] * im.shape[1] for im in crop_ims]))]

    def _apodized_before_pad(self, image):
        """Tukey window on the cropped image."""
        if not self.pad_apodisation:
            return image
        n_rows, n_cols = image.shape
        return image * np.outer(tukey(n_rows, alpha=self.pad_alpha),
                                tukey(n_cols, alpha=self.pad_alpha))

    def _pad_image(self, image):
        """Pad on the common grid, centroid at the centre, unit power."""
        ref_h, ref_w = self._ref_image_size().shape
        H = next_fast_len(ref_h + 2 * self.pad_frame)
        W = next_fast_len(ref_w + 2 * self.pad_frame)

        apo_im = self._apodized_before_pad(image)
        im_h, im_w = apo_im.shape
        old_xc, old_yc = get_centroid_xy(apo_im)

        # position of the image in the grid, clipped to the grid
        top = int(round(H / 2 - old_yc))
        left = int(round(W / 2 - old_xc))
        src_y0, src_x0 = max(0, -top), max(0, -left)
        dst_y0, dst_x0 = max(0, top), max(0, left)
        h_copy = min(im_h - src_y0, H - dst_y0)
        w_copy = min(im_w - src_x0, W - dst_x0)
        if h_copy <= 0 or w_copy <= 0:
            raise ValueError("Image outside the padded grid: increase pad_frame "
                             "or square_size")

        # padded region: constant level 1e-6 x image maximum (at least 1e-12),
        # plus 1e-6 of this level, added again after normalisation
        level = max(1e-6 * np.max(apo_im), 1e-12)
        offset = 1e-6 * level
        padded_im = np.full((H, W), fill_value=level, dtype=float)
        padded_im += offset
        padded_im[dst_y0:dst_y0 + h_copy, dst_x0:dst_x0 + w_copy] = \
            apo_im[src_y0:src_y0 + h_copy, src_x0:src_x0 + w_copy]

        padded_im = norm_sum(padded_im)
        padded_im += offset
        return padded_im, get_centroid_xy(padded_im)

    def process_resize_image_set(self):
        """Padded images and their centroids."""
        crop_ims, _, bounds = self._crop_image_set()
        res_ims, res_cdms = [], []
        for im in crop_ims:
            padded_im, padded_cdm = self._pad_image(im)
            res_ims.append(padded_im)
            res_cdms.append(padded_cdm)

        logger.info(f"{len(res_ims)} images preprocessed, grid {res_ims[0].shape}")
        for i, (im, cdm) in enumerate(zip(res_ims, res_cdms)):
            logger.debug(f"Image #{i}: shape {im.shape}, centroid {cdm}, sum {np.sum(im):.4f}")

        if self.plot:
            plot_crop_lims(ims=self.image_set,
                           cdms=[get_centroid_xy(im) for im in self.image_set],
                           crop_ims=res_ims, crop_cdms=res_cdms,
                           bounds=bounds, zs=self.zs)
            plt.show()
        return res_ims, res_cdms


class ImageSetTrt:
    """
    Full preprocessing of an image set (BackgroundRemover then Resizer).
    The defaults are the parameters used in the article.

    Args:
        image_set: list of 2D intensity images (converted to float)
        remove_background, bkg_zone, crop_before, gaussian_filter, sigma:
            see BackgroundRemover
        threshold, min_width, square_size, pad_frame, pad_apodisation,
            pad_alpha: see Resizer
        plot_bkg, plot_resize: plots of the two steps
        distances: plane positions (m), only used for the plot titles

    config: dict of all the preprocessing parameters (saved with the results)
    """

    def __init__(self,
                 image_set,
                 remove_background=True,
                 bkg_zone=30,
                 crop_before=2,
                 gaussian_filter=True,
                 sigma=2,
                 threshold=None,
                 min_width=20,
                 square_size=4,
                 pad_frame=250,
                 pad_apodisation=True,
                 pad_alpha=0.3,
                 plot_bkg=False,
                 plot_resize=False,
                 distances=None):
        self.image_set = image_set
        self.distances = distances
        self.bkg_dict = dict(crop_before=crop_before, bkg_zone=bkg_zone,
                             gaussian_filter=gaussian_filter, sigma=sigma,
                             background_removal=remove_background)
        self.res_dict = dict(threshold=threshold, min_width=min_width,
                             square_size=square_size, pad_frame=pad_frame,
                             pad_apodisation=pad_apodisation, pad_alpha=pad_alpha)
        self.plot_bkg = plot_bkg
        self.plot_resize = plot_resize

    @property
    def config(self) -> dict:
        """Preprocessing parameters, with the ImageSetTrt argument names."""
        bkg = {**self.bkg_dict}
        bkg["remove_background"] = bkg.pop("background_removal")
        return {**bkg, **self.res_dict,
                "plot_bkg": self.plot_bkg, "plot_resize": self.plot_resize}

    def _check_images(self):
        """2D arrays, converted to float."""
        images = []
        for i, im in enumerate(self.image_set):
            im = np.asarray(im, dtype=float)
            if im.ndim != 2:
                raise ValueError(f"Image #{i} must be 2D, shape = {im.shape}")
            images.append(im)
        return images

    def process_image_set(self):
        """Preprocessed images (common grid, unit power) and their centroids."""
        images = self._check_images()
        bkg_ims, _ = BackgroundRemover(images, distances=self.distances,
                                       plot=self.plot_bkg,
                                       **self.bkg_dict).process_bkg_image_set()
        return Resizer(bkg_ims, distances=self.distances, plot=self.plot_resize,
                       **self.res_dict).process_resize_image_set()
