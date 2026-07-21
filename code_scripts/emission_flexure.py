"""Wavelength-dependent flexure curves from sky-emission lines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from rv_helpers import (
    C_KMS, DEFAULT_EMISSION_LINE_DIR, _float_array, _line_wavelengths_from_catalog as _helper_line_wavelengths_from_catalog, _read_channel,
    convert_air_to_vacuum, least_squares_errors, load_emission_line_catalog, read_sky_model_at_trace,
)


def _filter_line_wavelengths_by_separation(line_wavelengths, min_separation):
    if min_separation is None:
        return np.asarray(line_wavelengths, dtype=float)
    line_wavelengths = np.asarray(line_wavelengths, dtype=float)
    if len(line_wavelengths) <= 1:
        return line_wavelengths
    order = np.argsort(line_wavelengths)
    sorted_wl = line_wavelengths[order]
    previous_sep = np.r_[np.inf, np.diff(sorted_wl)]
    next_sep = np.r_[np.diff(sorted_wl), np.inf]
    keep_sorted = np.minimum(previous_sep, next_sep) >= float(min_separation)
    keep = np.zeros(len(line_wavelengths), dtype=bool)
    keep[order] = keep_sorted
    return line_wavelengths[keep]


def _line_wavelengths_from_catalog(emission_lines, channel=None, emission_line_dir=DEFAULT_EMISSION_LINE_DIR):
    """
    Normalize an emission-line catalog into an array of air wavelengths.

    Accepted inputs are a one-dimensional list/array, a DataFrame with a
    wavelength column, a dict such as ``{'U': u_em_tab, 'G': g_em_tab}``,
    or ``None`` to load the default Hanuschik tables.
    """
    if emission_lines is None:
        emission_lines = load_emission_line_catalog(emission_line_dir=emission_line_dir, channel=channel)
    return _helper_line_wavelengths_from_catalog(emission_lines, channel=channel)


def _gaussian_plus_line(params, wavelength):
    amp, mean, sigma, c0, c1 = params
    x = wavelength - mean
    return amp * np.exp(-0.5 * (x / sigma) ** 2) + c0 + c1 * x


def _sigma_clip_flexure_anchors(curve, good, sigma_clip, channel, residual_floor=3):
    """Reject isolated flexure anchors without clipping a real I-band trend."""
    if sigma_clip is None or np.sum(good) < 3:
        return good

    values = curve["Flexure Correction"].values.astype(float)
    channel = str(channel).upper() if channel is not None else ""

    # The I-band emission curve is commonly U-shaped across its broad
    # wavelength range. Clipping about a single median can therefore reject a
    # precise edge anchor merely because it provides real wavelength leverage.
    if channel == "I" and np.sum(good) >= 5:
        wavelength = curve["Wavelength"].values.astype(float)
        center = np.nanmedian(wavelength[good])
        scale_x = 0.5 * np.ptp(wavelength[good])
        if np.isfinite(scale_x) and scale_x > 0:
            x = (wavelength - center) / scale_x
            degree = min(2, np.sum(good) - 2)
            initial = np.polyfit(x[good], values[good], degree)
            initial_residual = values[good] - np.polyval(initial, x[good])
            scale = 1.4826 * np.nanmedian(np.abs(initial_residual - np.nanmedian(initial_residual)))
            scale = max(scale, residual_floor)
            robust = least_squares(
                lambda coeff: (np.polyval(coeff, x[good]) - values[good]) / scale,
                initial,
                loss="soft_l1",
                f_scale=1,
            )
            residual = values - np.polyval(robust.x, x)
            center_residual = np.nanmedian(residual[good])
            scatter = 1.4826 * np.nanmedian(np.abs(residual[good] - center_residual))
            scatter = max(scatter, residual_floor)
            return good & (np.abs(residual - center_residual) < sigma_clip * scatter)

    median = np.nanmedian(values[good])
    scatter = 1.4826 * np.nanmedian(np.abs(values[good] - median))
    scatter = max(scatter, residual_floor)
    return good & (np.abs(values - median) < sigma_clip * scatter)


def fit_sky_emission_line(wavelength, sky_flux, rest_wavelength, window_size=5, min_pixels=5):
    """Fit one sky-emission line with a Gaussian plus linear continuum."""
    wavelength = _float_array(wavelength)
    sky_flux = _float_array(sky_flux)
    m = (wavelength > rest_wavelength - window_size) & (wavelength < rest_wavelength + window_size) & np.isfinite(wavelength) & np.isfinite(sky_flux)
    if np.sum(m) < min_pixels:
        raise ValueError("too few pixels around emission line")

    x = wavelength[m]
    y = sky_flux[m]
    c0 = np.nanmedian(y)
    amp = np.nanmax(y) - c0
    if not np.isfinite(amp) or amp <= 0:
        raise ValueError("emission line has non-positive amplitude")

    p0 = np.array([amp, rest_wavelength, 1.0, c0, 0.0], dtype=float)
    lo = np.array([0.0, rest_wavelength - window_size, 0.05, -np.inf, -np.inf], dtype=float)
    hi = np.array([np.inf, rest_wavelength + window_size, window_size, np.inf, np.inf], dtype=float)

    scale = 1.4826 * np.nanmedian(np.abs(y - np.nanmedian(y)))
    if not np.isfinite(scale) or scale <= 0:
        scale = max(np.nanstd(y), 1.0)

    def residuals(params):
        return (_gaussian_plus_line(params, x) - y) / scale

    result = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=1000)
    if not result.success:
        raise ValueError(result.message)

    params = result.x
    if params[2] <= 0 or abs(params[1] - rest_wavelength) > window_size:
        raise ValueError("unphysical emission-line fit")

    flex_corr = C_KMS * (params[1] - rest_wavelength) / rest_wavelength
    errs = least_squares_errors(result)
    flex_err = np.nan
    if len(errs) > 1 and np.isfinite(errs[1]):
        flex_err = C_KMS * abs(errs[1] / rest_wavelength)
    return params, flex_corr, flex_err


def derive_sky_emission_flexure_curve(
    fits_file, emission_lines=None, channel=None, trace_column=None, trace_y=None, window_size=5, min_lines=1, obs_wavelengths_are_air=True,
    empirical_coeffs=None, curve_sigma_clip=4, max_flexure_error=35, plot=False, emission_line_dir=DEFAULT_EMISSION_LINE_DIR,
    emission_min_strength="default", emission_min_line_separation="default",
):
    """
    Derive a wavelength-dependent flexure curve from sky-emission lines.

    The returned table uses the same core columns as the telluric+stellar curve,
    so downstream RV code can evaluate either curve in the same way.
    """
    sky_wl, sky_flux, header, fitted_trace_y = read_sky_model_at_trace(fits_file, trace_column=trace_column, trace_y=trace_y)
    if channel is None:
        channel = _read_channel(header, fits_file, sky_wl)

    if emission_lines is None:
        emission_lines = load_emission_line_catalog(
            emission_line_dir=emission_line_dir, channel=channel, min_strength=emission_min_strength, min_separation=emission_min_line_separation
        )
    line_wavelengths = _line_wavelengths_from_catalog(emission_lines, channel=channel, emission_line_dir=emission_line_dir)
    if emission_lines is not None and not hasattr(emission_lines, "columns") and emission_min_line_separation not in {None, "default"}:
        line_wavelengths = _filter_line_wavelengths_by_separation(line_wavelengths, emission_min_line_separation)
    wl_min, wl_max = np.nanmin(sky_wl), np.nanmax(sky_wl)
    line_wavelengths = line_wavelengths[(line_wavelengths > wl_min) & (line_wavelengths < wl_max)]

    rows = []
    for rest_wavelength in line_wavelengths:
        try:
            params, flex_corr, flex_err = fit_sky_emission_line(sky_wl, sky_flux, rest_wavelength, window_size=window_size)
        except Exception:
            continue

        if empirical_coeffs is not None:
            m, b = empirical_coeffs
            flex_corr = (flex_corr - b) / (1 + m)

        curve_wavelength = convert_air_to_vacuum(np.array([rest_wavelength]))[0] if obs_wavelengths_are_air else rest_wavelength
        rows.append(
            {
                "Wavelength": curve_wavelength, "Air Wavelength [A]": rest_wavelength, "Flexure Correction": flex_corr, "Flexure Correction Error": flex_err,
                "Amplitude": params[0], "Fit Mean": params[1], "Fit Sigma": params[2], "Trace Pixel": fitted_trace_y, "Success": True,
            }
        )

    curve = pd.DataFrame(rows)
    if len(curve) == 0 or len(curve) < min_lines:
        curve = pd.DataFrame(
            columns=[
                "Wavelength", "Air Wavelength [A]", "Flexure Correction", "Flexure Correction Error", "Amplitude", "Fit Mean", "Fit Sigma", "Trace Pixel",
                "Success", "Good",
            ]
        )
        curve.attrs["Fallback Flexure"] = np.nan
        curve.attrs["Channel"] = channel
        curve.attrs["Flexure Source"] = "sky_emission"
        return curve

    good = np.isfinite(curve["Flexure Correction"].values)
    if "Flexure Correction Error" in curve:
        err = curve["Flexure Correction Error"].values.astype(float)
        good &= (~np.isfinite(err)) | (err < max_flexure_error)

    good = _sigma_clip_flexure_anchors(curve, good, curve_sigma_clip, channel)

    curve["Good"] = good
    fallback = np.nanmedian(curve.loc[curve["Good"], "Flexure Correction"]) if np.any(curve["Good"]) else np.nan
    curve.attrs["Fallback Flexure"] = fallback
    curve.attrs["Channel"] = channel
    curve.attrs["Flexure Source"] = "sky_emission"

    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.scatter(curve["Air Wavelength [A]"], curve["Flexure Correction"], c=curve["Good"], cmap="coolwarm")
        plt.xlabel(r"Observed-frame air wavelength [$\mathrm{\AA}$]")
        plt.ylabel("Emission flexure [km/s]")
    return curve


__all__ = ["derive_sky_emission_flexure_curve", "fit_sky_emission_line"]
