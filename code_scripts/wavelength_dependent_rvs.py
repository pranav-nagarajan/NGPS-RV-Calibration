"""Public entry point for NGPS wavelength-dependent radial velocities."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from absorption_flexure import derive_telluric_stellar_flexure_curve, evaluate_telluric_flexure_curve, telluric_good_pixel_mask
from emission_flexure import derive_sky_emission_flexure_curve, fit_sky_emission_line
from rv_helpers import (
    DEFAULT_BOSZ_GRID, DEFAULT_BOSZ_WAVELENGTHS, DEFAULT_EMISSION_LINE_DIR, DEFAULT_TELLURIC_GRID, DEFAULT_TELLURIC_GRID_DIR, DEFAULT_TEMPLATE_RESOLUTION,
    DEFAULT_TELLURIC_MODEL_RESOLUTION, NGPS_CHANNEL_RANGES, _float_array, _header_float, _read_airmass, _read_channel, _read_coordinate_degrees,
    _read_observation_time, barycentric_correction, convert_air_to_vacuum, degrade_spectrum_resolution, doppler_shift, estimate_model_continuum_flux_error,
    estimate_normalized_flux_error,
    load_bosz_wavelengths, measure_resolution, medfilt_fixed_window_AA, read_reduced_2d_spectrum, retrieve_bosz_spectrum, select_telluric_grid_path,
    set_data_dir, template_from_inputs, _resolve_path,
)


DEFAULT_EXAMPLE_TEMPLATE = "template_spectra/bosz2024_mp_t6000_g+4.0_m+0.00_a+0.00_c+0.00_v2_r5000_resam.txt.gz"
G_MGB_REGION = (5150, 5700)
R_H_ALPHA_REGION = (6400, 6600)
I_CA_TRIPLET_REGION = (8400, 9000)
DEFAULT_RV_REGIONS = {"G": G_MGB_REGION, "R": R_H_ALPHA_REGION, "I": I_CA_TRIPLET_REGION}


@dataclass
class RVResult:
    """Container returned by ``measure_rv(..., return_details=True)``."""

    rv: float
    rv_error: float
    barycentric_correction: float
    jd_utc: float
    airmass: float
    channel: str
    flexure_source: str
    stellar_rv_guess: float
    n_chunks_used: int
    n_flexure_anchors: int
    spectral_resolution: float
    telluric_grid_path: str
    spectrum_snr: float
    spectrum_flux_error: float
    low_snr: bool
    flexure_curve: pd.DataFrame = field(repr=False)
    chunk_rvs: pd.DataFrame = field(repr=False)
    initial_rv_guess: Optional[float] = None
    initial_rv_source: str = ""
    resolution_table: Optional[pd.DataFrame] = field(default=None, repr=False)
    diagnostic_figures: Optional[dict] = field(default=None, repr=False)


def load_example_template(template_path=DEFAULT_EXAMPLE_TEMPLATE, wavelength_path=DEFAULT_BOSZ_WAVELENGTHS):
    """
    Load the BOSZ template shipped with the public tutorial.

    The template file stores two columns that are converted into normalized
    flux as ``column_0 / column_1``, matching the original tutorial.
    """
    wave = load_bosz_wavelengths(wavelength_path)
    spec = np.genfromtxt(_resolve_path(template_path))
    if spec.ndim == 2 and spec.shape[1] >= 2:
        flux = spec[:, 0] / spec[:, 1]
    else:
        flux = spec
    return _float_array(wave), _float_array(flux)


def _public_flexure_source_name(flexure_source):
    source = str(flexure_source).strip().lower()
    aliases = {
        "auto": "auto",
        "default": "auto",
        "telluric": "telluric_stellar",
        "telluric_absorption": "telluric_stellar",
        "absorption": "telluric_stellar",
        "sky": "sky_emission",
        "emission": "sky_emission",
    }
    return aliases.get(source, source)


def measure_ngps_rv(
    fits_file, template_wavelength=None, template_flux=None, *, template_path=DEFAULT_EXAMPLE_TEMPLATE, template_wavelength_path=DEFAULT_BOSZ_WAVELENGTHS,
    flexure_source="auto", rv_bounds=None, rv_windows=None, stellar_rv_guess=None, rv_prior=None, return_details=True, **measure_kwargs,
):
    """
    Measure one NGPS radial velocity with tutorial-friendly defaults.

    Parameters most users need are just ``fits_file`` plus either an explicit
    ``template_wavelength``/``template_flux`` pair or a ``template_path``. The
    stellar RV guess is an optional helper for the optimizer and, when supplied,
    is also used as the RV prior unless ``rv_prior`` is explicitly set. It does
    not need to be the true answer.
    """
    if (template_wavelength is None) != (template_flux is None):
        raise ValueError("Provide both template_wavelength and template_flux, or provide neither to use template_path.")
    if template_wavelength is None:
        template_wavelength, template_flux = load_example_template(template_path=template_path, wavelength_path=template_wavelength_path)

    if rv_bounds is not None and rv_windows is not None:
        raise ValueError("Use either rv_bounds or rv_windows, not both.")
    if rv_bounds is not None:
        measure_kwargs.setdefault("rv_bounds", rv_bounds)
    if rv_windows is not None:
        measure_kwargs.setdefault("rv_windows", rv_windows)

    resolved_flexure_source = _public_flexure_source_name(flexure_source)
    if rv_prior is None and stellar_rv_guess is not None:
        rv_prior = stellar_rv_guess
    measure_kwargs.setdefault("stellar_rv_guess", stellar_rv_guess)
    measure_kwargs.setdefault("rv_prior", rv_prior)
    measure_kwargs.setdefault("flexure_source", resolved_flexure_source)

    return measure_rv(
        fits_file, template_wavelength=template_wavelength, template_flux=template_flux, return_details=return_details, **measure_kwargs
    )


def derive_flexure_curve(
    fits_file, template_wavelength=None, template_flux=None, *, template_path=DEFAULT_EXAMPLE_TEMPLATE, template_wavelength_path=DEFAULT_BOSZ_WAVELENGTHS,
    flexure_source="auto", stellar_rv_guess=0.0, **measure_kwargs,
):
    """
    Return only the flexure curve for one spectrum.

    Internally this runs the same measurement path as ``measure_ngps_rv`` so
    the flexure table matches the RV result a user would get afterward.
    """
    measure_kwargs.pop("return_details", None)
    result = measure_ngps_rv(
        fits_file, template_wavelength=template_wavelength, template_flux=template_flux, template_path=template_path,
        template_wavelength_path=template_wavelength_path, flexure_source=flexure_source, stellar_rv_guess=stellar_rv_guess,
        return_details=True, **measure_kwargs,
    )
    return result.flexure_curve


def _stellar_template_wavelengths_for_flexure(temp_wl, template_wavelengths_are_air):
    return convert_air_to_vacuum(temp_wl) if template_wavelengths_are_air else temp_wl


def _stellar_wavelengths_for_rv(obs_wl, temp_wl, obs_wavelengths_are_air, template_wavelengths_are_air):
    if obs_wavelengths_are_air and not template_wavelengths_are_air:
        return convert_air_to_vacuum(obs_wl), temp_wl
    if template_wavelengths_are_air and not obs_wavelengths_are_air:
        return obs_wl, convert_air_to_vacuum(temp_wl)
    return obs_wl, temp_wl


def radial_velocity_masked(
    obs_wavl, obs_flux, temp_wavl, temp_flux, fit_mask=None, obs_fluxerr=None, flex_corr=0, bary_corr=0, rv_prior=None, rv_half_width=150, coarse_step=0.3,
    fine_half_width=5, fine_step=0.03, lims=None, return_quality=False, residual_spike_abs_threshold=2.0,
):
    """Measure a stellar RV with an optional telluric good-pixel mask."""
    quality = {
        "N Fit Pixels": 0,
        "Flux Error": np.nan,
        "Best Fit Chi2": np.nan,
        "Reduced Chi2": np.nan,
        "Fine Chi2 Edge Delta": np.nan,
        "Coarse Chi2 Edge Delta": np.nan,
        "Second Minimum Delta Chi2": np.nan,
        "RV Grid Edge Distance": np.nan,
        "Coarse Best RV": np.nan,
        "Residual Spike Fraction": np.nan,
        "Continuum Noise Pixels": 0,
        "Adaptive Model Continuum": False,
        "Flux Error Method": "user supplied" if obs_fluxerr is not None else "normalized scatter fallback",
    }

    def finish(rv, rv_err):
        if return_quality:
            return float(rv), float(rv_err), quality
        return float(rv), float(rv_err)

    obs_wavl = _float_array(obs_wavl)
    obs_flux = _float_array(obs_flux)
    temp_wavl = _float_array(temp_wavl)
    temp_flux = _float_array(temp_flux)

    continuum = medfilt_fixed_window_AA(obs_wavl, obs_flux, window_AA=21)
    norm_flux = np.divide(obs_flux, continuum, out=np.full_like(obs_flux, np.nan, dtype=float), where=continuum != 0)

    temp_mask = (temp_wavl > np.nanmin(obs_wavl) - 10) & (temp_wavl < np.nanmax(obs_wavl) + 10)
    if fit_mask is None:
        fit_mask = np.ones_like(obs_wavl, dtype=bool)
    spike_mask = fit_mask & np.isfinite(norm_flux)
    m = fit_mask & np.isfinite(norm_flux) & (norm_flux > 0.05) & (norm_flux < 2.0)
    if lims is not None:
        spike_mask &= (obs_wavl > lims[0]) & (obs_wavl < lims[1])
        m &= (obs_wavl > lims[0]) & (obs_wavl < lims[1])
    quality["N Fit Pixels"] = int(np.sum(m))
    if np.sum(m) < 20 or np.sum(temp_mask) < 5:
        return finish(np.nan, np.nan)

    if obs_fluxerr is None:
        flux_error = estimate_normalized_flux_error(obs_wavl, obs_flux, mask=fit_mask, continuum_lims=lims, continuum_window_AA=21)
        norm_err = np.full_like(norm_flux, flux_error, dtype=float)
    else:
        norm_err = _float_array(obs_fluxerr) / continuum
    quality["Flux Error"] = float(flux_error) if obs_fluxerr is None else np.nan
    m &= np.isfinite(norm_err) & (norm_err > 0)
    spike_mask &= np.isfinite(norm_err) & (norm_err > 0)
    quality["N Fit Pixels"] = int(np.sum(m))
    if np.sum(m) < 20:
        return finish(np.nan, np.nan)

    if rv_prior is None:
        shifts = np.arange(-600, 600, coarse_step)
    else:
        shifts = np.arange(rv_prior - rv_half_width, rv_prior + rv_half_width, coarse_step)

    chi2s = []
    for shift in shifts:
        shifted_template = doppler_shift(temp_wavl[temp_mask], temp_flux[temp_mask], shift)
        model = np.interp(obs_wavl, temp_wavl[temp_mask], shifted_template, left=1, right=1)
        chi2s.append(np.nansum(((model - norm_flux) ** 2 / norm_err**2)[m]))
    chi2s = np.array(chi2s)
    if not np.any(np.isfinite(chi2s)):
        return finish(np.nan, np.nan)
    coarse_best_idx = int(np.nanargmin(chi2s))
    best_rv = shifts[coarse_best_idx]
    fine_shifts = np.arange(best_rv - fine_half_width, best_rv + fine_half_width, fine_step)
    fine_chi2s = []
    for shift in fine_shifts:
        shifted_template = doppler_shift(temp_wavl[temp_mask], temp_flux[temp_mask], shift)
        model = np.interp(obs_wavl, temp_wavl[temp_mask], shifted_template, left=1, right=1)
        fine_chi2s.append(np.nansum(((model - norm_flux) ** 2 / norm_err**2)[m]))
    fine_chi2s = np.array(fine_chi2s)
    if not np.any(np.isfinite(fine_chi2s)):
        return finish(np.nan, np.nan)
    fine_best_idx = int(np.nanargmin(fine_chi2s))
    best_rv = fine_shifts[fine_best_idx]
    shifted_template = doppler_shift(temp_wavl[temp_mask], temp_flux[temp_mask], best_rv)
    model = np.interp(obs_wavl, temp_wavl[temp_mask], shifted_template, left=1, right=1)

    if obs_fluxerr is None:
        model_flux_error, n_continuum, adaptive_continuum, _ = estimate_model_continuum_flux_error(
            norm_flux, model, mask=m, return_details=True,
        )
        if np.isfinite(model_flux_error) and model_flux_error > 0:
            chi2_scale = (float(flux_error) / float(model_flux_error)) ** 2
            chi2s *= chi2_scale
            fine_chi2s *= chi2_scale
            flux_error = float(model_flux_error)
            norm_err = np.full_like(norm_flux, flux_error, dtype=float)
            quality["Flux Error"] = flux_error
            quality["Continuum Noise Pixels"] = int(n_continuum)
            quality["Adaptive Model Continuum"] = bool(adaptive_continuum)
            quality["Flux Error Method"] = "model continuum residuals"

    coarse_min = float(chi2s[coarse_best_idx])
    quality["Coarse Best RV"] = float(shifts[coarse_best_idx])
    quality["Coarse Chi2 Edge Delta"] = float(np.nanmin([chi2s[0], chi2s[-1]]) - coarse_min)
    quality["RV Grid Edge Distance"] = float(min(shifts[coarse_best_idx] - shifts[0], shifts[-1] - shifts[coarse_best_idx]))
    separated = np.abs(shifts - shifts[coarse_best_idx]) >= 20
    if np.any(separated & np.isfinite(chi2s)):
        quality["Second Minimum Delta Chi2"] = float(np.nanmin(chi2s[separated]) - coarse_min)

    fine_min = float(fine_chi2s[fine_best_idx])
    spike_threshold = float(residual_spike_abs_threshold)
    if np.isfinite(spike_threshold) and spike_threshold > 0 and np.any(spike_mask):
        quality["Residual Spike Fraction"] = float(np.mean(np.abs((norm_flux - model)[spike_mask]) > spike_threshold))
    quality["Best Fit Chi2"] = fine_min
    quality["Reduced Chi2"] = float(fine_min / max(np.sum(m) - 1, 1))
    quality["Fine Chi2 Edge Delta"] = float(np.nanmin([fine_chi2s[0], fine_chi2s[-1]]) - fine_min)

    one_sigma = np.argsort(np.abs(fine_chi2s - np.nanmin(fine_chi2s) - 1))
    rv_err = np.nan
    if len(one_sigma) >= 2:
        rv_err = np.average(np.abs(fine_shifts[one_sigma[:2]] - best_rv))

    corrected_rv = best_rv - flex_corr + bary_corr
    return finish(corrected_rv, rv_err)


def combine_chunk_rvs(
    results, max_rv_err=25, sigma_clip=3, error_floor=3, min_chunks=3, chunk_quality=True, min_chunk_fit_pixels=30,
    min_chunk_chi2_edge_delta=0.1, min_chunk_chi2_second_delta=1, min_chunk_rv_grid_edge_distance=5, max_chunk_reduced_chi2=np.inf,
    max_chunk_residual_spike_fraction=np.inf,
):
    """Robustly combine chunk RVs and add a ``Use in Combined RV`` flag."""
    results = results.copy()
    rv = results["RV"].values.astype(float)
    rv_err = results["RV Error"].values.astype(float)
    good = np.isfinite(rv) & np.isfinite(rv_err) & (rv_err > 0) & (rv_err < max_rv_err)
    quality_good = np.ones(len(results), dtype=bool)
    quality_reason = np.full(len(results), "", dtype=object)

    def reject(where, reason):
        quality_good[where] = False
        quality_reason[where & (quality_reason == "")] = reason

    if chunk_quality:
        if "N Fit Pixels" in results:
            nfit = results["N Fit Pixels"].values.astype(float)
            reject(~np.isfinite(nfit) | (nfit < min_chunk_fit_pixels), "too few fit pixels")
        if "Fine Chi2 Edge Delta" in results:
            edge_delta = results["Fine Chi2 Edge Delta"].values.astype(float)
            reject(~np.isfinite(edge_delta) | (edge_delta < min_chunk_chi2_edge_delta), "flat fine chi2 minimum")
        if "Second Minimum Delta Chi2" in results:
            second_delta = results["Second Minimum Delta Chi2"].values.astype(float)
            reject(np.isfinite(second_delta) & (second_delta < min_chunk_chi2_second_delta), "ambiguous coarse chi2 minimum")
        if "RV Grid Edge Distance" in results:
            edge_distance = results["RV Grid Edge Distance"].values.astype(float)
            reject(~np.isfinite(edge_distance) | (edge_distance < min_chunk_rv_grid_edge_distance), "rv minimum near search edge")
        if np.isfinite(max_chunk_reduced_chi2) and "Reduced Chi2" in results:
            reduced_chi2 = results["Reduced Chi2"].values.astype(float)
            reject(~np.isfinite(reduced_chi2) | (reduced_chi2 > max_chunk_reduced_chi2), "large reduced chi2")
        if np.isfinite(max_chunk_residual_spike_fraction) and "Residual Spike Fraction" in results:
            spike_fraction = results["Residual Spike Fraction"].values.astype(float)
            reject(~np.isfinite(spike_fraction) | (spike_fraction > max_chunk_residual_spike_fraction), "messy residual spikes")

    results["Chunk Quality Passed"] = quality_good
    results["Chunk Quality Reason"] = quality_reason
    good &= quality_good

    for _ in range(4):
        if np.sum(good) < min_chunks:
            break
        center = np.nanmedian(rv[good])
        scatter = 1.4826 * np.nanmedian(np.abs(rv[good] - center))
        scatter = max(scatter, error_floor, np.nanmedian(rv_err[good]))
        new_good = good & (np.abs(rv - center) < sigma_clip * scatter)
        if np.all(new_good == good):
            break
        good = new_good

    results["Use in Combined RV"] = good
    if np.sum(good) == 0:
        return np.nan, np.nan, results

    weights = 1 / (rv_err[good] ** 2 + error_floor**2)
    combined_rv = np.average(rv[good], weights=weights)
    formal_err = np.sqrt(1 / np.sum(weights))
    if np.sum(good) > 1:
        scatter_err = 1.4826 * np.nanmedian(np.abs(rv[good] - np.nanmedian(rv[good]))) / np.sqrt(np.sum(good))
        combined_err = max(formal_err, scatter_err)
    else:
        combined_err = formal_err
    return combined_rv, combined_err, results


def _default_boundaries(obs_wl, channel):
    obs_wl = _float_array(obs_wl)
    finite = np.isfinite(obs_wl)
    if np.sum(finite) < 2:
        raise ValueError("Cannot build RV windows without a finite wavelength grid.")

    wl_min = np.nanmin(obs_wl[finite])
    wl_max = np.nanmax(obs_wl[finite])
    preferred_by_channel = {
        "U": (3800, 4360),
        "G": (5150, 5700),
        # Include H-alpha while still using robust chunks across the telluric-
        # anchored red side of the R arm.
        "R": np.arange(6400, 7501, 100),
        "I": np.arange(8400, 9001, 100),
    }
    channel_key = str(channel).upper()
    preferred = preferred_by_channel.get(channel_key)
    if preferred is not None and wl_min < preferred[0] and wl_max > preferred[-1]:
        return np.array(preferred, dtype=float)

    if channel_key in NGPS_CHANNEL_RANGES:
        channel_low, channel_high = NGPS_CHANNEL_RANGES[channel_key]
        low = max(channel_low, wl_min + 10)
        high = min(channel_high, wl_max - 10)
    else:
        low = wl_min + 10
        high = wl_max - 10

    if high - low < 100:
        low = wl_min + 10
        high = wl_max - 10
    if high - low < 100:
        raise ValueError("Spectrum does not cover enough wavelength range for chunked RV fitting.")
    return np.arange(low, high + 1, 100)


def rv_bounds_to_windows(rv_bounds, step_AA=100):
    """Convert ``(low, high)`` RV-fitting bounds into chunk boundaries."""
    low, high = np.asarray(rv_bounds, dtype=float)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("rv_bounds must be a finite (low, high) pair with high > low.")
    step_AA = float(step_AA)
    if not np.isfinite(step_AA) or step_AA <= 0:
        raise ValueError("rv_window_step must be a positive finite value.")
    boundaries = np.arange(low, high + 0.5 * step_AA, step_AA, dtype=float)
    if len(boundaries) < 2 or boundaries[-1] < high:
        boundaries = np.append(boundaries, high)
    else:
        boundaries[-1] = high
    return boundaries


def _snr_lims_from_inputs(obs_wl, channel, rv_windows=None, rv_bounds=None, rv_window_step=100):
    if rv_windows is not None:
        boundaries = np.asarray(rv_windows, dtype=float)
    elif rv_bounds is not None:
        boundaries = rv_bounds_to_windows(rv_bounds, step_AA=rv_window_step)
    else:
        boundaries = _default_boundaries(obs_wl, channel)
    return float(boundaries[0]), float(boundaries[-1])


def _chunked_rvs_for_spectrum(
    obs_wl, obs_flux, temp_wl, temp_flux, airmass, bary_corr, flexure_curve, boundaries, obs_wavelengths_are_air=True, obs_fluxerr=None,
    rv_prior=None, rv_half_width=150, curve_poly_degree=None, max_telluric_absorption=0.03, fallback_flexure=np.nan, max_rv_err=25, sigma_clip=3,
    error_floor=3,
    telluric_grid_path=DEFAULT_TELLURIC_GRID, template_wavelengths_are_air=True, spectral_resolution=None,
    telluric_model_resolution=DEFAULT_TELLURIC_MODEL_RESOLUTION, chunk_quality=True, min_chunk_fit_pixels=30, min_chunk_chi2_edge_delta=0.1,
    min_chunk_chi2_second_delta=1, min_chunk_rv_grid_edge_distance=5, max_chunk_reduced_chi2=np.inf,
    max_chunk_residual_spike_fraction=np.inf, residual_spike_abs_threshold=2.0,
):
    rows = []
    obs_wl = _float_array(obs_wl)
    obs_fluxerr = None if obs_fluxerr is None else _float_array(obs_fluxerr)
    if obs_fluxerr is not None and len(obs_fluxerr) != len(obs_wl):
        raise ValueError("obs_fluxerr must have the same length as the extracted observed spectrum.")
    rv_wl, rv_temp_wl = _stellar_wavelengths_for_rv(
        obs_wl, temp_wl, obs_wavelengths_are_air=obs_wavelengths_are_air, template_wavelengths_are_air=template_wavelengths_are_air
    )
    good_tell_cache = {}

    for low, high in zip(boundaries[:-1], boundaries[1:]):
        center = 0.5 * (low + high)
        flex_wl = convert_air_to_vacuum(np.array([center]))[0] if obs_wavelengths_are_air else center
        flex_corr = evaluate_telluric_flexure_curve(flex_wl, flexure_curve, fallback_flexure=fallback_flexure, poly_degree=curve_poly_degree)

        obs_bool = (obs_wl > low - 10) & (obs_wl < high + 10)
        quality = {}
        if not np.isfinite(flex_corr) or np.sum(obs_bool) < 20:
            rv, rv_err = np.nan, np.nan
        else:
            key = round(float(flex_corr), 4)
            if key not in good_tell_cache:
                good_tell_cache[key] = telluric_good_pixel_mask(
                    obs_wl, airmass, flex_corr=flex_corr, max_telluric_absorption=max_telluric_absorption, obs_wavelengths_are_air=obs_wavelengths_are_air,
                    telluric_grid_path=telluric_grid_path, spectral_resolution=spectral_resolution, telluric_model_resolution=telluric_model_resolution,
                )
            fit_lims = convert_air_to_vacuum(np.array([low, high])) if obs_wavelengths_are_air and not template_wavelengths_are_air else (low, high)
            chunk_fluxerr = None if obs_fluxerr is None else obs_fluxerr[obs_bool]
            rv, rv_err, quality = radial_velocity_masked(
                rv_wl[obs_bool], obs_flux[obs_bool], rv_temp_wl, temp_flux, fit_mask=good_tell_cache[key][obs_bool], obs_fluxerr=chunk_fluxerr,
                flex_corr=flex_corr, bary_corr=bary_corr, rv_prior=rv_prior, rv_half_width=rv_half_width, lims=fit_lims, return_quality=True,
                residual_spike_abs_threshold=residual_spike_abs_threshold,
            )

        row = {"Wavelength Min": low, "Wavelength Max": high, "Wavelength Mid": center, "RV": rv, "RV Error": rv_err, "Flexure Correction": flex_corr}
        row.update(quality)
        rows.append(row)

    chunk_results = pd.DataFrame(rows)
    return combine_chunk_rvs(
        chunk_results, max_rv_err=max_rv_err, sigma_clip=sigma_clip, error_floor=error_floor, chunk_quality=chunk_quality,
        min_chunk_fit_pixels=min_chunk_fit_pixels, min_chunk_chi2_edge_delta=min_chunk_chi2_edge_delta,
        min_chunk_chi2_second_delta=min_chunk_chi2_second_delta, min_chunk_rv_grid_edge_distance=min_chunk_rv_grid_edge_distance,
        max_chunk_reduced_chi2=max_chunk_reduced_chi2, max_chunk_residual_spike_fraction=max_chunk_residual_spike_fraction,
    )


def measure_rv(
    fits_file, *, template_wavelength=None, template_flux=None, teff=None, logg=None, mh=None, stellar_rv_guess=None, rv_prior=None, obs_fluxerr=None,
    rv_windows=None, rv_bounds=None, rv_window_step=100, ra=None, dec=None, mjd=None, jd=None, airmass=None, trace_column=None, aperture_half_width=2,
    trace_y=None,
    obs_wavelengths_are_air=True,
    template_wavelengths_are_air=True, curve_poly_degree=None, fallback="anchor", rv_half_width=150, max_telluric_absorption=0.03, max_rv_err=25, sigma_clip=3,
    error_floor=3, model_kind="multiplicative", fit_rv_offset=False, flexure_source="auto", emission_lines=None,
    emission_line_dir=DEFAULT_EMISSION_LINE_DIR, emission_window_size=5, emission_min_lines=1, emission_min_strength="default",
    emission_min_line_separation="default", emission_empirical_coeffs=None,
    window_AA=120, step_AA=60, min_telluric_depth=0.02, min_telluric_area=0.2, max_flexure_error=35, min_telluric_scale=0.15,
    max_telluric_scale=2.8, max_stellar_scale=2.3, min_good_telluric_depth=0.08, curve_sigma_clip=4, spectral_resolution="auto",
    blue_anchor_wavelength_max=6563, min_blue_good_telluric_depth=None, max_blue_flexure_error=None, min_blue_telluric_scale=0.3,
    max_blue_telluric_scale=2.5, max_blue_stellar_scale=1.2, min_blue_anchors=2, max_blue_flexure_deviation=12,
    resolution_emission_lines=None, resolution_window_size=5, resolution_min_lines=1, telluric_grid_path="auto", telluric_grid_dir=DEFAULT_TELLURIC_GRID_DIR,
    telluric_model_resolution=DEFAULT_TELLURIC_MODEL_RESOLUTION, template_resolution=DEFAULT_TEMPLATE_RESOLUTION, degrade_template=True,
    chunk_quality=True, min_chunk_fit_pixels=30, min_chunk_chi2_edge_delta=0.1, min_chunk_chi2_second_delta=1, min_chunk_rv_grid_edge_distance=5,
    max_chunk_reduced_chi2=np.inf, max_chunk_residual_spike_fraction="default", residual_spike_abs_threshold=2.0,
    min_snr="default", snr_lims=None, snr_window_AA=51, auto_seed_rv=True,
    bosz_wave_path=DEFAULT_BOSZ_WAVELENGTHS, bosz_grid_dir=DEFAULT_BOSZ_GRID, plot_diagnostics=False, diagnostic_kwargs=None, return_details=False,
):
    """
    Measure a barycentric- and flexure-corrected RV from one reduced 2D FITS file.

    By default the function returns ``(rv, rv_error)`` in km/s. Set
    ``return_details=True`` to receive an ``RVResult`` with the barycentric
    correction, flexure curve, and per-chunk RV table. ``rv_bounds`` and
    ``rv_windows`` only choose the wavelength chunks used for the final stellar
    RV fit; they do not restrict the telluric or sky-emission flexure-curve
    derivation.
    """
    user_supplied_stellar_guess = stellar_rv_guess is not None
    user_supplied_rv_prior = rv_prior is not None
    initial_rv_guess = None
    initial_rv_source = ""

    obs_wl, obs_flux, header = read_reduced_2d_spectrum(fits_file, trace_column=trace_column, aperture_half_width=aperture_half_width, trace_y=trace_y)
    obs_fluxerr = None if obs_fluxerr is None else _float_array(obs_fluxerr)
    if obs_fluxerr is not None and len(obs_fluxerr) != len(obs_wl):
        raise ValueError("obs_fluxerr must have the same length as the extracted observed spectrum.")
    channel = _read_channel(header, fits_file, obs_wl)
    ra_deg, dec_deg = _read_coordinate_degrees(header, ra=ra, dec=dec)
    obstime = _read_observation_time(header, mjd=mjd, jd=jd)
    bary_corr = barycentric_correction(ra_deg, dec_deg, obstime)
    obs_airmass = _read_airmass(header, airmass=airmass)
    flexure_source = str(flexure_source).strip().lower()
    if flexure_source in {"auto", "default"}:
        if channel == "G":
            flexure_source = "sky_emission"
        elif channel == "U":
            flexure_source = "unavailable"
        else:
            flexure_source = "telluric_stellar"

    if snr_lims is None:
        snr_lims = _snr_lims_from_inputs(obs_wl, channel, rv_windows=rv_windows, rv_bounds=rv_bounds, rv_window_step=rv_window_step)
    spectrum_flux_error, spectrum_snr = estimate_normalized_flux_error(
        obs_wl, obs_flux, continuum_lims=snr_lims, continuum_window_AA=snr_window_AA, return_snr=True
    )
    if isinstance(min_snr, str):
        if min_snr.lower() != "default":
            raise ValueError("min_snr must be numeric, None, or 'default'.")
        resolved_min_snr = 4.0 if channel == "U" else 5.0
    else:
        resolved_min_snr = min_snr
    if isinstance(max_chunk_residual_spike_fraction, str):
        if max_chunk_residual_spike_fraction.lower() != "default":
            raise ValueError("max_chunk_residual_spike_fraction must be numeric, None, or 'default'.")
        resolved_max_chunk_residual_spike_fraction = 0.015 if channel == "G" else np.inf
    elif max_chunk_residual_spike_fraction is None:
        resolved_max_chunk_residual_spike_fraction = np.inf
    else:
        resolved_max_chunk_residual_spike_fraction = float(max_chunk_residual_spike_fraction)
    if channel == "U" and flexure_source == "unavailable":
        warnings.warn(
            "U-band spectra have no strong telluric absorption features in the default tutorial configuration, "
            "so a flexure-corrected RV cannot be fit automatically.",
            RuntimeWarning,
        )
        result = RVResult(
            rv=np.nan, rv_error=np.nan, barycentric_correction=float(bary_corr), jd_utc=float(obstime.jd), airmass=float(obs_airmass), channel=channel,
            flexure_source=flexure_source, stellar_rv_guess=np.nan if stellar_rv_guess is None else float(stellar_rv_guess), n_chunks_used=0,
            n_flexure_anchors=0, spectral_resolution=np.nan, telluric_grid_path="", spectrum_snr=float(spectrum_snr),
            spectrum_flux_error=float(spectrum_flux_error), low_snr=False, initial_rv_guess=initial_rv_guess, initial_rv_source=initial_rv_source,
            flexure_curve=pd.DataFrame(), chunk_rvs=pd.DataFrame(), resolution_table=None,
        )
        return result if return_details else (result.rv, result.rv_error)
    low_snr = bool(resolved_min_snr is not None and (not np.isfinite(spectrum_snr) or spectrum_snr < float(resolved_min_snr)))
    if low_snr:
        result = RVResult(
            rv=np.nan, rv_error=np.nan, barycentric_correction=float(bary_corr), jd_utc=float(obstime.jd), airmass=float(obs_airmass), channel=channel,
            flexure_source=flexure_source, stellar_rv_guess=np.nan if stellar_rv_guess is None else float(stellar_rv_guess), n_chunks_used=0,
            n_flexure_anchors=0, spectral_resolution=np.nan, telluric_grid_path="", spectrum_snr=float(spectrum_snr),
            spectrum_flux_error=float(spectrum_flux_error), low_snr=True, initial_rv_guess=initial_rv_guess, initial_rv_source=initial_rv_source,
            flexure_curve=pd.DataFrame(), chunk_rvs=pd.DataFrame(), resolution_table=None,
        )
        return result if return_details else (result.rv, result.rv_error)

    resolution_table = None
    if isinstance(spectral_resolution, str):
        if spectral_resolution.lower() != "auto":
            raise ValueError("spectral_resolution must be 'auto', None, or a numeric resolving power.")
        measured_resolution, resolution_table = measure_resolution(
            fits_file, emission_lines=resolution_emission_lines, channel=channel, trace_column=trace_column, trace_y=trace_y,
            window_size=resolution_window_size, min_lines=resolution_min_lines, return_table=True,
        )
    elif spectral_resolution is None:
        measured_resolution = np.nan
    else:
        measured_resolution = float(spectral_resolution)

    if isinstance(telluric_grid_path, str) and telluric_grid_path.lower() == "auto":
        resolved_telluric_grid_path = select_telluric_grid_path(
            measured_resolution, telluric_grid_dir=telluric_grid_dir, fallback_grid_path=DEFAULT_TELLURIC_GRID
        )
    else:
        resolved_telluric_grid_path = telluric_grid_path

    temp_wl, temp_flux = template_from_inputs(
        header, template_wavelength=template_wavelength, template_flux=template_flux, teff=teff, logg=logg, mh=mh, bosz_wave_path=bosz_wave_path,
        bosz_grid_dir=bosz_grid_dir,
    )
    diagnostic_template_flux = temp_flux.copy()
    if degrade_template and np.isfinite(measured_resolution):
        temp_flux = degrade_spectrum_resolution(temp_wl, temp_flux, output_resolution=measured_resolution, input_resolution=template_resolution)
    flexure_temp_wl = _stellar_template_wavelengths_for_flexure(temp_wl, template_wavelengths_are_air)

    should_auto_seed = (
        bool(auto_seed_rv)
        and channel in {"R", "I"}
        and flexure_source in ["telluric_stellar", "telluric", "absorption"]
        and not user_supplied_stellar_guess
        and not user_supplied_rv_prior
    )
    if should_auto_seed:
        try:
            seed_result = measure_rv(
                fits_file, template_wavelength=temp_wl, template_flux=temp_flux, stellar_rv_guess=None, rv_prior=None, obs_fluxerr=obs_fluxerr,
                rv_windows=rv_windows, rv_bounds=rv_bounds, rv_window_step=rv_window_step, ra=ra, dec=dec, mjd=mjd, jd=jd, airmass=airmass,
                trace_column=trace_column, aperture_half_width=aperture_half_width, trace_y=trace_y, obs_wavelengths_are_air=obs_wavelengths_are_air,
                template_wavelengths_are_air=template_wavelengths_are_air, curve_poly_degree=curve_poly_degree, fallback=fallback,
                rv_half_width=rv_half_width, max_telluric_absorption=max_telluric_absorption, max_rv_err=max_rv_err, sigma_clip=sigma_clip,
                error_floor=error_floor, model_kind=model_kind, fit_rv_offset=fit_rv_offset, flexure_source="sky_emission", emission_lines=emission_lines,
                emission_line_dir=emission_line_dir, emission_window_size=emission_window_size, emission_min_lines=emission_min_lines,
                emission_min_strength=emission_min_strength, emission_min_line_separation=emission_min_line_separation,
                emission_empirical_coeffs=emission_empirical_coeffs, resolution_emission_lines=resolution_emission_lines,
                resolution_window_size=resolution_window_size, resolution_min_lines=resolution_min_lines, telluric_grid_path=resolved_telluric_grid_path,
                telluric_grid_dir=telluric_grid_dir, telluric_model_resolution=telluric_model_resolution, template_resolution=template_resolution,
                degrade_template=False, chunk_quality=chunk_quality, min_chunk_fit_pixels=min_chunk_fit_pixels,
                min_chunk_chi2_edge_delta=min_chunk_chi2_edge_delta, min_chunk_chi2_second_delta=min_chunk_chi2_second_delta,
                min_chunk_rv_grid_edge_distance=min_chunk_rv_grid_edge_distance, max_chunk_reduced_chi2=max_chunk_reduced_chi2,
                max_chunk_residual_spike_fraction=max_chunk_residual_spike_fraction, residual_spike_abs_threshold=residual_spike_abs_threshold,
                min_snr=resolved_min_snr, snr_lims=snr_lims, snr_window_AA=snr_window_AA, auto_seed_rv=False, plot_diagnostics=False,
                return_details=True,
            )
            if np.isfinite(seed_result.rv):
                initial_rv_guess = float(seed_result.rv)
                initial_rv_source = "sky_emission"
                stellar_rv_guess = initial_rv_guess
                rv_prior = initial_rv_guess
        except Exception as exc:
            initial_rv_source = f"sky_emission failed: {exc}"

    if stellar_rv_guess is None and rv_prior is not None:
        stellar_rv_guess = rv_prior
    if stellar_rv_guess is None:
        stellar_rv_guess = _header_float(header, ("RV", "VRAD", "VHELIO"), default=0.0)
    stellar_rv_guess = float(stellar_rv_guess)

    if rv_windows is not None and rv_bounds is not None:
        raise ValueError("Use either rv_windows or rv_bounds, not both.")
    if rv_windows is not None:
        boundaries = np.asarray(rv_windows, dtype=float)
        if boundaries.ndim != 1 or len(boundaries) < 2:
            raise ValueError("rv_windows must be a one-dimensional list of wavelength boundaries.")
    elif rv_bounds is not None:
        boundaries = rv_bounds_to_windows(rv_bounds, step_AA=rv_window_step)
    else:
        boundaries = _default_boundaries(obs_wl, channel)

    if flexure_source in ["telluric_stellar", "telluric", "absorption"]:
        flexure_curve = derive_telluric_stellar_flexure_curve(
            obs_wl, obs_flux, flexure_temp_wl, temp_flux, obs_airmass, stellar_rv_guess, bary_corr, channel=channel, window_AA=window_AA, step_AA=step_AA,
            min_telluric_depth=min_telluric_depth, min_telluric_area=min_telluric_area, obs_wavelengths_are_air=obs_wavelengths_are_air, model_kind=model_kind,
            fit_rv_offset=fit_rv_offset, curve_sigma_clip=curve_sigma_clip, max_flexure_error=max_flexure_error, min_telluric_scale=min_telluric_scale,
            max_telluric_scale=max_telluric_scale, max_stellar_scale=max_stellar_scale, min_good_telluric_depth=min_good_telluric_depth,
            blue_anchor_wavelength_max=blue_anchor_wavelength_max, min_blue_good_telluric_depth=min_blue_good_telluric_depth,
            max_blue_flexure_error=max_blue_flexure_error, min_blue_telluric_scale=min_blue_telluric_scale,
            max_blue_telluric_scale=max_blue_telluric_scale, max_blue_stellar_scale=max_blue_stellar_scale, min_blue_anchors=min_blue_anchors,
            max_blue_flexure_deviation=max_blue_flexure_deviation,
            telluric_grid_path=resolved_telluric_grid_path, spectral_resolution=measured_resolution, telluric_model_resolution=telluric_model_resolution,
        )
    elif flexure_source in ["sky_emission", "emission", "sky"]:
        flexure_curve = derive_sky_emission_flexure_curve(
            fits_file, emission_lines, channel=channel, trace_column=trace_column, trace_y=trace_y, window_size=emission_window_size,
            min_lines=emission_min_lines, obs_wavelengths_are_air=obs_wavelengths_are_air, empirical_coeffs=emission_empirical_coeffs,
            curve_sigma_clip=curve_sigma_clip, max_flexure_error=max_flexure_error, plot=False, emission_line_dir=emission_line_dir,
            emission_min_strength=emission_min_strength, emission_min_line_separation=emission_min_line_separation,
        )
    else:
        raise ValueError("flexure_source must be 'telluric_stellar' or 'sky_emission'.")

    fallback_flexure = flexure_curve.attrs.get("Fallback Flexure", np.nan)
    if fallback == "zero":
        fallback_flexure = 0.0
    elif fallback != "anchor":
        fallback_flexure = np.nan

    combined_rv, combined_err, chunk_rvs = _chunked_rvs_for_spectrum(
        obs_wl, obs_flux, temp_wl, temp_flux, obs_airmass, bary_corr, flexure_curve, boundaries, obs_wavelengths_are_air=obs_wavelengths_are_air,
        obs_fluxerr=obs_fluxerr, rv_prior=rv_prior, rv_half_width=rv_half_width, curve_poly_degree=curve_poly_degree, max_telluric_absorption=max_telluric_absorption,
        fallback_flexure=fallback_flexure, max_rv_err=max_rv_err, sigma_clip=sigma_clip, error_floor=error_floor,
        telluric_grid_path=resolved_telluric_grid_path, template_wavelengths_are_air=template_wavelengths_are_air, spectral_resolution=measured_resolution,
        telluric_model_resolution=telluric_model_resolution, chunk_quality=chunk_quality, min_chunk_fit_pixels=min_chunk_fit_pixels,
        min_chunk_chi2_edge_delta=min_chunk_chi2_edge_delta, min_chunk_chi2_second_delta=min_chunk_chi2_second_delta,
        min_chunk_rv_grid_edge_distance=min_chunk_rv_grid_edge_distance, max_chunk_reduced_chi2=max_chunk_reduced_chi2,
        max_chunk_residual_spike_fraction=resolved_max_chunk_residual_spike_fraction, residual_spike_abs_threshold=residual_spike_abs_threshold,
    )
    flexure_curve.attrs["Measured Resolution"] = measured_resolution
    flexure_curve.attrs["Telluric Grid Path"] = str(resolved_telluric_grid_path)
    flexure_curve.attrs["Telluric Model Resolution"] = telluric_model_resolution
    flexure_curve.attrs["Template Resolution"] = template_resolution
    flexure_curve.attrs["Template Degraded"] = bool(degrade_template and np.isfinite(measured_resolution))

    result = RVResult(
        rv=float(combined_rv), rv_error=float(combined_err), barycentric_correction=float(bary_corr), jd_utc=float(obstime.jd), airmass=float(obs_airmass),
        channel=channel, flexure_source=flexure_source, stellar_rv_guess=stellar_rv_guess,
        n_chunks_used=int(np.sum(chunk_rvs["Use in Combined RV"].values)) if "Use in Combined RV" in chunk_rvs else 0,
        n_flexure_anchors=int(np.sum(flexure_curve["Good"].values)) if "Good" in flexure_curve else 0, spectral_resolution=float(measured_resolution),
        telluric_grid_path=str(resolved_telluric_grid_path), spectrum_snr=float(spectrum_snr), spectrum_flux_error=float(spectrum_flux_error),
        low_snr=False, initial_rv_guess=initial_rv_guess, initial_rv_source=initial_rv_source, flexure_curve=flexure_curve,
        chunk_rvs=chunk_rvs, resolution_table=resolution_table,
    )
    if plot_diagnostics:
        from rv_diagnostics import plot_rv_diagnostics

        diagnostic_kwargs = {} if diagnostic_kwargs is None else dict(diagnostic_kwargs)
        result.diagnostic_figures = plot_rv_diagnostics(fits_file, result, temp_wl, diagnostic_template_flux, **diagnostic_kwargs)
    return result if return_details else (result.rv, result.rv_error)


measure_rv_from_reduced_2d = measure_rv


__all__ = [
    "RVResult", "load_example_template", "measure_ngps_rv", "derive_flexure_curve", "measure_rv", "measure_rv_from_reduced_2d",
    "read_reduced_2d_spectrum", "set_data_dir", "derive_telluric_stellar_flexure_curve",
    "derive_sky_emission_flexure_curve", "fit_sky_emission_line", "evaluate_telluric_flexure_curve", "radial_velocity_masked", "combine_chunk_rvs",
    "retrieve_bosz_spectrum", "measure_resolution", "estimate_normalized_flux_error", "rv_bounds_to_windows", "R_H_ALPHA_REGION", "I_CA_TRIPLET_REGION",
    "G_MGB_REGION", "DEFAULT_RV_REGIONS",
]
