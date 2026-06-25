"""Wavelength-dependent flexure curves from telluric absorption lines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from rv_helpers import (
    C_KMS, DEFAULT_TELLURIC_GRID, DEFAULT_TELLURIC_MODEL_RESOLUTION, NGPS_CHANNEL_RANGES, _float_array, _odd_window, convert_air_to_vacuum, doppler_shift,
    least_squares_errors, medfilt_fixed_window_AA, telluric_model_for_airmass,
)


def _preferred_telluric_anchor_windows(channel):
    preferred = {
        "U": [],
        "G": [],
        "R": [(6860, 6950), (7585, 7650)],
        "I": [(7585, 7650), (9300, 9400), (6860, 6950)],
    }
    return preferred.get(str(channel).upper(), [])


def _best_telluric_anchor_window(fit_wl, tell_wl, tell_flux, channel, obs_wavelengths_are_air, min_pixels=20):
    fit_wl = _float_array(fit_wl)
    finite = fit_wl[np.isfinite(fit_wl)]
    if len(finite) == 0:
        return None

    candidates = _preferred_telluric_anchor_windows(channel)
    if not candidates and str(channel).upper() in NGPS_CHANNEL_RANGES:
        candidates = [NGPS_CHANNEL_RANGES[str(channel).upper()]]

    for low, high in candidates:
        low_fit, high_fit = convert_air_to_vacuum(np.array([low, high]))
        tell_m = (tell_wl > low_fit) & (tell_wl < high_fit)
        if np.sum((fit_wl > low_fit) & (fit_wl < high_fit)) < min_pixels or np.sum(tell_m) < 10:
            continue
        absorption = np.clip(1 - tell_flux[tell_m], 0, None)
        depth = np.nanmax(absorption)
        area = np.trapz(absorption, tell_wl[tell_m])
        if depth >= 0.005 and area >= 0.02:
            return float(low_fit), float(high_fit)

    overlap_low = max(np.nanmin(finite), np.nanmin(tell_wl))
    overlap_high = min(np.nanmax(finite), np.nanmax(tell_wl))
    if overlap_high - overlap_low < 40:
        return None

    best_window = None
    best_score = 0.0
    for center in np.arange(overlap_low + 30, overlap_high - 30 + 1, 30):
        low, high = center - 30, center + 30
        obs_m = (fit_wl > low) & (fit_wl < high)
        tell_m = (tell_wl > low) & (tell_wl < high)
        if np.sum(obs_m) < min_pixels or np.sum(tell_m) < 10:
            continue
        absorption = np.clip(1 - tell_flux[tell_m], 0, None)
        depth = np.nanmax(absorption)
        area = np.trapz(absorption, tell_wl[tell_m])
        score = depth * max(area, 0)
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_window = (float(low), float(high))
    return best_window


def standard_telluric_anchor_flexure(
    obs_wl, obs_flux, airmass, channel="R", obs_wavelengths_are_air=True, flex_range=100, telluric_grid_path=DEFAULT_TELLURIC_GRID,
    spectral_resolution=None, telluric_model_resolution=DEFAULT_TELLURIC_MODEL_RESOLUTION,
):
    """Quick telluric flexure estimate used as a starting guess/fallback."""
    fit_wl = convert_air_to_vacuum(obs_wl) if obs_wavelengths_are_air else _float_array(obs_wl)
    tell_wl, tell_flux, _ = telluric_model_for_airmass(
        airmass, telluric_grid_path, output_resolution=spectral_resolution, input_resolution=telluric_model_resolution
    )
    anchor_window = _best_telluric_anchor_window(fit_wl, tell_wl, tell_flux, channel, obs_wavelengths_are_air)
    if anchor_window is None:
        return 0.0
    low, high = anchor_window

    obs_flux = _float_array(obs_flux)
    continuum = medfilt_fixed_window_AA(fit_wl, obs_flux, window_AA=101)
    norm_flux = np.divide(obs_flux, continuum, out=np.full_like(obs_flux, np.nan, dtype=float), where=continuum != 0)
    m = (fit_wl > low) & (fit_wl < high) & np.isfinite(norm_flux) & (norm_flux > 0.05) & (norm_flux < 2.0)
    if np.sum(m) < 20:
        return 0.0

    tell_mask = (tell_wl > low - 50) & (tell_wl < high + 50)
    if np.sum(tell_mask) < 10:
        return 0.0
    shifts = np.arange(-flex_range, flex_range + 0.1, 0.2)
    chi2s = []
    for shift in shifts:
        shifted = doppler_shift(tell_wl[tell_mask], tell_flux[tell_mask], shift)
        model = np.interp(fit_wl[m], tell_wl[tell_mask], shifted, left=1, right=1)
        chi2s.append(np.nansum((model - norm_flux[m]) ** 2))
    return float(shifts[int(np.nanargmin(chi2s))])


def build_telluric_flexure_windows(
    wl_min, wl_max, airmass, window_AA=120, step_AA=60, min_telluric_depth=0.02, min_telluric_area=0.2, telluric_grid_path=DEFAULT_TELLURIC_GRID,
    spectral_resolution=None, telluric_model_resolution=DEFAULT_TELLURIC_MODEL_RESOLUTION,
):
    """Choose wavelength windows that contain enough telluric absorption to fit."""
    tell_wl, tell_flux, grid_airmass = telluric_model_for_airmass(
        airmass, telluric_grid_path, output_resolution=spectral_resolution, input_resolution=telluric_model_resolution
    )
    centers = np.arange(wl_min + window_AA / 2, wl_max - window_AA / 2 + step_AA, step_AA)
    rows = []
    for center in centers:
        low, high = center - window_AA / 2, center + window_AA / 2
        m = (tell_wl > low) & (tell_wl < high)
        if np.sum(m) < 10:
            continue
        absorption = np.clip(1 - tell_flux[m], 0, None)
        depth = np.nanmax(absorption)
        area = np.trapz(absorption, tell_wl[m])
        if depth >= min_telluric_depth and area >= min_telluric_area:
            rows.append(
                {
                    "Window Min": low, "Window Max": high, "Wavelength": center, "Telluric Depth": depth, "Telluric Area": area,
                    "Telluric Grid Airmass": grid_airmass,
                }
            )
    return pd.DataFrame(rows)


def _joint_telluric_stellar_model(
    obs_wl, temp_wl, temp_flux, tell_wl, tell_flux, stellar_rv, bary_corr, flex_corr, telluric_scale, stellar_scale, continuum_coeffs, rv_offset=0,
    model_kind="multiplicative",
):
    margin = 25
    temp_mask = (temp_wl > np.min(obs_wl) - margin) & (temp_wl < np.max(obs_wl) + margin)
    tell_mask = (tell_wl > np.min(obs_wl) - margin) & (tell_wl < np.max(obs_wl) + margin)
    if np.sum(temp_mask) < 5 or np.sum(tell_mask) < 5:
        return np.full_like(obs_wl, np.nan, dtype=float)

    stellar_shift = stellar_rv + flex_corr - bary_corr + rv_offset
    shifted_star = doppler_shift(temp_wl[temp_mask], temp_flux[temp_mask], stellar_shift)
    shifted_tell = doppler_shift(tell_wl[tell_mask], tell_flux[tell_mask], flex_corr)
    star_interp = np.interp(obs_wl, temp_wl[temp_mask], shifted_star, left=1, right=1)
    tell_interp = np.interp(obs_wl, tell_wl[tell_mask], shifted_tell, left=1, right=1)

    x = 2 * (obs_wl - np.mean(obs_wl)) / (np.max(obs_wl) - np.min(obs_wl))
    continuum = np.ones_like(obs_wl, dtype=float)
    for power, coeff in enumerate(continuum_coeffs):
        continuum += coeff * x**power

    star_component = 1 + stellar_scale * (star_interp - 1)
    tell_component = 1 + telluric_scale * (tell_interp - 1)
    if model_kind == "multiplicative":
        spectral_model = star_component * tell_component
    else:
        spectral_model = 1 + stellar_scale * (star_interp - 1) + telluric_scale * (tell_interp - 1)
    return continuum * spectral_model


def quality_filter_telluric_stellar_curve(
    curve, max_flexure_error=35, min_telluric_scale=0.15, max_telluric_scale=2.8, max_stellar_scale=2.3, min_good_telluric_depth=0.08,
    blue_anchor_wavelength_max=6563, min_blue_good_telluric_depth=None, max_blue_flexure_error=None, min_blue_telluric_scale=0.3,
    max_blue_telluric_scale=2.5, max_blue_stellar_scale=1.2, min_blue_anchors=2, max_blue_flexure_deviation=12,
):
    """Mark only well-constrained telluric+stellar windows as flexure anchors."""
    curve = curve.copy()
    if len(curve) == 0:
        curve["Good"] = False
        return curve

    success = curve["Success"].fillna(False).values.astype(bool) if "Success" in curve else np.zeros(len(curve), dtype=bool)
    flexure = curve["Flexure Correction"].values.astype(float) if "Flexure Correction" in curve else np.full(len(curve), np.nan)
    err = curve["Flexure Correction Error"].values.astype(float) if "Flexure Correction Error" in curve else np.full(len(curve), np.nan)
    telluric_scale = curve["Telluric Scale"].values.astype(float) if "Telluric Scale" in curve else np.full(len(curve), np.nan)
    stellar_scale = curve["Stellar Scale"].values.astype(float) if "Stellar Scale" in curve else np.full(len(curve), np.nan)
    depth = curve["Telluric Depth"].values.astype(float) if "Telluric Depth" in curve else np.full(len(curve), np.nan)
    wavelength = curve["Wavelength"].values.astype(float) if "Wavelength" in curve else np.full(len(curve), np.nan)

    base_good = success & np.isfinite(flexure)
    if "Flexure Correction Error" in curve:
        base_good &= np.isfinite(err) & (err < max_flexure_error)
    if "Telluric Scale" in curve:
        base_good &= np.isfinite(telluric_scale) & (telluric_scale > min_telluric_scale) & (telluric_scale < max_telluric_scale)
    if "Stellar Scale" in curve:
        base_good &= np.isfinite(stellar_scale) & (stellar_scale < max_stellar_scale)

    good = base_good.copy()
    if min_good_telluric_depth is not None and "Telluric Depth" in curve:
        good &= np.isfinite(depth) & (depth >= min_good_telluric_depth)

    blue_good = np.zeros(len(curve), dtype=bool)
    if min_blue_good_telluric_depth is not None and max_blue_flexure_error is not None:
        blue_good = base_good.copy()
        blue_good &= np.isfinite(wavelength) & (wavelength < blue_anchor_wavelength_max)
        blue_good &= np.isfinite(depth) & (depth >= min_blue_good_telluric_depth)
        blue_good &= np.isfinite(err) & (err <= max_blue_flexure_error)
        blue_good &= np.isfinite(telluric_scale) & (telluric_scale >= min_blue_telluric_scale) & (telluric_scale <= max_blue_telluric_scale)
        blue_good &= np.isfinite(stellar_scale) & (stellar_scale <= max_blue_stellar_scale)
        if np.sum(blue_good) < min_blue_anchors:
            blue_good[:] = False
        elif max_blue_flexure_deviation is not None:
            median_blue_flexure = np.nanmedian(flexure[blue_good])
            blue_good &= np.abs(flexure - median_blue_flexure) <= max_blue_flexure_deviation
            if np.sum(blue_good) < min_blue_anchors:
                blue_good[:] = False

    standard_good = good.copy()
    good |= blue_good
    curve["Standard Good Anchor"] = standard_good
    curve["Relaxed Blue Anchor"] = blue_good
    curve["Good"] = good
    return curve


def fit_telluric_stellar_flexure_window(
    obs_wl, obs_flux, temp_wl, temp_flux, airmass, stellar_rv, bary_corr, window, flex_guess=0, flex_range=100, rv_offset_range=20, fit_rv_offset=False,
    continuum_deg=1, continuum_window_AA=101, model_kind="multiplicative", min_pixels=40, min_telluric_depth=0.015, sigma_clip=5,
    telluric_grid_path=DEFAULT_TELLURIC_GRID, spectral_resolution=None, telluric_model_resolution=DEFAULT_TELLURIC_MODEL_RESOLUTION,
):
    """Fit one telluric+stellar window and return its flexure correction."""
    obs_wl = _float_array(obs_wl)
    obs_flux = _float_array(obs_flux)
    temp_wl = _float_array(temp_wl)
    temp_flux = _float_array(temp_flux)
    tell_wl, tell_flux, grid_airmass = telluric_model_for_airmass(
        airmass, telluric_grid_path, output_resolution=spectral_resolution, input_resolution=telluric_model_resolution
    )

    low, high = window
    m = (obs_wl > low) & (obs_wl < high) & np.isfinite(obs_wl) & np.isfinite(obs_flux)
    if np.sum(m) < min_pixels:
        return {"Success": False, "Message": "too few observed pixels", "Wavelength": np.mean(window)}

    continuum = medfilt_fixed_window_AA(obs_wl, obs_flux, window_AA=_odd_window(continuum_window_AA))
    norm_flux = np.divide(obs_flux, continuum, out=np.full_like(obs_flux, np.nan, dtype=float), where=continuum != 0)
    fit_wl = obs_wl[m]
    fit_flux = norm_flux[m]
    good_flux = np.isfinite(fit_flux) & (fit_flux > 0.05) & (fit_flux < 2.5)
    fit_wl = fit_wl[good_flux]
    fit_flux = fit_flux[good_flux]
    if len(fit_wl) < min_pixels:
        return {"Success": False, "Message": "too few finite normalized pixels", "Wavelength": np.mean(window)}

    tell_here = np.interp(fit_wl, tell_wl, tell_flux, left=1, right=1)
    tell_depth = np.nanmax(np.clip(1 - tell_here, 0, None))
    if tell_depth < min_telluric_depth:
        return {"Success": False, "Message": "telluric absorption too weak", "Wavelength": np.mean(window), "Telluric Depth": tell_depth}

    resid_scale = 1.4826 * np.nanmedian(np.abs(fit_flux - np.nanmedian(fit_flux)))
    if not np.isfinite(resid_scale) or resid_scale <= 0:
        resid_scale = 0.03

    n_cont = continuum_deg + 1
    p0 = [flex_guess, 1.0, 1.0]
    lo = [flex_guess - flex_range, 0.0, 0.0]
    hi = [flex_guess + flex_range, 3.0, 2.5]
    if fit_rv_offset:
        p0.append(0.0)
        lo.append(-rv_offset_range)
        hi.append(rv_offset_range)
    p0 += [0.0] * n_cont
    lo += [-0.35] * n_cont
    hi += [0.35] * n_cont

    def unpack(params):
        flex_corr = params[0]
        telluric_scale = params[1]
        stellar_scale = params[2]
        offset_idx = 3
        if fit_rv_offset:
            rv_offset = params[3]
            offset_idx = 4
        else:
            rv_offset = 0
        return flex_corr, telluric_scale, stellar_scale, rv_offset, params[offset_idx:]

    def residuals(params, use_mask=None):
        flex_corr, telluric_scale, stellar_scale, rv_offset, continuum_coeffs = unpack(params)
        model = _joint_telluric_stellar_model(
            fit_wl, temp_wl, temp_flux, tell_wl, tell_flux, stellar_rv, bary_corr, flex_corr, telluric_scale, stellar_scale, continuum_coeffs,
            rv_offset=rv_offset, model_kind=model_kind,
        )
        resid = (model - fit_flux) / resid_scale
        if use_mask is not None:
            resid = resid[use_mask]
        return resid[np.isfinite(resid)]

    result = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=800)
    if sigma_clip is not None and result.success:
        flex_corr, telluric_scale, stellar_scale, rv_offset, continuum_coeffs = unpack(result.x)
        model = _joint_telluric_stellar_model(
            fit_wl, temp_wl, temp_flux, tell_wl, tell_flux, stellar_rv, bary_corr, flex_corr, telluric_scale, stellar_scale, continuum_coeffs,
            rv_offset=rv_offset, model_kind=model_kind,
        )
        full_resid = (model - fit_flux) / resid_scale
        finite = np.isfinite(full_resid)
        scatter = 1.4826 * np.nanmedian(np.abs(full_resid[finite] - np.nanmedian(full_resid[finite])))
        if np.isfinite(scatter) and scatter > 0:
            fit_mask = finite & (np.abs(full_resid) < sigma_clip * scatter)
            if np.sum(fit_mask) >= min_pixels:
                result = least_squares(lambda p: residuals(p, fit_mask), result.x, bounds=(lo, hi), max_nfev=800)

    errs = least_squares_errors(result)
    flex_corr, telluric_scale, stellar_scale, rv_offset, continuum_coeffs = unpack(result.x)
    near_edge = abs(flex_corr - flex_guess) > 0.95 * flex_range

    return {
        "Success": bool(result.success and not near_edge), "Message": result.message, "Wavelength": np.mean(window), "Window Min": low, "Window Max": high,
        "Flexure Correction": flex_corr, "Flexure Correction Error": errs[0] if len(errs) > 0 else np.nan, "Telluric Scale": telluric_scale,
        "Stellar Scale": stellar_scale, "RV Offset": rv_offset, "Telluric Depth": tell_depth, "N Pixels": len(fit_wl), "Telluric Grid Airmass": grid_airmass,
        "Cost": result.cost,
    }


def derive_telluric_stellar_flexure_curve(
    obs_wl, obs_flux, temp_wl, temp_flux, airmass, stellar_rv, bary_corr, channel="R", windows=None, flex_guess=None, flex_range=100, window_AA=120, step_AA=60,
    min_telluric_depth=0.02, min_telluric_area=0.2, obs_wavelengths_are_air=True, model_kind="multiplicative", fit_rv_offset=False, sigma_clip=5, curve_sigma_clip=4,
    max_flexure_error=35, min_telluric_scale=0.15, max_telluric_scale=2.8, max_stellar_scale=2.3, min_good_telluric_depth=0.08,
    blue_anchor_wavelength_max=6563, min_blue_good_telluric_depth=None, max_blue_flexure_error=None, min_blue_telluric_scale=0.3,
    max_blue_telluric_scale=2.5, max_blue_stellar_scale=1.2, min_blue_anchors=2, max_blue_flexure_deviation=12,
    telluric_grid_path=DEFAULT_TELLURIC_GRID, spectral_resolution=None, telluric_model_resolution=DEFAULT_TELLURIC_MODEL_RESOLUTION,
):
    """Fit flexure anchors over one spectrum and return a wavelength-flexure table."""
    obs_wl = _float_array(obs_wl)
    obs_flux = _float_array(obs_flux)
    fit_wl = convert_air_to_vacuum(obs_wl) if obs_wavelengths_are_air else obs_wl

    if flex_guess is None:
        flex_guess = standard_telluric_anchor_flexure(
            obs_wl, obs_flux, airmass, channel=channel, obs_wavelengths_are_air=obs_wavelengths_are_air, telluric_grid_path=telluric_grid_path,
            spectral_resolution=spectral_resolution, telluric_model_resolution=telluric_model_resolution,
        )

    if windows is None:
        windows = build_telluric_flexure_windows(
            np.nanmin(fit_wl), np.nanmax(fit_wl), airmass, window_AA=window_AA, step_AA=step_AA, min_telluric_depth=min_telluric_depth,
            min_telluric_area=min_telluric_area, telluric_grid_path=telluric_grid_path, spectral_resolution=spectral_resolution,
            telluric_model_resolution=telluric_model_resolution,
        )
    elif isinstance(windows, pd.DataFrame):
        windows = windows.copy()
    else:
        windows = pd.DataFrame([{"Window Min": w[0], "Window Max": w[1], "Wavelength": np.mean(w)} for w in windows])

    rows = []
    this_guess = flex_guess
    for _, row in windows.iterrows():
        result = fit_telluric_stellar_flexure_window(
            fit_wl, obs_flux, temp_wl, temp_flux, airmass, stellar_rv, bary_corr, (row["Window Min"], row["Window Max"]), flex_guess=this_guess,
            flex_range=flex_range, fit_rv_offset=fit_rv_offset, model_kind=model_kind, min_telluric_depth=min_telluric_depth, sigma_clip=sigma_clip,
            telluric_grid_path=telluric_grid_path, spectral_resolution=spectral_resolution, telluric_model_resolution=telluric_model_resolution,
        )
        if result.get("Success"):
            this_guess = result["Flexure Correction"]
        rows.append(result)

    curve = pd.DataFrame(rows)
    for col in ("Wavelength", "Flexure Correction", "Flexure Correction Error", "Success"):
        if col not in curve:
            curve[col] = np.nan
    curve["Success"] = curve["Success"].fillna(False).values.astype(bool)
    curve = quality_filter_telluric_stellar_curve(
        curve, max_flexure_error=max_flexure_error, min_telluric_scale=min_telluric_scale, max_telluric_scale=max_telluric_scale,
        max_stellar_scale=max_stellar_scale, min_good_telluric_depth=min_good_telluric_depth, blue_anchor_wavelength_max=blue_anchor_wavelength_max,
        min_blue_good_telluric_depth=min_blue_good_telluric_depth, max_blue_flexure_error=max_blue_flexure_error,
        min_blue_telluric_scale=min_blue_telluric_scale, max_blue_telluric_scale=max_blue_telluric_scale,
        max_blue_stellar_scale=max_blue_stellar_scale, min_blue_anchors=min_blue_anchors, max_blue_flexure_deviation=max_blue_flexure_deviation,
    )

    good_vals = curve.loc[curve["Good"], "Flexure Correction"].values.astype(float) if len(curve) else []
    if len(good_vals) >= 3 and curve_sigma_clip is not None:
        med = np.nanmedian(good_vals)
        scatter = 1.4826 * np.nanmedian(np.abs(good_vals - med))
        scatter = max(scatter, 5)
        curve.loc[curve["Good"], "Good"] = np.abs(good_vals - med) < curve_sigma_clip * scatter
        if "Standard Good Anchor" in curve:
            curve["Standard Good Anchor"] &= curve["Good"]
        if "Relaxed Blue Anchor" in curve:
            curve["Relaxed Blue Anchor"] &= curve["Good"]

    curve.attrs["Fallback Flexure"] = flex_guess
    curve.attrs["Channel"] = channel
    curve.attrs["Airmass"] = airmass
    curve.attrs["Model Kind"] = model_kind
    curve.attrs["Telluric Grid Path"] = str(telluric_grid_path)
    return curve


def evaluate_telluric_flexure_curve(wavelength, curve, fallback_flexure=np.nan, poly_degree=None):
    """Evaluate a flexure curve by polynomial fit, interpolation, or fallback."""
    scalar = np.isscalar(wavelength)
    wavelength = np.atleast_1d(np.asarray(wavelength, dtype=float))
    required = {"Good", "Wavelength", "Flexure Correction"}
    if curve is None or len(curve) == 0 or not required.issubset(set(curve.columns)):
        vals = np.full_like(wavelength, fallback_flexure, dtype=float)
        return vals[0] if scalar else vals

    good = curve["Good"].values.astype(bool) & np.isfinite(curve["Wavelength"].values) & np.isfinite(curve["Flexure Correction"].values)
    x = curve["Wavelength"].values[good].astype(float)
    y = curve["Flexure Correction"].values[good].astype(float)
    if len(x) == 0:
        vals = np.full_like(wavelength, fallback_flexure, dtype=float)
    elif len(x) == 1:
        vals = np.full_like(wavelength, y[0], dtype=float)
    elif poly_degree is not None and len(x) > poly_degree:
        vals = np.poly1d(np.polyfit(x, y, poly_degree))(wavelength)
    else:
        order = np.argsort(x)
        vals = np.interp(wavelength, x[order], y[order], left=y[order][0], right=y[order][-1])
    return vals[0] if scalar else vals


def telluric_good_pixel_mask(
    obs_wl, airmass, flex_corr=0, max_telluric_absorption=0.03, obs_wavelengths_are_air=True, edge_AA=10, telluric_grid_path=DEFAULT_TELLURIC_GRID,
    spectral_resolution=None, telluric_model_resolution=DEFAULT_TELLURIC_MODEL_RESOLUTION,
):
    """Mask pixels where the shifted telluric model is deeper than requested."""
    obs_wl = _float_array(obs_wl)
    tell_wl, tell_flux, _ = telluric_model_for_airmass(
        airmass, telluric_grid_path, output_resolution=spectral_resolution, input_resolution=telluric_model_resolution
    )
    compare_wl = convert_air_to_vacuum(obs_wl) if obs_wavelengths_are_air else obs_wl
    shifted_tell = doppler_shift(tell_wl, tell_flux, flex_corr)
    tell_interp = np.interp(compare_wl, tell_wl, shifted_tell, left=1, right=1)
    good = tell_interp > (1 - max_telluric_absorption)
    good &= (obs_wl > np.nanmin(obs_wl) + edge_AA) & (obs_wl < np.nanmax(obs_wl) - edge_AA)
    return good


__all__ = ["standard_telluric_anchor_flexure", "derive_telluric_stellar_flexure_curve", "evaluate_telluric_flexure_curve", "telluric_good_pixel_mask"]
