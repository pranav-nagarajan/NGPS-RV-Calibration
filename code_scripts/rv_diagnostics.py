"""Diagnostic plots for NGPS wavelength-dependent RV measurements."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from astropy.io import fits

from absorption_flexure import evaluate_telluric_flexure_curve, telluric_good_pixel_mask
from rv_helpers import (
    _float_array, _read_channel, convert_air_to_vacuum, degrade_spectrum_resolution, doppler_shift, estimate_normalized_flux_error, medfilt_fixed_window_AA,
    load_emission_line_catalog, read_reduced_2d_spectrum, read_sky_model_at_trace, telluric_model_for_airmass,
)
from wavelength_dependent_rvs import _stellar_wavelengths_for_rv


def _curve_from_result(rv_result_or_curve):
    return getattr(rv_result_or_curve, "flexure_curve", rv_result_or_curve)


def _relaxed_blue_anchor_mask(curve, good, *, blue_anchor_wavelength_max=6563, standard_depth=0.08):
    if "Relaxed Blue Anchor" in curve:
        return good & curve["Relaxed Blue Anchor"].fillna(False).astype(bool).values
    if not {"Wavelength", "Telluric Depth"}.issubset(curve.columns):
        return np.zeros(len(curve), dtype=bool)
    wavelength = curve["Wavelength"].values.astype(float)
    depth = curve["Telluric Depth"].values.astype(float)
    return good & np.isfinite(wavelength) & (wavelength < blue_anchor_wavelength_max) & np.isfinite(depth) & (depth < standard_depth)


def _weak_blue_candidate_mask(curve, *, blue_anchor_wavelength_max=6563, min_blue_depth=0.02, standard_depth=0.08):
    if not {"Wavelength", "Telluric Depth"}.issubset(curve.columns):
        return np.zeros(len(curve), dtype=bool)
    wavelength = curve["Wavelength"].values.astype(float)
    depth = curve["Telluric Depth"].values.astype(float)
    candidate = np.isfinite(wavelength) & (wavelength < blue_anchor_wavelength_max)
    candidate &= np.isfinite(depth) & (depth >= min_blue_depth) & (depth < standard_depth)
    return candidate


def _chunk_rows(rv_result, chunks="used", max_chunks=None):
    table = rv_result.chunk_rvs.copy()
    if chunks == "used" and "Use in Combined RV" in table:
        table = table[table["Use in Combined RV"].fillna(False).astype(bool)]
    elif chunks not in {"used", "all"}:
        table = table.iloc[list(chunks)]
    if max_chunks is not None:
        table = table.head(int(max_chunks))
    return table.reset_index(drop=True)


def _normalized_observed_spectrum(wavelength, flux, continuum_window_AA=21):
    continuum = medfilt_fixed_window_AA(wavelength, flux, window_AA=continuum_window_AA)
    norm_flux = np.divide(flux, continuum, out=np.full_like(flux, np.nan, dtype=float), where=continuum != 0)
    return norm_flux, continuum


def _normalize_target_text(value):
    text = value.decode() if isinstance(value, bytes) else str(value)
    text = text.strip()
    return f"GESJ{text}" if text[:1].isdigit() else text


def find_reduced_spectrum_file(spectra_dir, *, target=None, channel=None, slit_width=None, slit_tolerance=0.03):
    """Find a reduced 2D spectrum matching target, channel, and slit width."""
    spectra_dir = Path(spectra_dir).expanduser()
    target_norm = _normalize_target_text(target).lower() if target is not None else None
    channel = channel.upper() if channel is not None else None
    matches = []
    for path in sorted(spectra_dir.glob("*.fits")):
        try:
            with fits.open(path) as hdul:
                header = hdul[0].header
                file_target = _normalize_target_text(header.get("NAME", header.get("OBJECT", path.stem)))
                file_channel = _read_channel(header, path)
                file_slit = float(header.get("SLITW", np.nan))
        except Exception:
            continue
        if target_norm is not None and file_target.lower() != target_norm:
            continue
        if channel is not None and file_channel != channel:
            continue
        if slit_width is not None and np.isfinite(file_slit) and abs(file_slit - float(slit_width)) > float(slit_tolerance):
            continue
        matches.append(path)
    if len(matches) == 0:
        raise FileNotFoundError("No matching reduced 2D spectrum found.")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches[:5])
        raise ValueError(f"Found {len(matches)} matching spectra; please narrow the selection. First matches: {names}")
    return matches[0]


def plot_reduced_1d_spectrum(
    spectra_dir_or_file, *, target=None, channel=None, slit_width=None, slit_tolerance=0.03, aperture_half_width=2, trace_column=None, trace_y=None,
    normalize=False, continuum_window_AA=21, wavelength_lims=None, ax=None,
):
    """Plot the extracted 1D reduced spectrum for one object/night/channel/slit setup."""
    import matplotlib.pyplot as plt

    path = Path(spectra_dir_or_file).expanduser()
    if path.is_dir():
        path = find_reduced_spectrum_file(path, target=target, channel=channel, slit_width=slit_width, slit_tolerance=slit_tolerance)
    wavelength, flux, header = read_reduced_2d_spectrum(path, trace_column=trace_column, aperture_half_width=aperture_half_width, trace_y=trace_y)
    wavelength = _float_array(wavelength)
    flux = _float_array(flux)
    plot_flux = flux
    ylabel = "Flux"
    if normalize:
        plot_flux, _ = _normalized_observed_spectrum(wavelength, flux, continuum_window_AA=continuum_window_AA)
        ylabel = "Normalized flux"
    mask = np.isfinite(wavelength) & np.isfinite(plot_flux)
    if wavelength_lims is not None:
        low, high = np.asarray(wavelength_lims, dtype=float)
        mask &= (wavelength >= low) & (wavelength <= high)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    else:
        fig = ax.figure
    ax.plot(wavelength[mask], plot_flux[mask], color="black", linewidth=0.8)
    title_target = _normalize_target_text(header.get("NAME", header.get("OBJECT", path.stem)))
    title_channel = _read_channel(header, path, wavelength=wavelength, default=channel or "")
    title_slit = header.get("SLITW", slit_width)
    ax.set_title(f"{title_target} {title_channel} slit={title_slit}\"")
    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel(ylabel)
    return fig, ax, path


def _prepared_template(rv_result, template_wavelength, template_flux):
    temp_wl = _float_array(template_wavelength)
    temp_flux = _float_array(template_flux)
    attrs = getattr(rv_result, "flexure_curve", {}).attrs if hasattr(getattr(rv_result, "flexure_curve", None), "attrs") else {}
    if attrs.get("Template Degraded", False):
        input_resolution = attrs.get("Template Resolution", 5000)
        temp_flux = degrade_spectrum_resolution(temp_wl, temp_flux, output_resolution=rv_result.spectral_resolution, input_resolution=input_resolution)
    return temp_wl, temp_flux


def _shared_continuum_model(x, observed_flux, model, mask, degree=1):
    good = mask & np.isfinite(x) & np.isfinite(observed_flux) & np.isfinite(model) & (model != 0)
    if degree is None or np.sum(good) < max(5, degree + 2):
        scale = np.nanmedian(observed_flux[good] / model[good]) if np.any(good) else np.nanmedian(observed_flux[np.isfinite(observed_flux)])
        return np.full_like(observed_flux, scale if np.isfinite(scale) else 1.0, dtype=float)
    xx = 2 * (x - np.nanmean(x[good])) / max(np.nanmax(x[good]) - np.nanmin(x[good]), 1e-6)
    coeffs = np.polyfit(xx[good], observed_flux[good] / model[good], int(degree))
    continuum = np.polyval(coeffs, xx)
    bad = ~np.isfinite(continuum) | (continuum == 0)
    if np.any(bad):
        fallback = np.nanmedian(continuum[~bad]) if np.any(~bad) else np.nanmedian(observed_flux[good] / model[good])
        continuum[bad] = fallback if np.isfinite(fallback) else 1.0
    return continuum


def chunk_model_diagnostic(
    fits_file, rv_result, template_wavelength, template_flux, chunk, *, obs_wavelengths_are_air=True, template_wavelengths_are_air=True,
    aperture_half_width=2, trace_column=None, trace_y=None, max_telluric_absorption=0.03, continuum_window_AA=21, continuum_degree="rv",
    normalize_model_for_display=False,
):
    """Return observed spectrum, multiplicative model, mask, and metadata for one RV chunk."""
    obs_wl, obs_flux, _ = read_reduced_2d_spectrum(fits_file, trace_column=trace_column, aperture_half_width=aperture_half_width, trace_y=trace_y)
    obs_wl = _float_array(obs_wl)
    obs_flux = _float_array(obs_flux)
    temp_wl, temp_flux = _prepared_template(rv_result, template_wavelength, template_flux)
    rv_wl, rv_temp_wl = _stellar_wavelengths_for_rv(
        obs_wl, temp_wl, obs_wavelengths_are_air=obs_wavelengths_are_air, template_wavelengths_are_air=template_wavelengths_are_air
    )

    low = float(chunk["Wavelength Min"])
    high = float(chunk["Wavelength Max"])
    flex_corr = float(chunk["Flexure Correction"])
    corrected_rv = float(chunk["RV"]) if np.isfinite(chunk["RV"]) else float(rv_result.rv)
    stellar_shift = corrected_rv - rv_result.barycentric_correction + flex_corr

    obs_bool = (obs_wl > low - 10) & (obs_wl < high + 10)
    temp_mask = (rv_temp_wl > np.nanmin(rv_wl[obs_bool]) - 10) & (rv_temp_wl < np.nanmax(rv_wl[obs_bool]) + 10)
    shifted_star = doppler_shift(rv_temp_wl[temp_mask], temp_flux[temp_mask], stellar_shift)
    stellar_model = np.interp(rv_wl, rv_temp_wl[temp_mask], shifted_star, left=1, right=1)

    telluric_resolution = getattr(rv_result.flexure_curve, "attrs", {}).get("Telluric Model Resolution", None)
    tell_wl, tell_flux, _ = telluric_model_for_airmass(
        rv_result.airmass, rv_result.telluric_grid_path, output_resolution=rv_result.spectral_resolution, input_resolution=telluric_resolution
    )
    compare_wl = convert_air_to_vacuum(obs_wl) if obs_wavelengths_are_air else obs_wl
    shifted_tell = doppler_shift(tell_wl, tell_flux, flex_corr)
    telluric_model = np.interp(compare_wl, tell_wl, shifted_tell, left=1, right=1)
    multiplicative_model = stellar_model * telluric_model

    fit_mask = telluric_good_pixel_mask(
        obs_wl, rv_result.airmass, flex_corr=flex_corr, max_telluric_absorption=max_telluric_absorption, obs_wavelengths_are_air=obs_wavelengths_are_air,
        telluric_grid_path=rv_result.telluric_grid_path, spectral_resolution=rv_result.spectral_resolution, telluric_model_resolution=telluric_resolution,
    )
    fit_mask &= (obs_wl > low) & (obs_wl < high) & np.isfinite(obs_flux)
    if continuum_degree in {"rv", "median", "filter"}:
        continuum = medfilt_fixed_window_AA(obs_wl, obs_flux, window_AA=continuum_window_AA)
    else:
        continuum = _shared_continuum_model(rv_wl, obs_flux, multiplicative_model, fit_mask, degree=continuum_degree)
    norm_flux = np.divide(obs_flux, continuum, out=np.full_like(obs_flux, np.nan, dtype=float), where=continuum != 0)
    display_model = multiplicative_model
    if normalize_model_for_display:
        model_continuum = medfilt_fixed_window_AA(rv_wl, multiplicative_model, window_AA=continuum_window_AA)
        display_model = np.divide(
            multiplicative_model,
            model_continuum,
            out=np.full_like(multiplicative_model, np.nan, dtype=float),
            where=model_continuum != 0,
        )
    return {
        "wavelength": rv_wl, "observed": norm_flux, "stellar_model": stellar_model, "telluric_model": telluric_model,
        "multiplicative_model": display_model, "raw_multiplicative_model": multiplicative_model,
        "fit_mask": fit_mask, "chunk_mask": obs_bool, "rv": corrected_rv, "stellar_shift": stellar_shift,
        "flexure_correction": flex_corr, "chunk": chunk,
    }


def chunk_chi2_curve(
    fits_file, rv_result, template_wavelength, template_flux, chunk, *, rv_grid=None, rv_half_width=50, rv_step=0.2, obs_wavelengths_are_air=True,
    template_wavelengths_are_air=True, aperture_half_width=2, trace_column=None, trace_y=None, max_telluric_absorption=0.03, continuum_window_AA=21,
):
    """Recompute the RV-fit chi-squared curve for one chunk."""
    obs_wl, obs_flux, _ = read_reduced_2d_spectrum(fits_file, trace_column=trace_column, aperture_half_width=aperture_half_width, trace_y=trace_y)
    obs_wl = _float_array(obs_wl)
    obs_flux = _float_array(obs_flux)
    norm_flux, _ = _normalized_observed_spectrum(obs_wl, obs_flux, continuum_window_AA=continuum_window_AA)
    temp_wl, temp_flux = _prepared_template(rv_result, template_wavelength, template_flux)
    rv_wl, rv_temp_wl = _stellar_wavelengths_for_rv(
        obs_wl, temp_wl, obs_wavelengths_are_air=obs_wavelengths_are_air, template_wavelengths_are_air=template_wavelengths_are_air
    )

    low = float(chunk["Wavelength Min"])
    high = float(chunk["Wavelength Max"])
    flex_corr = float(chunk["Flexure Correction"])
    center_rv = float(chunk["RV"]) if np.isfinite(chunk["RV"]) else float(rv_result.rv)
    if rv_grid is None:
        rv_grid = np.arange(center_rv - rv_half_width, center_rv + rv_half_width + 0.5 * rv_step, rv_step)
    rv_grid = np.asarray(rv_grid, dtype=float)

    telluric_resolution = getattr(rv_result.flexure_curve, "attrs", {}).get("Telluric Model Resolution", None)
    fit_mask = telluric_good_pixel_mask(
        obs_wl, rv_result.airmass, flex_corr=flex_corr, max_telluric_absorption=max_telluric_absorption, obs_wavelengths_are_air=obs_wavelengths_are_air,
        telluric_grid_path=rv_result.telluric_grid_path, spectral_resolution=rv_result.spectral_resolution, telluric_model_resolution=telluric_resolution,
    )
    fit_mask &= (obs_wl > low) & (obs_wl < high) & np.isfinite(norm_flux) & (norm_flux > 0.05) & (norm_flux < 2.0)
    flux_error = estimate_normalized_flux_error(obs_wl, obs_flux, mask=fit_mask, continuum_lims=(low, high), continuum_window_AA=continuum_window_AA)
    norm_err = np.full_like(norm_flux, flux_error, dtype=float)

    chi2 = np.full_like(rv_grid, np.nan, dtype=float)
    if np.sum(fit_mask) < 20:
        return rv_grid, chi2
    temp_mask = (rv_temp_wl > np.nanmin(rv_wl[fit_mask]) - 10) & (rv_temp_wl < np.nanmax(rv_wl[fit_mask]) + 10)
    if np.sum(temp_mask) < 5:
        return rv_grid, chi2

    for i, corrected_rv in enumerate(rv_grid):
        stellar_shift = corrected_rv - rv_result.barycentric_correction + flex_corr
        shifted_template = doppler_shift(rv_temp_wl[temp_mask], temp_flux[temp_mask], stellar_shift)
        model = np.interp(rv_wl, rv_temp_wl[temp_mask], shifted_template, left=1, right=1)
        chi2[i] = np.nansum(((model - norm_flux) ** 2 / norm_err**2)[fit_mask])
    return rv_grid, chi2


def plot_flexure_curve(
    rv_result_or_curve, *, ax=None, poly_degree=None, show_bad=True, show_evaluated=True, chunk_table=None, show_chunk_evaluation=True,
    show_halpha=True, show_blue_candidates=True, blue_anchor_wavelength_max=6563, min_blue_anchor_depth=0.02, standard_anchor_depth=0.08,
    simple=False,
):
    """Plot flexure correction as a function of wavelength."""
    import matplotlib.pyplot as plt

    curve = _curve_from_result(rv_result_or_curve)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if simple:
        show_bad = False
        show_blue_candidates = False
        show_halpha = False
        show_chunk_evaluation = False

    good = curve["Good"].fillna(False).astype(bool).values if "Good" in curve else np.ones(len(curve), dtype=bool)
    relaxed_blue = _relaxed_blue_anchor_mask(
        curve, good, blue_anchor_wavelength_max=blue_anchor_wavelength_max, standard_depth=standard_anchor_depth
    )
    weak_blue_candidate = _weak_blue_candidate_mask(
        curve, blue_anchor_wavelength_max=blue_anchor_wavelength_max, min_blue_depth=min_blue_anchor_depth, standard_depth=standard_anchor_depth
    )
    rejected_blue_candidate = weak_blue_candidate & ~relaxed_blue & ~good
    standard_good = good if simple else good & ~relaxed_blue
    rejected = ~good
    if show_blue_candidates:
        rejected &= ~rejected_blue_candidate
    if show_bad and np.any(rejected):
        ax.scatter(curve.loc[rejected, "Wavelength"], curve.loc[rejected, "Flexure Correction"], color="0.75", s=18, label="Rejected")
    if show_bad and show_blue_candidates and np.any(rejected_blue_candidate):
        ax.scatter(
            curve.loc[rejected_blue_candidate, "Wavelength"], curve.loc[rejected_blue_candidate, "Flexure Correction"], facecolors="none",
            edgecolors="tab:green", marker="D", s=30, label="Rejected blue candidate",
        )
    if np.any(standard_good):
        ax.scatter(curve.loc[standard_good, "Wavelength"], curve.loc[standard_good, "Flexure Correction"], color="tab:blue", s=24, label="Anchors")
    if not simple and np.any(relaxed_blue):
        ax.scatter(
            curve.loc[relaxed_blue, "Wavelength"], curve.loc[relaxed_blue, "Flexure Correction"], color="tab:green", marker="D", s=30,
            label="Relaxed blue anchor",
        )
    good_min = np.nanmin(curve.loc[good, "Wavelength"]) if np.any(good) else np.nan
    good_max = np.nanmax(curve.loc[good, "Wavelength"]) if np.any(good) else np.nan
    if show_evaluated and np.sum(good) >= 2:
        x = np.linspace(good_min, good_max, 300)
        y = evaluate_telluric_flexure_curve(x, curve, fallback_flexure=np.nan, poly_degree=poly_degree)
        ax.plot(x, y, color="black", linewidth=1.5, label="Flexure curve")
    if chunk_table is None and hasattr(rv_result_or_curve, "chunk_rvs"):
        chunk_table = rv_result_or_curve.chunk_rvs
    if chunk_table is not None and len(chunk_table):
        ax.scatter(chunk_table["Wavelength Mid"], chunk_table["Flexure Correction"], marker="x", color="tab:orange", label="RV chunks")
        if show_chunk_evaluation and np.sum(good) >= 1:
            chunk_min = np.nanmin(chunk_table["Wavelength Mid"].values.astype(float))
            chunk_max = np.nanmax(chunk_table["Wavelength Mid"].values.astype(float))
            x = np.linspace(chunk_min, chunk_max, 300)
            y = evaluate_telluric_flexure_curve(x, curve, fallback_flexure=np.nan, poly_degree=poly_degree)
            inside = (x >= good_min) & (x <= good_max)
            if np.any(inside):
                ax.plot(x[inside], y[inside], color="tab:orange", linewidth=1.2, alpha=0.8, label="Chunk-range evaluation")
            if np.any(~inside):
                ax.plot(x[~inside], y[~inside], color="tab:orange", linewidth=1.2, linestyle="--", alpha=0.8, label="Extrapolated/clamped evaluation")

    x_values = []
    if len(curve) and "Wavelength" in curve:
        x_values.extend(curve["Wavelength"].values.astype(float))
    if chunk_table is not None and len(chunk_table) and "Wavelength Mid" in chunk_table:
        x_values.extend(chunk_table["Wavelength Mid"].values.astype(float))
    x_values = np.asarray(x_values, dtype=float)
    finite_x = x_values[np.isfinite(x_values)]
    if show_halpha and len(finite_x) and np.nanmin(finite_x) < blue_anchor_wavelength_max < np.nanmax(finite_x):
        ax.axvline(blue_anchor_wavelength_max, color="0.45", linestyle=":", linewidth=1, label="H-alpha")

    ax.set_xlabel("Wavelength [A]")
    ax.set_ylabel("Flexure correction [km/s]")
    ax.legend()
    return ax


def plot_emission_line_fits(
    fits_file, rv_result_or_curve=None, *, channel=None, emission_lines=None, line_selection="good", max_lines=12, window_size=5, trace_column=None,
    trace_y=None, emission_line_dir="emission_lines", emission_min_strength="default", emission_min_line_separation="default", axes=None,
):
    """Plot sky-emission line data and single-Gaussian fits used for flexure."""
    import matplotlib.pyplot as plt
    from emission_flexure import _gaussian_plus_line, _line_wavelengths_from_catalog, fit_sky_emission_line

    sky_wl, sky_flux, header, _ = read_sky_model_at_trace(fits_file, trace_column=trace_column, trace_y=trace_y)
    if channel is None:
        channel = _read_channel(header, fits_file, sky_wl)

    curve = _curve_from_result(rv_result_or_curve) if rv_result_or_curve is not None else None
    if curve is not None and hasattr(curve, "columns") and "Air Wavelength [A]" in curve:
        rows = curve.copy()
        if line_selection == "good" and "Good" in rows:
            rows = rows[rows["Good"].fillna(False).astype(bool)]
        elif line_selection in {"bad", "rejected"} and "Good" in rows:
            rows = rows[~rows["Good"].fillna(False).astype(bool)]
        elif line_selection != "all":
            raise ValueError("line_selection must be 'good', 'bad', 'rejected', or 'all'.")
        line_wavelengths = rows["Air Wavelength [A]"].values.astype(float)
    else:
        if emission_lines is None:
            emission_lines = load_emission_line_catalog(
                emission_line_dir=emission_line_dir, channel=channel, min_strength=emission_min_strength,
                min_separation=emission_min_line_separation,
            )
        line_wavelengths = _line_wavelengths_from_catalog(emission_lines, channel=channel, emission_line_dir=emission_line_dir)
        rows = None

    finite = np.isfinite(line_wavelengths)
    line_wavelengths = line_wavelengths[finite]
    line_wavelengths = line_wavelengths[(line_wavelengths > np.nanmin(sky_wl)) & (line_wavelengths < np.nanmax(sky_wl))]
    if max_lines is not None:
        line_wavelengths = line_wavelengths[: int(max_lines)]

    n_lines = max(len(line_wavelengths), 1)
    if axes is None:
        fig, axes = plt.subplots(n_lines, 1, figsize=(7, 2.4 * n_lines), squeeze=False, constrained_layout=True)
        axes = axes[:, 0]
    else:
        fig = axes[0].figure if isinstance(axes, (list, tuple, np.ndarray)) else axes.figure
        axes = np.atleast_1d(axes)

    if len(line_wavelengths) == 0:
        axes[0].text(0.5, 0.5, "No emission lines selected", ha="center", va="center", transform=axes[0].transAxes)
        return fig, axes

    for ax, rest_wavelength in zip(axes, line_wavelengths):
        m = (sky_wl > rest_wavelength - window_size) & (sky_wl < rest_wavelength + window_size) & np.isfinite(sky_wl) & np.isfinite(sky_flux)
        ax.plot(sky_wl[m], sky_flux[m], color="black", linewidth=1, label="Sky model")
        ax.axvline(rest_wavelength, color="0.55", linestyle=":", linewidth=1, label="Catalog")
        annotation = "fit failed"
        try:
            params, flex_corr, flex_err = fit_sky_emission_line(sky_wl, sky_flux, rest_wavelength, window_size=window_size)
            x_model = np.linspace(rest_wavelength - window_size, rest_wavelength + window_size, 300)
            ax.plot(x_model, _gaussian_plus_line(params, x_model), color="tab:red", linewidth=1.2, label="Gaussian fit")
            ax.axvline(params[1], color="tab:red", linestyle="--", linewidth=1, label="Fit mean")
            if np.isfinite(flex_err):
                annotation = f"flexure =\n{flex_corr:.1f} +/- {flex_err:.1f} km/s"
            else:
                annotation = f"flexure =\n{flex_corr:.1f} km/s"
        except Exception:
            pass
        ax.set_title("")
        ax.text(
            0.04,
            0.94,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.0},
        )
        ax.set_ylabel("Sky flux")
        ax.legend(loc="best", fontsize=8)
    axes[min(len(line_wavelengths), len(axes)) - 1].set_xlabel("Wavelength [A]")
    return fig, axes


def plot_chunk_model_overlays(fits_file, rv_result, template_wavelength, template_flux, *, chunks="used", max_chunks=None, axes=None, **diagnostic_kwargs):
    """Plot normalized observed spectra with the multiplicative stellar*telluric model for RV chunks."""
    import matplotlib.pyplot as plt

    rows = _chunk_rows(rv_result, chunks=chunks, max_chunks=max_chunks)
    diagnostic_kwargs.setdefault("normalize_model_for_display", True)
    if axes is None:
        fig, axes = plt.subplots(max(len(rows), 1), 1, figsize=(8, 3.0 * max(len(rows), 1)), squeeze=False, constrained_layout=True)
        axes = axes[:, 0]
    else:
        fig = axes[0].figure if isinstance(axes, (list, tuple, np.ndarray)) else axes.figure
        axes = np.atleast_1d(axes)

    for ax, (_, chunk) in zip(axes, rows.iterrows()):
        diag = chunk_model_diagnostic(fits_file, rv_result, template_wavelength, template_flux, chunk, **diagnostic_kwargs)
        m = diag["chunk_mask"]
        ax.plot(diag["wavelength"][m], diag["observed"][m], color="black", linewidth=1, label="Observed")
        ax.plot(diag["wavelength"][m], diag["multiplicative_model"][m], color="tab:red", linewidth=1.2, label="Stellar * telluric")
        bad = m & ~diag["fit_mask"]
        if np.any(bad):
            ax.scatter(diag["wavelength"][bad], diag["observed"][bad], color="0.75", s=6, label="Masked")
        ax.set_title(f"{chunk['Wavelength Min']:.0f}-{chunk['Wavelength Max']:.0f} A, RV={diag['rv']:.2f} km/s")
        ax.set_ylabel("Normalized flux")
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Wavelength [A]")
    fig.set_constrained_layout_pads(hspace=0.12)
    return fig, axes


def plot_chunk_chi2_curves(fits_file, rv_result, template_wavelength, template_flux, *, chunks="used", max_chunks=None, axes=None, **chi2_kwargs):
    """Plot chi-squared as a function of corrected RV for each selected chunk."""
    import matplotlib.pyplot as plt

    rows = _chunk_rows(rv_result, chunks=chunks, max_chunks=max_chunks)
    if axes is None:
        fig, axes = plt.subplots(max(len(rows), 1), 1, figsize=(8, 3.0 * max(len(rows), 1)), squeeze=False, constrained_layout=True)
        axes = axes[:, 0]
    else:
        fig = axes[0].figure if isinstance(axes, (list, tuple, np.ndarray)) else axes.figure
        axes = np.atleast_1d(axes)

    for ax, (_, chunk) in zip(axes, rows.iterrows()):
        rv_grid, chi2 = chunk_chi2_curve(fits_file, rv_result, template_wavelength, template_flux, chunk, **chi2_kwargs)
        ax.plot(rv_grid, chi2 - np.nanmin(chi2), color="tab:blue")
        if np.isfinite(chunk["RV"]):
            ax.axvline(chunk["RV"], color="black", linestyle="--", linewidth=1)
        ax.set_title(f"{chunk['Wavelength Min']:.0f}-{chunk['Wavelength Max']:.0f} A")
        ax.set_ylabel("Delta chi-squared")
    axes[-1].set_xlabel("Corrected RV [km/s]")
    fig.set_constrained_layout_pads(hspace=0.12)
    return fig, axes


def plot_rv_diagnostics(
    fits_file, rv_result, template_wavelength, template_flux, *, chunks="used", max_chunks=6, flexure_kwargs=None, model_kwargs=None, chi2_kwargs=None,
    emission_fit_kwargs=None,
):
    """Convenience wrapper returning flexure, model-overlay, chi-squared, and emission-line figures."""
    import matplotlib.pyplot as plt

    flexure_kwargs = {} if flexure_kwargs is None else dict(flexure_kwargs)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    chi2_kwargs = {} if chi2_kwargs is None else dict(chi2_kwargs)
    emission_fit_kwargs = {} if emission_fit_kwargs is None else dict(emission_fit_kwargs)
    fig_flex, ax_flex = plt.subplots(figsize=(7, 4))
    plot_flexure_curve(rv_result, ax=ax_flex, chunk_table=rv_result.chunk_rvs, **flexure_kwargs)
    fig_model, axes_model = plot_chunk_model_overlays(fits_file, rv_result, template_wavelength, template_flux, chunks=chunks, max_chunks=max_chunks, **model_kwargs)
    fig_chi2, axes_chi2 = plot_chunk_chi2_curves(fits_file, rv_result, template_wavelength, template_flux, chunks=chunks, max_chunks=max_chunks, **chi2_kwargs)
    figures = {"flexure": (fig_flex, ax_flex), "model_overlays": (fig_model, axes_model), "chi2_curves": (fig_chi2, axes_chi2)}
    if str(getattr(rv_result, "flexure_source", "")).lower() in {"sky_emission", "emission", "sky"}:
        fig_emission, axes_emission = plot_emission_line_fits(fits_file, rv_result, max_lines=max_chunks, **emission_fit_kwargs)
        figures["emission_line_fits"] = (fig_emission, axes_emission)
    return figures


__all__ = [
    "chunk_chi2_curve", "chunk_model_diagnostic", "find_reduced_spectrum_file", "plot_chunk_chi2_curves", "plot_chunk_model_overlays",
    "plot_emission_line_fits", "plot_flexure_curve", "plot_reduced_1d_spectrum", "plot_rv_diagnostics",
]
