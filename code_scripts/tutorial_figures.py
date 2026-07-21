"""Plotting helpers for the extracted 1D science and sky spectra in the NGPS RV tutorial notebook."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from rv_helpers import (
    C_KMS,
    convert_air_to_vacuum,
    doppler_shift,
    load_emission_line_catalog,
    read_reduced_2d_spectrum,
    read_sky_model_at_trace,
)

DATA_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TARGET_NAME = "GESJ17452610+0515305"
DEFAULT_SLIT_WIDTH_ARCSEC = 0.5
DEFAULT_NIGHT = "2026-05-15"

def _odd_window(n):
    n = int(round(n))
    return n + 1 if n % 2 == 0 else max(n, 3)


def _median_continuum(flux, window=41):
    return medfilt(np.asarray(flux, dtype=float), kernel_size=_odd_window(window))


def _shared_continuum_model(x, observed_flux, model, mask, degree=1):
    good = mask & np.isfinite(x) & np.isfinite(observed_flux) & np.isfinite(model) & (model != 0)
    if np.sum(good) < max(5, degree + 2):
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


_telluric_display_cache = {}


def _telluric_display_model(wavelength, flexure_kms=0.0, airmass=1.154):
    cache_key = (round(float(flexure_kms), 2), round(float(airmass), 3))
    if cache_key not in _telluric_display_cache:
        with np.load(DATA_DIR / "template_spectra" / "telluric_grid_airmass_default.npz") as npz:
            tell_wl = npz["wl_grid"]
            airmass_vals = npz["airmass_vals"]
            tell_grid = npz["telluric_grids"]
            idx = int(np.nanargmin(np.abs(airmass_vals - airmass)))
            tell_flux = tell_grid[idx]
        dw = float(np.nanmedian(np.diff(tell_wl)))
        fwhm = float(np.nanmedian(tell_wl)) / 5000.0
        sigma_pix = fwhm / (2 * np.sqrt(2 * np.log(2)) * dw)
        if np.isfinite(sigma_pix) and sigma_pix > 0:
            tell_flux = gaussian_filter1d(tell_flux, sigma_pix)
        shifted_tell = doppler_shift(tell_wl, tell_flux, flexure_kms)
        _telluric_display_cache[cache_key] = (tell_wl, shifted_tell)
    tell_wl, shifted_tell = _telluric_display_cache[cache_key]
    compare_wl = convert_air_to_vacuum(wavelength)
    return np.interp(compare_wl, tell_wl, shifted_tell, left=1, right=1)


def _local_telluric_window_normalization(
    wavelength,
    flux,
    low,
    high,
    *,
    flexure_kms=0.0,
    airmass=1.154,
    apply_low=None,
    apply_high=None,
    margin=45,
    min_transmission=0.97,
    cap_clear_envelope=False,
    continuum_degree=1,
):
    tell_model = _telluric_display_model(wavelength, flexure_kms=flexure_kms, airmass=airmass)
    fit_region = (wavelength > low - margin) & (wavelength < high + margin) & np.isfinite(flux)
    good = fit_region & (tell_model > min_transmission)
    if np.count_nonzero(good) < 8:
        outside = fit_region & ((wavelength < low - 5) | (wavelength > high + 5))
        good = outside & np.isfinite(flux)
    continuum = _shared_continuum_model(wavelength, flux, tell_model, good, degree=continuum_degree)
    normalized = np.divide(flux, continuum, out=np.full_like(flux, np.nan, dtype=float), where=continuum != 0)
    if cap_clear_envelope:
        clear = fit_region & (tell_model > min_transmission) & np.isfinite(normalized) & np.isfinite(tell_model) & (tell_model > 0)
        edge_clear = clear & (((wavelength > low) & (wavelength < low + 110)) | ((wavelength > high - 110) & (wavelength < high + 120)))
        if np.count_nonzero(edge_clear) >= 10:
            ratio = normalized / tell_model
            left_edge = edge_clear & (wavelength < low + 110)
            right_edge = edge_clear & (wavelength > high - 110)
            anchors_x, anchors_y = [], []
            if np.count_nonzero(left_edge) >= 5:
                anchors_x.append(float(np.nanmedian(wavelength[left_edge])))
                anchors_y.append(float(np.nanmedian(ratio[left_edge])))
            if np.count_nonzero(right_edge) >= 5:
                anchors_x.append(float(np.nanmedian(wavelength[right_edge])))
                anchors_y.append(float(np.nanmedian(ratio[right_edge])))
            if len(anchors_x) == 2 and np.all(np.isfinite(anchors_y)):
                edge_trend = np.interp(wavelength, anchors_x, anchors_y)
                normalized = np.divide(normalized, edge_trend, out=np.full_like(normalized, np.nan), where=np.isfinite(edge_trend) & (edge_trend != 0))
        clear = fit_region & (tell_model > min_transmission) & np.isfinite(normalized)
        if np.count_nonzero(clear) >= 8:
            scale = np.nanpercentile(normalized[clear], 90)
            if np.isfinite(scale) and scale > 1:
                normalized = normalized / scale
        left_anchor = (wavelength > low - 55) & (wavelength < low + 35) & (tell_model > min_transmission) & np.isfinite(normalized)
        right_anchor = (wavelength > high - 35) & (wavelength < high + 55) & (tell_model > min_transmission) & np.isfinite(normalized)
        anchors_x, anchors_y = [], []
        if np.count_nonzero(left_anchor) >= 5:
            anchors_x.append(float(np.nanmedian(wavelength[left_anchor])))
            anchors_y.append(float(np.nanmedian(normalized[left_anchor])))
        if np.count_nonzero(right_anchor) >= 5:
            anchors_x.append(float(np.nanmedian(wavelength[right_anchor])))
            anchors_y.append(float(np.nanmedian(normalized[right_anchor])))
        if len(anchors_x) == 2 and np.all(np.isfinite(anchors_y)):
            edge_trend = np.interp(wavelength, anchors_x, anchors_y)
            normalized = np.divide(normalized, edge_trend, out=np.full_like(normalized, np.nan), where=np.isfinite(edge_trend) & (edge_trend != 0))
            clear = fit_region & (tell_model > min_transmission) & np.isfinite(normalized)
            if np.count_nonzero(clear) >= 8:
                scale = np.nanpercentile(normalized[clear], 90)
                if np.isfinite(scale) and scale > 1:
                    normalized = normalized / scale
    if apply_low is None:
        apply_low = low
    if apply_high is None:
        apply_high = high
    apply = (wavelength >= apply_low) & (wavelength <= apply_high) & np.isfinite(normalized)
    return normalized, apply


def _normalize_science_for_display(wavelength, flux, channel):
    continuum = _median_continuum(flux, 41)
    norm = np.divide(flux, continuum, out=np.full_like(flux, np.nan, dtype=float), where=continuum != 0)
    local_windows = {
        "R": [
            (6860, 6965, 40.0, None, None, False),
            (7160, 7390, 38.0, None, None, False),
            (7590, 7705, 40.0, None, None, False),
        ],
        "I": [
            (7578, 7705, -78.0, 7545, 7705, True),
            (8120, 8400, -68.0, None, None, False),
            (8900, 9800, -58.0, 8870, 9900, True, 3),
        ],
    }
    for row in local_windows.get(channel, []):
        if len(row) == 6:
            low, high, flexure, apply_low, apply_high, cap_clear = row
            degree = 1
        else:
            low, high, flexure, apply_low, apply_high, cap_clear, degree = row
        local_norm, apply = _local_telluric_window_normalization(
            wavelength,
            flux,
            low,
            high,
            flexure_kms=flexure,
            apply_low=apply_low,
            apply_high=apply_high,
            cap_clear_envelope=cap_clear,
            continuum_degree=degree,
        )
        norm[apply] = local_norm[apply]
    if channel == "I":
        blue_edge = (wavelength >= 7545) & (wavelength <= 7578) & np.isfinite(norm)
        continuum_edge = blue_edge & (norm > 0.65) & (norm < 1.25)
        if np.count_nonzero(continuum_edge) >= 5:
            coeffs = np.polyfit(wavelength[continuum_edge], norm[continuum_edge], 1)
            trend = np.polyval(coeffs, wavelength[blue_edge])
            norm[blue_edge] = np.divide(norm[blue_edge], trend, out=norm[blue_edge], where=np.isfinite(trend) & (trend != 0))
        tell_model = _telluric_display_model(wavelength, flexure_kms=-58.0, airmass=1.154)
        red_edge = (wavelength >= 9820) & (wavelength <= 10040) & np.isfinite(norm)
        red_clear = red_edge & (tell_model > 0.985) & (norm > 0.75) & (norm < 1.25)
        if np.count_nonzero(red_clear) >= 8:
            x = wavelength[red_clear]
            y = norm[red_clear] / tell_model[red_clear]
            xx = 2 * (x - np.nanmean(x)) / max(np.nanmax(x) - np.nanmin(x), 1e-6)
            coeffs = np.polyfit(xx, y, 2 if np.count_nonzero(red_clear) >= 12 else 1)
            target = (wavelength >= 9780) & (wavelength <= 10050) & np.isfinite(norm)
            target_xx = 2 * (wavelength[target] - np.nanmean(x)) / max(np.nanmax(x) - np.nanmin(x), 1e-6)
            trend = np.polyval(coeffs, target_xx)
            blend = np.clip((wavelength[target] - 9780) / 45.0, 0.0, 1.0)
            divisor = (1 - blend) + blend * trend
            norm[target] = np.divide(norm[target], divisor, out=norm[target], where=np.isfinite(divisor) & (divisor != 0))
            local_bump = (wavelength >= 9860) & (wavelength <= 9900) & (tell_model > 0.985) & np.isfinite(norm)
            if np.count_nonzero(local_bump) >= 8:
                bump_ratio = np.nanmedian(norm[local_bump] / tell_model[local_bump])
                if np.isfinite(bump_ratio) and bump_ratio > 1.003:
                    bump_target = (wavelength >= 9820) & (wavelength <= 9940) & np.isfinite(norm)
                    bump_profile = np.exp(-0.5 * ((wavelength[bump_target] - 9880.0) / 24.0) ** 2)
                    bump_divisor = 1.0 + (bump_ratio - 1.0) * bump_profile
                    norm[bump_target] = np.divide(norm[bump_target], bump_divisor, out=norm[bump_target], where=np.isfinite(bump_divisor) & (bump_divisor != 0))
    return norm


def _replace_with_neighbor_interpolation(wavelength, flux, center, half_width=2.6, side_width=6.0):
    replace = np.abs(wavelength - center) <= half_width
    if not np.any(replace):
        return flux
    side = ((wavelength >= center - half_width - side_width) & (wavelength < center - half_width)) | (
        (wavelength > center + half_width) & (wavelength <= center + half_width + side_width)
    )
    side &= np.isfinite(wavelength) & np.isfinite(flux)
    if np.count_nonzero(side) >= 2:
        order = np.argsort(wavelength[side])
        flux[replace] = np.interp(wavelength[replace], wavelength[side][order], flux[side][order])
    else:
        local = _median_continuum(flux, 9)
        flux[replace] = local[replace]
    return flux


SCIENCE_CHANNEL_WINDOWS = {
    "U": (3100, 4300),
    "G": (4170, 5900),
    "R": (5800, 7870),
    "I": (7545, 10400),
}
SCIENCE_Y_LIMITS = {
    "U": (0.05, 1.75),
    "G": (0.45, 1.35),
    "R": (0.15, 1.25),
    "I": (0.00, 1.35),
}
TELLURIC_SHADE_BANDS = {
    "U": [],
    "G": [],
    "R": [(6860, 6965), (7160, 7390), (7590, 7705)],
    "I": [(7578, 7705), (8120, 8400), (8900, 9800)],
}
DISPLAY_SPIKE_MASKS = {
    "G": [(4746.52, 2.6)],
    "R": [(6576.13, 2.0), (7216.10, 0.75)],
    "I": [(8670.45, 2.6), (8674.04, 2.6)],
}
STELLAR_LINE_ANNOTATIONS = {
    "U": [
        (r"Ca II K", (3933.66,)),
        (r"Ca II H", (3968.47,)),
    ],
    "G": [
        (r"H$\beta$", (4861.33,)),
        (r"Mg I b triplet", (5167.32, 5172.68, 5183.60)),
    ],
    "R": [(r"H$\alpha$", (6562.80,))],
    "I": [(r"Ca II triplet", (8498.02, 8542.09, 8662.14))],
}
SKY_CHANNEL_WINDOWS = {
    "U": (3100, 4300),
    "G": (4170, 5900),
    "R": (5800, 7870),
    "I": (7545, 10400),
}
SKY_Y_LIMITS = {
    "U": (-25, 500),
    "G": (-75, 1250),
    "R": (-125, 1200),
    "I": (-500, 7750),
}


def plot_science_spectra_ugri(
    observed_spectra, *, channels=("U", "G", "R", "I"), figsize=(10, 10.6),
    target_name=DEFAULT_TARGET_NAME, slit_width_arcsec=DEFAULT_SLIT_WIDTH_ARCSEC, night=DEFAULT_NIGHT,
    annotate_stellar_lines=False, stellar_line_velocity_kms=None,
):
    fig, axes = plt.subplots(len(channels), 1, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f'{target_name}, {slit_width_arcsec:.2f}" slit, {night}', fontsize=15)
    for ax, channel in zip(axes, channels):
        wavelength, flux, _ = read_reduced_2d_spectrum(observed_spectra[channel])
        norm = _normalize_science_for_display(wavelength, flux, channel)
        low, high = SCIENCE_CHANNEL_WINDOWS[channel]
        visible = np.isfinite(wavelength) & np.isfinite(norm) & (wavelength >= low) & (wavelength <= high)
        in_telluric = np.zeros_like(wavelength, dtype=bool)
        for band_low, band_high in TELLURIC_SHADE_BANDS[channel]:
            in_telluric |= (wavelength >= band_low) & (wavelength <= band_high)
        local = _median_continuum(norm, 9)
        resid = norm - local
        plot_norm = norm.copy()
        usable = visible & ~in_telluric
        scatter = 1.4826 * np.nanmedian(np.abs(resid[usable] - np.nanmedian(resid[usable]))) if np.any(usable) else np.nan
        if np.isfinite(scatter) and scatter > 0:
            spike = visible & (resid > max(0.20, 6 * scatter))
            plot_norm[spike] = local[spike]
        for spike_wavelength, half_width in DISPLAY_SPIKE_MASKS.get(channel, []):
            plot_norm = _replace_with_neighbor_interpolation(wavelength, plot_norm, spike_wavelength, half_width=half_width)
        for band_low, band_high in TELLURIC_SHADE_BANDS[channel]:
            ax.axvspan(band_low, band_high, color="0.86", zorder=0)
        ax.plot(wavelength[visible], plot_norm[visible], color="black", linewidth=0.7, zorder=2)
        ax.set_xlim(low, high)
        ax.set_ylim(*SCIENCE_Y_LIMITS[channel])
        if annotate_stellar_lines:
            ymin, ymax = SCIENCE_Y_LIMITS[channel]
            label_y = ymax - 0.045 * (ymax - ymin)
            line_top = ymax - 0.13 * (ymax - ymin)
            line_bottom = max(1.04, ymax - 0.24 * (ymax - ymin))
            for label, line_wavelengths in STELLAR_LINE_ANNOTATIONS.get(channel, []):
                line_wavelengths = np.asarray(line_wavelengths, dtype=float)
                if stellar_line_velocity_kms is not None:
                    velocity = stellar_line_velocity_kms.get(channel, 0.0) if isinstance(stellar_line_velocity_kms, dict) else stellar_line_velocity_kms
                    line_wavelengths = line_wavelengths * (1 + float(velocity) / C_KMS)
                in_view = line_wavelengths[(line_wavelengths >= low) & (line_wavelengths <= high)]
                if not len(in_view):
                    continue
                ax.vlines(
                    in_view, line_bottom, line_top, color="tab:blue", linewidth=0.9,
                    alpha=0.9, zorder=3,
                )
                text_x = float(np.mean(in_view))
                text_ha = "center"
                if channel == "U" and label == "Ca II K":
                    text_x -= 4
                    text_ha = "right"
                elif channel == "U" and label == "Ca II H":
                    text_x += 4
                    text_ha = "left"
                ax.text(
                    text_x, label_y, label, color="tab:blue",
                    ha=text_ha, va="top", fontsize=11, zorder=4,
                )
        ax.set_title(f"{channel} channel extracted 1D science spectrum")
        ax.set_xlabel(r"Observed-frame air wavelength [$\mathrm{\AA}$]")
        ax.set_ylabel("Normalized flux")
    return fig, axes


def plot_sky_emission_spectra_ugri(
    observed_spectra, *, channels=("U", "G", "R", "I"), figsize=(10, 10.6),
    target_name=DEFAULT_TARGET_NAME, slit_width_arcsec=DEFAULT_SLIT_WIDTH_ARCSEC, night=DEFAULT_NIGHT,
):
    fig, axes = plt.subplots(len(channels), 1, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(f'{target_name}, {slit_width_arcsec:.2f}" slit, {night}', fontsize=15)
    for ax, channel in zip(axes, channels):
        wavelength, sky_flux, _, _ = read_sky_model_at_trace(observed_spectra[channel])
        sky_continuum = _median_continuum(sky_flux, 41)
        sky_minus_continuum = sky_flux - sky_continuum
        low, high = SKY_CHANNEL_WINDOWS[channel]
        visible = np.isfinite(wavelength) & np.isfinite(sky_minus_continuum) & (wavelength >= low) & (wavelength <= high)
        ax.plot(wavelength[visible], sky_minus_continuum[visible], color="black", linewidth=0.7, zorder=2)
        if channel != "U":
            line_table = load_emission_line_catalog(channel=channel)
            if len(line_table):
                line_wavelengths = line_table["Air Wavelength [A]"].values.astype(float)
                in_view = line_wavelengths[(line_wavelengths >= low) & (line_wavelengths <= high)]
                ymin, ymax = SKY_Y_LIMITS[channel]
                marker_top = ymax - 0.04 * (ymax - ymin)
                marker_bottom = marker_top - 0.10 * (ymax - ymin)
                ax.vlines(in_view, marker_bottom, marker_top, color="tab:red", linewidth=0.55, alpha=0.9, zorder=3)
        ax.set_xlim(low, high)
        ax.set_ylim(*SKY_Y_LIMITS[channel])
        ax.set_title(f"{channel} channel extracted 1D sky spectrum")
        ax.set_xlabel(r"Observed-frame air wavelength [$\mathrm{\AA}$]")
        ax.set_ylabel("Sky - continuum\n(arbitrary units)")
    return fig, axes


__all__ = ["plot_science_spectra_ugri", "plot_sky_emission_spectra_ugri"]
