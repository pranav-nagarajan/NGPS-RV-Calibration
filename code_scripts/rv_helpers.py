"""Shared IO, metadata, template, and numerical helpers for NGPS RV work."""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.io import fits
from astropy.time import Time
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares
from scipy.signal import medfilt


C_KMS = 299792.458
# The public tutorial bundle keeps code in code_scripts/ and support files
# in sibling folders rooted at the repository directory. Users can still pass
# absolute paths or call set_data_dir(...) for another project layout.
DEFAULT_TELLURIC_GRID = "template_spectra/telluric_grid_airmass_default.npz"
DEFAULT_TELLURIC_GRID_DIR = "template_spectra"
DEFAULT_TELLURIC_MODEL_RESOLUTION = 100000
DEFAULT_EMISSION_LINE_DIR = "emission_lines"
DEFAULT_BOSZ_WAVELENGTHS = "template_spectra/bosz2024_wave_r5000.txt"
DEFAULT_BOSZ_GRID = "template_spectra"
DEFAULT_TEMPLATE_RESOLUTION = 5000
NGPS_CHANNEL_RANGES = {
    "U": (3100.0, 4360.0),
    "G": (4170.0, 5900.0),
    "R": (5610.0, 7940.0),
    "I": (7560.0, 10400.0),
}
EMISSION_LINE_CHANNEL_RANGES = {
    "U": NGPS_CHANNEL_RANGES["U"],
    "G": NGPS_CHANNEL_RANGES["G"],
    "R": (NGPS_CHANNEL_RANGES["R"][0], 7820.0),
    "I": (7820.0, NGPS_CHANNEL_RANGES["I"][1]),
}
DEFAULT_EMISSION_LINE_MIN_STRENGTH = {
    "G": 2.0,
    "R": 5.0,
    "I": 30.0,
}
DEFAULT_EMISSION_LINE_MIN_SEPARATION = {
    "G": 10.0,
    "R": 10.0,
    "I": 10.0,
    "U": 10.0,
}
DEFAULT_RESOLUTION_LINES = {
    "U": np.array([3142.6768, 3145.5142, 3212.1833, 3297.1917, 3372.0657, 3485.1995, 3747.0691, 3750.0967]),
    "G": np.array([5197.9282, 5200.2856, 5577.3467, 5889.9590, 5895.9321]),
    "R": np.array([5889.9590, 5895.9321, 6300.3086, 6363.7827, 6863.9707, 6923.2200, 7276.4050]),
    "I": np.array([8399.1758, 8827.1123, 8885.8564, 8919.6100, 8943.3950, 8958.0630, 9003.2200, 9375.9766]),
}

_telluric_cache = {}
_telluric_resolution_cache = {}
_telluric_degraded_cache = {}
_telluric_grid_index_cache = {}
_emission_line_cache = {}
_bosz_wavelength_cache = {}
_bosz_index_cache = {}


def _default_data_dir() -> Path:
    if "__file__" not in globals():
        return Path.cwd()
    script_dir = Path(__file__).resolve().parent
    return script_dir.parent if script_dir.name == "code_scripts" else script_dir


DATA_DIR = _default_data_dir()


def set_data_dir(data_dir):
    """Point the module at the folder containing templates and telluric grids."""
    global DATA_DIR, _telluric_cache, _telluric_resolution_cache, _telluric_degraded_cache, _telluric_grid_index_cache, _emission_line_cache, _bosz_wavelength_cache
    global _bosz_index_cache
    DATA_DIR = Path(data_dir).expanduser().resolve()
    _telluric_cache = {}
    _telluric_resolution_cache = {}
    _telluric_degraded_cache = {}
    _telluric_grid_index_cache = {}
    _emission_line_cache = {}
    _bosz_wavelength_cache = {}
    _bosz_index_cache = {}


def _resolve_path(path, base_dir=None) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path(base_dir) if base_dir is not None else DATA_DIR) / path


def _float_array(arr):
    return np.asarray(getattr(arr, "value", arr), dtype=float)


def _odd_window(window_AA):
    window_AA = int(np.round(window_AA))
    if window_AA % 2 == 0:
        window_AA += 1
    return max(window_AA, 3)


def _header_float(header, keys, default=np.nan):
    for key in keys:
        if key in header:
            try:
                return float(header[key])
            except Exception:
                continue
    return default


def _palomar_location():
    try:
        return EarthLocation.of_site("Palomar")
    except Exception:
        return EarthLocation.from_geodetic(lon=-116.863 * u.deg, lat=33.356 * u.deg, height=1706 * u.m)


def _read_coordinate_degrees(header, ra=None, dec=None):
    if ra is not None and dec is not None:
        return float(ra), float(dec)

    ra_value = ra
    dec_value = dec
    for key in ("RA", "OBJRA", "TARGRA", "CAT-RA", "RADEG"):
        if ra_value is None and key in header:
            ra_value = header[key]
    for key in ("DEC", "DECL", "OBJDEC", "TARGDEC", "CAT-DEC", "DECDEG"):
        if dec_value is None and key in header:
            dec_value = header[key]

    if ra_value is None or dec_value is None:
        raise ValueError("RA and Dec must be supplied or present in the FITS header.")

    try:
        return float(ra_value), float(dec_value)
    except Exception:
        coord = SkyCoord(str(ra_value), str(dec_value), unit=(u.hourangle, u.deg), frame="icrs")
        return coord.ra.deg, coord.dec.deg


def _read_observation_time(header, mjd=None, jd=None):
    if jd is not None:
        return Time(float(jd), format="jd", scale="utc", location=_palomar_location())
    if mjd is not None:
        return Time(float(mjd), format="mjd", scale="utc", location=_palomar_location())

    for key in ("MJD", "MJD-OBS", "MJDOBS"):
        if key in header:
            return Time(float(header[key]), format="mjd", scale="utc", location=_palomar_location())
    for key in ("JD", "JD-OBS", "JDOBS"):
        if key in header:
            return Time(float(header[key]), format="jd", scale="utc", location=_palomar_location())
    if "DATE-OBS" in header:
        return Time(header["DATE-OBS"], format="fits", scale="utc", location=_palomar_location())
    raise ValueError("Observation time must be supplied or present as MJD/JD/DATE-OBS in the FITS header.")


def barycentric_correction(ra, dec, time):
    """Return the barycentric velocity correction in km/s."""
    coordinate = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    return coordinate.radial_velocity_correction(obstime=time).to(u.km / u.s).value


def _read_airmass(header, airmass=None):
    if airmass is not None:
        return float(airmass)
    value = _header_float(header, ("AIRMASS", "SECZ"), default=np.nan)
    if not np.isfinite(value):
        raise ValueError("Airmass must be supplied or present in the FITS header.")
    return value


def _channel_from_wavelength(wavelength, default="R"):
    wavelength = _float_array(wavelength)
    finite = wavelength[np.isfinite(wavelength)]
    if len(finite) == 0:
        return default

    wl_min, wl_max = np.nanmin(finite), np.nanmax(finite)
    overlaps = {}
    for channel, (low, high) in NGPS_CHANNEL_RANGES.items():
        overlaps[channel] = max(0.0, min(wl_max, high) - max(wl_min, low))
    best_channel = max(overlaps, key=overlaps.get)
    if overlaps[best_channel] > 0:
        return best_channel

    median = np.nanmedian(finite)
    centers = {channel: 0.5 * (low + high) for channel, (low, high) in NGPS_CHANNEL_RANGES.items()}
    return min(centers, key=lambda channel: abs(median - centers[channel]))


def _read_channel(header, filename, wavelength=None, default="R"):
    value = str(header.get("SPEC_ID", header.get("CHANNEL", ""))).strip().upper()
    if value in NGPS_CHANNEL_RANGES:
        return value
    name = Path(filename).name.upper()
    for channel in NGPS_CHANNEL_RANGES:
        if re.search(rf"(^|[_\-.]){channel}([_\-.]|$)", name):
            return channel
    if wavelength is not None and np.isfinite(np.nanmedian(wavelength)):
        return _channel_from_wavelength(wavelength, default=default)
    return default


def _trace_position(science, trace_column, search_width=30):
    y_expected = science.shape[0] // 2
    y_min = max(0, y_expected - search_width)
    y_max = min(science.shape[0], y_expected + search_width)
    return y_min + int(np.nanargmax(science[y_min:y_max, trace_column]))


def _wavelength_1d(wavelength_data, science_shape, y0, y1, trace_y):
    wavelength_data = _float_array(wavelength_data)
    if wavelength_data.ndim == 1:
        return wavelength_data

    if wavelength_data.shape == science_shape:
        return np.nanmedian(wavelength_data[y0:y1, :], axis=0)

    if wavelength_data.shape[-1] == science_shape[-1]:
        row = int(np.clip(trace_y, 0, wavelength_data.shape[0] - 1))
        return wavelength_data[row, :]

    if wavelength_data.shape[0] == science_shape[-1]:
        col = int(np.clip(trace_y, 0, wavelength_data.shape[1] - 1))
        return wavelength_data[:, col]

    raise ValueError("Could not turn the FITS wavelength extension into a 1D wavelength grid.")


def read_reduced_2d_spectrum(fits_file, trace_column=None, aperture_half_width=2, trace_y=None, sky_hdu=2, wavelength_hdu=3):
    """Read a reduced NGPS 2D spectrum and return a simple extracted 1D spectrum."""
    with fits.open(_resolve_path(fits_file, base_dir=Path.cwd())) as hdul:
        header = hdul[0].header.copy()
        science = _float_array(hdul[1].data)
        sky = np.zeros_like(science)
        if len(hdul) > sky_hdu and hdul[sky_hdu].data is not None:
            candidate = _float_array(hdul[sky_hdu].data)
            if candidate.shape == science.shape:
                sky = candidate
        if len(hdul) <= wavelength_hdu or hdul[wavelength_hdu].data is None:
            raise ValueError("Reduced FITS file must contain a wavelength solution extension.")
        wavelength_data = _float_array(hdul[wavelength_hdu].data)

    sky_subtracted = science - sky
    if trace_column is None:
        trace_column = min(200, sky_subtracted.shape[1] - 1)
    trace_column = int(np.clip(trace_column, 0, sky_subtracted.shape[1] - 1))
    if trace_y is None:
        trace_y = _trace_position(sky_subtracted, trace_column)
    trace_y = int(np.clip(trace_y, 0, sky_subtracted.shape[0] - 1))

    half_width = max(1, int(aperture_half_width))
    y0 = max(0, trace_y - half_width)
    y1 = min(sky_subtracted.shape[0], trace_y + half_width + 1)
    wavelength = _wavelength_1d(wavelength_data, sky_subtracted.shape, y0, y1, trace_y)
    flux = np.nansum(sky_subtracted[y0:y1, :], axis=0)
    return wavelength, flux, header


def read_sky_model_at_trace(
    fits_file, trace_column=None, trace_y=None, sky_hdu=2, wavelength_hdu=3, aperture_half_width=2,
):
    """Extract the sky model through the same boxcar aperture as the science spectrum."""
    with fits.open(_resolve_path(fits_file, base_dir=Path.cwd())) as hdul:
        header = hdul[0].header.copy()
        science = _float_array(hdul[1].data)
        sky = _float_array(hdul[sky_hdu].data)
        wavelength_data = _float_array(hdul[wavelength_hdu].data)

    sky_subtracted = science - sky
    if trace_column is None:
        trace_column = min(200, sky_subtracted.shape[1] - 1)
    trace_column = int(np.clip(trace_column, 0, sky_subtracted.shape[1] - 1))
    if trace_y is None:
        trace_y = _trace_position(sky_subtracted, trace_column)
    trace_y = int(np.clip(trace_y, 0, sky_subtracted.shape[0] - 1))

    half_width = max(1, int(aperture_half_width))
    y0 = max(0, trace_y - half_width)
    y1 = min(sky.shape[0], trace_y + half_width + 1)
    wavelength = _wavelength_1d(wavelength_data, sky_subtracted.shape, y0, y1, trace_y)
    sky_flux = np.nansum(sky[y0:y1, :], axis=0)
    return wavelength, sky_flux, header, trace_y


def _candidate_emission_line_dirs(emission_line_dir=DEFAULT_EMISSION_LINE_DIR):
    path = _resolve_path(emission_line_dir)
    candidates = [path]
    if path.name == "emission_lines":
        candidates.append(path.with_name("Emission Lines"))
    elif path.name == "Emission Lines":
        candidates.append(path.with_name("emission_lines"))
    return candidates


def _add_emission_line_neighbor_separation(catalog):
    catalog = catalog.sort_values("Air Wavelength [A]").copy()
    wavelengths = catalog["Air Wavelength [A]"].values.astype(float)
    previous_sep = np.r_[np.inf, np.diff(wavelengths)]
    next_sep = np.r_[np.diff(wavelengths), np.inf]
    catalog["Nearest Neighbor Separation [A]"] = np.minimum(previous_sep, next_sep)
    return catalog


def _resolve_default_emission_line_setting(value, defaults, channel_key, name):
    if isinstance(value, str):
        if value.lower() != "default":
            raise ValueError(f"{name} must be numeric, None, or 'default'.")
        return defaults.get(channel_key)
    return value


def load_emission_line_catalog(
    emission_line_dir=DEFAULT_EMISSION_LINE_DIR, channel=None, wavelength_range=None, min_strength="default", min_separation="default"
):
    """
    Load Hanuschik-style sky-emission line tables as air wavelengths.

    Bundled Hanuschik tables use four numeric columns: line number, air
    wavelength, line width, and relative strength. User tables may omit the
    line number, but must provide wavelength and relative strength. A line
    width column may be included between them.
    """
    cache_key = str(emission_line_dir)
    if cache_key not in _emission_line_cache:
        rows = []
        for line_dir in _candidate_emission_line_dirs(emission_line_dir):
            if not line_dir.is_dir():
                continue
            for path in sorted(line_dir.glob("table*.dat.txt")):
                data = np.genfromtxt(path)
                if data.size == 0:
                    continue
                first_data_line = ""
                with open(path) as handle:
                    for line in handle:
                        stripped = line.strip()
                        if stripped and not stripped.startswith("#"):
                            first_data_line = stripped
                            break
                n_columns = len(first_data_line.split())
                if np.ndim(data) == 0:
                    data = np.asarray([[float(data)]])
                elif np.ndim(data) == 1:
                    if n_columns <= 1:
                        data = np.asarray(data, dtype=float).reshape(-1, 1)
                    else:
                        data = np.asarray(data, dtype=float).reshape(1, -1)
                if data.shape[1] < 1:
                    continue
                has_line_number = data.shape[1] >= 4 and np.nanmedian(data[:, 1]) > 1000 and np.nanmedian(data[:, 0]) < 1000
                if not has_line_number and data.shape[1] < 2:
                    raise ValueError(
                        f"{path} must contain at least air wavelength and relative strength columns. "
                        "Use either: wavelength_A strength, or wavelength_A width strength."
                    )
                for row in data:
                    if has_line_number:
                        line_number = int(row[0]) if np.isfinite(row[0]) else np.nan
                        wavelength = float(row[1])
                        width = float(row[2]) if data.shape[1] > 2 else np.nan
                        strength = float(row[3]) if data.shape[1] > 3 else np.nan
                    else:
                        line_number = np.nan
                        wavelength = float(row[0])
                        width = float(row[1]) if data.shape[1] > 2 else np.nan
                        strength = float(row[2]) if data.shape[1] > 2 else float(row[1])
                    rows.append(
                        {
                            "Table": path.name,
                            "Line Number": line_number,
                            "Air Wavelength [A]": wavelength,
                            "Width": width,
                            "Strength": strength,
                        }
                    )
            if rows:
                break
        _emission_line_cache[cache_key] = pd.DataFrame(rows, columns=["Table", "Line Number", "Air Wavelength [A]", "Width", "Strength"])

    catalog = _emission_line_cache[cache_key].copy()
    if len(catalog) == 0:
        return catalog

    channel_key = str(channel).upper() if channel is not None else None
    if channel_key is not None:
        if channel_key in EMISSION_LINE_CHANNEL_RANGES:
            low, high = EMISSION_LINE_CHANNEL_RANGES[channel_key]
            catalog = catalog[(catalog["Air Wavelength [A]"] >= low) & (catalog["Air Wavelength [A]"] <= high)]

    if wavelength_range is not None:
        low, high = wavelength_range
        catalog = catalog[(catalog["Air Wavelength [A]"] >= low) & (catalog["Air Wavelength [A]"] <= high)]

    min_strength = _resolve_default_emission_line_setting(min_strength, DEFAULT_EMISSION_LINE_MIN_STRENGTH, channel_key, "min_strength")
    if min_strength is not None and "Strength" in catalog:
        catalog = catalog[catalog["Strength"] >= float(min_strength)]

    catalog = _add_emission_line_neighbor_separation(catalog)
    min_separation = _resolve_default_emission_line_setting(
        min_separation, DEFAULT_EMISSION_LINE_MIN_SEPARATION, channel_key, "min_separation"
    )
    if min_separation is not None and "Nearest Neighbor Separation [A]" in catalog:
        catalog = catalog[catalog["Nearest Neighbor Separation [A]"] >= float(min_separation)]

    return catalog.reset_index(drop=True)


def _wavelength_column_from_2d_catalog(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2:
        return values
    if values.shape[1] == 0:
        return np.array([], dtype=float)
    if values.shape[1] >= 2 and np.nanmedian(values[:, 1]) > 1000:
        return values[:, 1]
    for idx in range(values.shape[1]):
        column = values[:, idx]
        if np.nanmedian(column) > 1000:
            return column
    return values[:, 0]


def _line_wavelengths_from_catalog(line_catalog, channel=None):
    """Normalize a line catalog into air wavelengths."""
    if line_catalog is None:
        channel_key = str(channel).upper() if channel is not None else "R"
        return DEFAULT_RESOLUTION_LINES.get(channel_key, DEFAULT_RESOLUTION_LINES["R"]).copy()

    if isinstance(line_catalog, str) and line_catalog.lower() in {"auto", "default", "hanuschik"}:
        line_catalog = load_emission_line_catalog(channel=channel)
    elif isinstance(line_catalog, (str, Path)):
        path = _resolve_path(line_catalog)
        if not path.exists():
            path = _resolve_path(line_catalog, base_dir=Path.cwd())
        if path.is_dir():
            line_catalog = load_emission_line_catalog(path, channel=channel)
        else:
            line_catalog = np.genfromtxt(path)

    if isinstance(line_catalog, dict):
        keys = [channel, str(channel).upper(), str(channel).lower()] if channel is not None else []
        for key in keys:
            if key in line_catalog:
                line_catalog = line_catalog[key]
                break
        else:
            values = []
            for value in line_catalog.values():
                values.extend(_line_wavelengths_from_catalog(value))
            return np.asarray(values, dtype=float)

    if isinstance(line_catalog, pd.DataFrame):
        for col in ("Air Wavelength [A]", "Air Wavelength", "Wavelength", "wavelength", "lambda"):
            if col in line_catalog.columns:
                return line_catalog[col].values.astype(float)
        raise ValueError("Could not find a wavelength column in the line catalog.")

    return _wavelength_column_from_2d_catalog(line_catalog).astype(float)


def _gaussian_plus_line(params, wavelength):
    amp, mean, sigma, c0, c1 = params
    x = wavelength - mean
    return amp * np.exp(-0.5 * (x / sigma) ** 2) + c0 + c1 * x


def fit_sky_line_resolution(wavelength, sky_flux, rest_wavelength, window_size=5, min_pixels=5, min_snr=5):
    """Fit one unresolved sky line and return its spectral resolution."""
    wavelength = _float_array(wavelength)
    sky_flux = _float_array(sky_flux)
    m = (wavelength > rest_wavelength - window_size) & (wavelength < rest_wavelength + window_size) & np.isfinite(wavelength) & np.isfinite(sky_flux)
    if np.sum(m) < min_pixels:
        raise ValueError("too few pixels around sky line")

    x = wavelength[m]
    y = sky_flux[m]
    c0 = np.nanmedian(y)
    amp = np.nanmax(y) - c0
    noise = 1.4826 * np.nanmedian(np.abs(y - np.nanmedian(y)))
    if not np.isfinite(noise) or noise <= 0:
        noise = max(np.nanstd(y), 1.0)
    if not np.isfinite(amp) or amp <= 0 or amp < min_snr * noise:
        raise ValueError("sky line is too weak for a resolution fit")

    p0 = np.array([amp, rest_wavelength, 1.0, c0, 0.0], dtype=float)
    lo = np.array([0.0, rest_wavelength - window_size, 0.05, -np.inf, -np.inf], dtype=float)
    hi = np.array([np.inf, rest_wavelength + window_size, window_size, np.inf, np.inf], dtype=float)

    def residuals(params):
        return (_gaussian_plus_line(params, x) - y) / noise

    result = least_squares(residuals, p0, bounds=(lo, hi), max_nfev=1000)
    if not result.success:
        raise ValueError(result.message)

    params = result.x
    if params[2] <= 0 or abs(params[1] - rest_wavelength) > window_size:
        raise ValueError("unphysical sky-line fit")

    fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(params[2])
    resolution = abs(params[1]) / fwhm
    errs = least_squares_errors(result)
    resolution_error = np.nan
    if len(errs) > 2 and np.isfinite(errs[2]) and params[2] != 0:
        resolution_error = resolution * abs(errs[2] / params[2])
    return params, resolution, resolution_error


def measure_resolution(
    fits_file, emission_lines=None, channel=None, trace_column=None, trace_y=None, window_size=5, min_lines=1, min_snr=5, resolution_range=(1500, 8000),
    sigma_clip=4, return_table=False,
):
    """
    Estimate spectral resolution from sharp sky-emission lines at the trace.

    Returns the median fitted resolving power. Set ``return_table=True`` to also
    receive the per-line fit table.
    """
    sky_wl, sky_flux, header, fitted_trace_y = read_sky_model_at_trace(fits_file, trace_column=trace_column, trace_y=trace_y)
    if channel is None:
        channel = _read_channel(header, fits_file, sky_wl)

    line_wavelengths = _line_wavelengths_from_catalog(emission_lines, channel=channel)
    wl_min, wl_max = np.nanmin(sky_wl), np.nanmax(sky_wl)
    line_wavelengths = line_wavelengths[(line_wavelengths > wl_min) & (line_wavelengths < wl_max)]

    rows = []
    for rest_wavelength in line_wavelengths:
        try:
            params, resolution, resolution_error = fit_sky_line_resolution(sky_wl, sky_flux, rest_wavelength, window_size=window_size, min_snr=min_snr)
        except Exception:
            continue

        fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(params[2])
        rows.append(
            {
                "Air Wavelength [A]": rest_wavelength, "Fit Mean": params[1], "Fit Sigma": params[2], "FWHM [A]": fwhm, "Resolution": resolution,
                "Resolution Error": resolution_error, "Amplitude": params[0], "Trace Pixel": fitted_trace_y, "Success": True,
            }
        )

    table = pd.DataFrame(rows)
    if len(table) == 0:
        table = pd.DataFrame(
            columns=["Air Wavelength [A]", "Fit Mean", "Fit Sigma", "FWHM [A]", "Resolution", "Resolution Error", "Amplitude", "Trace Pixel", "Success", "Good"]
        )
        return (np.nan, table) if return_table else np.nan

    good = np.isfinite(table["Resolution"].values)
    if resolution_range is not None:
        low, high = resolution_range
        good &= (table["Resolution"].values >= low) & (table["Resolution"].values <= high)

    if np.sum(good) >= 3 and sigma_clip is not None:
        vals = table.loc[good, "Resolution"].values.astype(float)
        med = np.nanmedian(vals)
        scatter = 1.4826 * np.nanmedian(np.abs(vals - med))
        scatter = max(scatter, 100)
        good_indices = np.flatnonzero(good)
        good[good_indices] = np.abs(vals - med) < sigma_clip * scatter

    table["Good"] = good
    resolution = np.nanmedian(table.loc[table["Good"], "Resolution"]) if np.any(table["Good"]) else np.nan
    if np.sum(table["Good"].values) < min_lines:
        resolution = np.nan
    return (float(resolution), table) if return_table else float(resolution)


def _resolve_telluric_grid_path(grid_path=DEFAULT_TELLURIC_GRID):
    path = _resolve_path(grid_path)
    if path.exists() or path.is_absolute():
        return path

    grid_name = Path(grid_path).name
    for grid_dir in _candidate_telluric_grid_dirs(DEFAULT_TELLURIC_GRID_DIR):
        candidate = grid_dir / grid_name
        if candidate.exists():
            return candidate
    return path


def _resolution_from_wavelength_sampling(wavelength):
    wavelength = _float_array(wavelength)
    finite = np.isfinite(wavelength)
    if np.sum(finite) < 2:
        return np.nan

    wave = wavelength[finite]
    dw = np.diff(wave)
    dw = dw[np.isfinite(dw) & (dw > 0)]
    if len(dw) == 0:
        return np.nan

    spacing = np.nanmedian(dw)
    median_wavelength = np.nanmedian(wave)
    if not (np.isfinite(spacing) and spacing > 0 and np.isfinite(median_wavelength)):
        return np.nan
    return float(median_wavelength / spacing)


def _telluric_input_resolution(grid_path=DEFAULT_TELLURIC_GRID, fallback=DEFAULT_TELLURIC_MODEL_RESOLUTION):
    path = _resolve_telluric_grid_path(grid_path)
    cache_key = str(path)
    if cache_key in _telluric_resolution_cache:
        return _telluric_resolution_cache[cache_key]

    resolution = float(fallback)
    if path.exists():
        try:
            with np.load(path) as tmp:
                for key in ("resolution", "resolving_power", "input_resolution", "model_resolution", "R"):
                    if key in tmp.files:
                        value = np.asarray(tmp[key]).astype(float)
                        if value.size and np.isfinite(np.nanmedian(value)):
                            resolution = float(np.nanmedian(value))
                            break
                else:
                    wl_grid = _float_array(tmp["wl_grid"])
                    telluric_grids = _float_array(tmp["telluric_grids"])
                    if telluric_grids.ndim == 2 and telluric_grids.shape[-1] == len(wl_grid):
                        sampled_resolution = _resolution_from_wavelength_sampling(wl_grid)
                        if np.isfinite(sampled_resolution):
                            resolution = sampled_resolution
        except Exception:
            resolution = float(fallback)

    _telluric_resolution_cache[cache_key] = resolution
    return resolution


def load_telluric_grid(grid_path=DEFAULT_TELLURIC_GRID):
    """Return the vacuum telluric wavelength grid, airmass grid, and spectra."""
    path = _resolve_telluric_grid_path(grid_path)
    cache_key = str(path)
    if cache_key not in _telluric_cache:
        with np.load(path) as tmp:
            wl_grid = _float_array(tmp["wl_grid"])
            airmass_vals = _float_array(tmp["airmass_vals"])
            telluric_grids = _float_array(tmp["telluric_grids"])

        if telluric_grids.ndim == 2 and telluric_grids.shape[-1] != len(wl_grid):
            if telluric_grids.shape[0] == len(wl_grid):
                telluric_grids = telluric_grids.T
            else:
                raise ValueError(
                    f"Telluric wavelength grid length ({len(wl_grid)}) does not match telluric spectra ({telluric_grids.shape[-1]}) in {path}. "
                    "The telluric grid must have one wavelength sample per model-spectrum pixel."
                )

        _telluric_cache[cache_key] = (wl_grid, airmass_vals, telluric_grids)
    return _telluric_cache[cache_key]


def _candidate_telluric_grid_dirs(telluric_grid_dir=DEFAULT_TELLURIC_GRID_DIR):
    path = _resolve_path(telluric_grid_dir)
    candidates = [path]
    if path.name == "Telluric_Models":
        candidates.append(path.with_name("Telluric Models"))
    elif path.name == "Telluric Models":
        candidates.append(path.with_name("Telluric_Models"))
    return candidates


def _parse_resolution_from_name(path):
    text = path.stem.lower()
    patterns = (r"(?:^|[_\-\s])r(?:es)?[_=\-\s]*(\d{3,5})(?:$|[_\-\s])", r"resolution[_=\-\s]*(\d{3,5})")
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is not None:
            return float(match.group(1))
    return np.nan


def available_telluric_grids(telluric_grid_dir=DEFAULT_TELLURIC_GRID_DIR):
    """Return available telluric grids and their parsed resolving powers."""
    cache_key = str(telluric_grid_dir)
    if cache_key in _telluric_grid_index_cache:
        return _telluric_grid_index_cache[cache_key].copy()

    rows = []
    for grid_dir in _candidate_telluric_grid_dirs(telluric_grid_dir):
        if not grid_dir.is_dir():
            continue
        for path in sorted(grid_dir.glob("*.npz")):
            resolution = _parse_resolution_from_name(path)
            if np.isfinite(resolution):
                rows.append({"Resolution": resolution, "Path": path})
        if rows:
            break

    table = pd.DataFrame(rows, columns=["Resolution", "Path"])
    _telluric_grid_index_cache[cache_key] = table
    return table.copy()


def select_telluric_grid_path(resolution, telluric_grid_dir=DEFAULT_TELLURIC_GRID_DIR, fallback_grid_path=DEFAULT_TELLURIC_GRID):
    """Choose the available telluric grid nearest to the requested resolution."""
    default_grid = _resolve_telluric_grid_path(fallback_grid_path)
    if default_grid.exists():
        return str(default_grid)

    grids = available_telluric_grids(telluric_grid_dir)
    if len(grids) == 0:
        return str(default_grid)

    target_resolution = float(resolution) if np.isfinite(resolution) else 4000.0
    idx = int(np.argmin(np.abs(grids["Resolution"].values.astype(float) - target_resolution)))
    return str(grids.iloc[idx]["Path"])


def load_bosz_wavelengths(wave_path=DEFAULT_BOSZ_WAVELENGTHS):
    path = _resolve_path(wave_path)
    cache_key = str(path)
    if cache_key not in _bosz_wavelength_cache:
        _bosz_wavelength_cache[cache_key] = np.genfromtxt(path)
    return _bosz_wavelength_cache[cache_key]


def _parse_bosz_filename(path):
    match = re.search(r"_t(?P<teff>\d+)_g(?P<logg>[+-]\d+\.\d)_m(?P<mh>[+-]\d+\.\d+)_", path.name)
    if match is None:
        return None
    return {"teff": float(match.group("teff")), "logg": float(match.group("logg")), "mh": float(match.group("mh")), "path": path}


def _bosz_grid_index(grid_dir=DEFAULT_BOSZ_GRID):
    grid_dir = _resolve_path(grid_dir)
    cache_key = str(grid_dir)
    if cache_key in _bosz_index_cache:
        return _bosz_index_cache[cache_key].copy()

    rows = []
    for path in sorted(grid_dir.glob("*.txt.gz")):
        parsed = _parse_bosz_filename(path)
        if parsed is not None:
            rows.append(parsed)
    if not rows:
        raise FileNotFoundError(f"No BOSZ templates found in {grid_dir}")
    _bosz_index_cache[cache_key] = pd.DataFrame(rows)
    return _bosz_index_cache[cache_key].copy()


def retrieve_bosz_spectrum(mh, teff, logg, grid_dir=DEFAULT_BOSZ_GRID):
    """Return the nearest BOSZ template flux array."""
    grid = _bosz_grid_index(grid_dir)
    d2 = ((grid["mh"].values - mh) / 0.25) ** 2 + ((grid["teff"].values - teff) / 250) ** 2 + ((grid["logg"].values - logg) / 0.5) ** 2
    path = grid.iloc[int(np.argmin(d2))]["path"]
    spec = np.genfromtxt(path)
    return spec[:, 0] / spec[:, 1]


def template_from_inputs(
    header, template_wavelength=None, template_flux=None, teff=None, logg=None, mh=None, bosz_wave_path=DEFAULT_BOSZ_WAVELENGTHS,
    bosz_grid_dir=DEFAULT_BOSZ_GRID,
):
    """Return a template from explicit arrays or nearest-neighbor BOSZ parameters."""
    if template_wavelength is not None and template_flux is not None:
        return _float_array(template_wavelength), _float_array(template_flux)

    if teff is None:
        teff = _header_float(header, ("TEFF", "T_EFF"), default=np.nan)
    if logg is None:
        logg = _header_float(header, ("LOGG", "LOG_G"), default=np.nan)
    if mh is None:
        mh = _header_float(header, ("MH", "M_H", "FEH", "FE_H", "[FE/H]"), default=np.nan)

    if not (np.isfinite(teff) and np.isfinite(logg) and np.isfinite(mh)):
        raise ValueError("Provide template_wavelength/template_flux, or provide teff/logg/mh " "for nearest-neighbor BOSZ template selection.")

    return load_bosz_wavelengths(bosz_wave_path), retrieve_bosz_spectrum(mh, teff, logg, grid_dir=bosz_grid_dir)


def degrade_spectrum_resolution(wavelength, flux, output_resolution, input_resolution=DEFAULT_TEMPLATE_RESOLUTION):
    """Convolve a spectrum from ``input_resolution`` down to ``output_resolution``."""
    wavelength = _float_array(wavelength)
    flux = _float_array(flux)
    if not (np.isfinite(output_resolution) and np.isfinite(input_resolution)):
        return flux.copy()
    output_resolution = float(output_resolution)
    input_resolution = float(input_resolution)
    if output_resolution <= 0 or input_resolution <= 0 or output_resolution >= input_resolution:
        return flux.copy()

    good = np.isfinite(wavelength) & np.isfinite(flux) & (wavelength > 0)
    if np.sum(good) < 5:
        return flux.copy()

    order = np.argsort(wavelength[good])
    wave_good = wavelength[good][order]
    flux_good = flux[good][order]
    log_wave = np.log(wave_good)
    dlog = np.nanmedian(np.diff(log_wave))
    if not np.isfinite(dlog) or dlog <= 0:
        return flux.copy()

    log_grid = np.arange(log_wave[0], log_wave[-1] + 0.5 * dlog, dlog)
    flux_grid = np.interp(log_grid, log_wave, flux_good)

    sigma_in = 1 / (input_resolution * 2 * np.sqrt(2 * np.log(2)))
    sigma_out = 1 / (output_resolution * 2 * np.sqrt(2 * np.log(2)))
    sigma_kernel = np.sqrt(max(0.0, sigma_out**2 - sigma_in**2))
    sigma_pix = sigma_kernel / dlog
    if not np.isfinite(sigma_pix) or sigma_pix <= 0:
        return flux.copy()

    degraded_grid = gaussian_filter1d(flux_grid, sigma_pix, mode="nearest", truncate=4.0)
    degraded = flux.copy()
    degraded[good] = np.interp(np.log(wavelength[good]), log_grid, degraded_grid)
    return degraded


def convert_air_to_vacuum(wave_air):
    """Air-to-vacuum wavelength conversion, valid above 2000 Angstrom."""
    wave_air = np.asarray(wave_air, dtype=float)
    wave_vac = np.copy(wave_air)
    for _ in range(2):
        sigma2 = (1e4 / wave_vac) ** 2
        factor = 1.0 + 0.05792105 / (238.0185 - sigma2) + 0.00167917 / (57.362 - sigma2)
        wave_vac = wave_air * factor
    return wave_vac


def medfilt_fixed_window_AA(wl, flux, window_AA=301):
    """Continuum estimate using a median filter on a 1 Angstrom grid."""
    wl = _float_array(wl)
    flux = _float_array(flux)
    good = np.isfinite(wl) & np.isfinite(flux)
    if np.sum(good) < 5:
        return np.full_like(flux, np.nanmedian(flux), dtype=float)

    wl_grid = np.arange(np.nanmin(wl[good]), np.nanmax(wl[good]), 1)
    flux_grid = np.interp(wl_grid, wl[good], flux[good])
    kernel = min(_odd_window(window_AA), _odd_window(max(3, len(wl_grid) - 1)))
    continuum_grid = medfilt(flux_grid, kernel)
    continuum = np.interp(wl, wl_grid, continuum_grid)
    bad = ~np.isfinite(continuum) | (continuum == 0)
    continuum[bad] = np.nanmedian(continuum[~bad]) if np.any(~bad) else np.nanmedian(flux[good])
    return continuum


def estimate_normalized_flux_error(
    wavelength, flux, mask=None, continuum_lims=None, continuum_window_AA=21, continuum_percentile=60, sigma_clip=4, min_pixels=20, error_floor=1e-4,
    return_snr=False,
):
    """
    Estimate normalized flux error from the variance of continuum-like pixels.

    The spectrum is continuum-normalized, then continuum-like pixels are chosen
    either from ``continuum_lims`` or from the upper part of the normalized flux
    distribution. The returned error is ``sqrt(var(normalized_flux))``.
    """
    wavelength = _float_array(wavelength)
    flux = _float_array(flux)
    continuum = medfilt_fixed_window_AA(wavelength, flux, window_AA=continuum_window_AA)
    norm_flux = np.divide(flux, continuum, out=np.full_like(flux, np.nan, dtype=float), where=continuum != 0)

    good = np.isfinite(wavelength) & np.isfinite(norm_flux) & (norm_flux > 0.05) & (norm_flux < 2.5)
    if mask is not None:
        good &= np.asarray(mask, dtype=bool)
    if continuum_lims is not None:
        low, high = continuum_lims
        good &= (wavelength > low) & (wavelength < high)

    if np.sum(good) < min_pixels:
        good = np.isfinite(wavelength) & np.isfinite(norm_flux) & (norm_flux > 0.05) & (norm_flux < 2.5)
        if continuum_lims is not None:
            low, high = continuum_lims
            good &= (wavelength > low) & (wavelength < high)

    if np.sum(good) == 0:
        flux_error = 1.0
        snr = 1.0
        return (flux_error, snr) if return_snr else flux_error

    if continuum_lims is None and np.sum(good) >= min_pixels:
        vals = norm_flux[good]
        threshold = np.nanpercentile(vals, continuum_percentile)
        high = np.nanpercentile(vals, 98)
        continuum_mask = good & (norm_flux >= threshold) & (norm_flux <= high)
    else:
        continuum_mask = good

    if np.sum(continuum_mask) < min_pixels:
        continuum_mask = good

    vals = norm_flux[continuum_mask].astype(float)
    vals = vals[np.isfinite(vals)]
    for _ in range(4):
        if len(vals) < min_pixels or sigma_clip is None:
            break
        center = np.nanmedian(vals)
        scatter = np.nanstd(vals, ddof=1) if len(vals) > 1 else np.nan
        if not np.isfinite(scatter) or scatter <= 0:
            break
        keep = np.abs(vals - center) < sigma_clip * scatter
        if np.all(keep):
            break
        vals = vals[keep]

    if len(vals) > 1:
        flux_error = float(np.sqrt(np.nanvar(vals, ddof=1)))
    else:
        flux_error = np.nan
    if not np.isfinite(flux_error) or flux_error <= 0:
        flux_error = 1.0
    flux_error = max(float(flux_error), float(error_floor))
    snr = 1 / flux_error if flux_error > 0 else np.nan
    return (flux_error, snr) if return_snr else flux_error


def estimate_model_continuum_flux_error(
    normalized_flux, model_flux, mask=None, model_continuum_min=0.98, model_continuum_max=1.02, adaptive_fraction=0.30,
    sigma_clip=4, clip_iterations=4, min_pixels=20, error_floor=1e-4, return_details=False,
):
    """Estimate flux error from pixels identified as continuum by the fitted model.

    The preferred pixels have stellar-model flux within two percent of unity.
    For line-rich chunks with too few such pixels, the least-absorbed fraction
    of valid pixels is selected by the model alone. The observed flux therefore
    does not determine which side of the noise distribution is retained.
    """
    normalized_flux = _float_array(normalized_flux)
    model_flux = _float_array(model_flux)
    if len(normalized_flux) != len(model_flux):
        raise ValueError("normalized_flux and model_flux must have the same length")

    valid = np.isfinite(normalized_flux) & np.isfinite(model_flux)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    continuum_mask = valid & (model_flux >= float(model_continuum_min)) & (model_flux <= float(model_continuum_max))
    adaptive = False
    if np.sum(continuum_mask) < min_pixels:
        eligible = np.flatnonzero(valid)
        if len(eligible) >= min_pixels:
            n_select = min(len(eligible), max(int(min_pixels), int(np.ceil(float(adaptive_fraction) * len(eligible)))))
            ranked = eligible[np.argsort(model_flux[eligible])]
            continuum_mask = np.zeros_like(valid, dtype=bool)
            continuum_mask[ranked[-n_select:]] = True
            adaptive = True

    residuals = normalized_flux[continuum_mask] - model_flux[continuum_mask]
    residuals = residuals[np.isfinite(residuals)]
    for _ in range(int(clip_iterations)):
        if len(residuals) < min_pixels or sigma_clip is None:
            break
        center = np.nanmedian(residuals)
        scatter = np.nanstd(residuals, ddof=1) if len(residuals) > 1 else np.nan
        if not np.isfinite(scatter) or scatter <= 0:
            break
        keep = np.abs(residuals - center) < float(sigma_clip) * scatter
        if np.all(keep):
            break
        residuals = residuals[keep]

    flux_error = np.nanstd(residuals, ddof=1) if len(residuals) > 1 else np.nan
    if np.isfinite(flux_error) and flux_error > 0:
        flux_error = max(float(flux_error), float(error_floor))
    else:
        flux_error = np.nan

    if return_details:
        return flux_error, int(len(residuals)), bool(adaptive), continuum_mask
    return flux_error


def doppler_shift(wavelength, flux, dv):
    """Evaluate ``flux`` at wavelengths Doppler shifted by ``dv`` km/s."""
    wavelength = _float_array(wavelength)
    flux = _float_array(flux)
    factor = np.sqrt((1 - dv / C_KMS) / (1 + dv / C_KMS))
    shifted_wavelength = wavelength * factor
    return np.interp(shifted_wavelength, wavelength, flux, left=1, right=1)


def telluric_model_for_airmass(
    airmass, grid_path=DEFAULT_TELLURIC_GRID, output_resolution=None, input_resolution=None, degrade_to_resolution=True,
):
    wl_ref, airmass_vals, telluric_grids = load_telluric_grid(grid_path)
    idx = int(np.round((airmass - airmass_vals[0]) / (airmass_vals[1] - airmass_vals[0])))
    idx = int(np.clip(idx, 0, len(airmass_vals) - 1))
    tell_flux = telluric_grids[idx]

    if degrade_to_resolution and output_resolution is not None and np.isfinite(output_resolution):
        if input_resolution is None:
            input_resolution = _telluric_input_resolution(grid_path)
        if np.isfinite(input_resolution) and float(output_resolution) < float(input_resolution):
            path = _resolve_telluric_grid_path(grid_path)
            cache_key = (str(path), idx, round(float(output_resolution), 6), round(float(input_resolution), 6))
            if cache_key not in _telluric_degraded_cache:
                _telluric_degraded_cache[cache_key] = degrade_spectrum_resolution(
                    wl_ref, tell_flux, output_resolution=output_resolution, input_resolution=input_resolution
                )
            tell_flux = _telluric_degraded_cache[cache_key]

    return wl_ref, tell_flux, airmass_vals[idx]


def least_squares_errors(result):
    """Return least-squares parameter errors from the Jacobian."""
    try:
        jac = result.jac
        _, s, vt = np.linalg.svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * s[0]
        s = s[s > threshold]
        vt = vt[: len(s)]
        cov = (vt.T / s**2) @ vt
        dof = max(1, result.fun.size - result.x.size)
        cov *= 2 * result.cost / dof
        return np.sqrt(np.diag(cov))
    except Exception:
        return np.full_like(result.x, np.nan, dtype=float)
