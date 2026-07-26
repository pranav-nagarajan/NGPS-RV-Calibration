"""Monte Carlo propagation of flexure-anchor uncertainties into stellar RVs.

The nominal pipeline evaluates a deterministic flexure curve at each stellar-RV
chunk.  This module draws the fitted flexure anchors, rebuilds the interpolated
curve for every draw, and measures the resulting variation of the weighted
combined RV.  The draws preserve the covariance between chunks caused by their
dependence on the same anchors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rv_helpers import convert_air_to_vacuum


@dataclass(frozen=True)
class FlexureMonteCarloResult:
    """Summary of flexure-anchor Monte Carlo propagation."""

    sigma_rv_kms: float
    median_shift_kms: float
    p16_shift_kms: float
    p84_shift_kms: float
    n_draws: int
    n_anchors: int
    n_chunks: int
    chunk_sigma_kms: np.ndarray
    combined_rv_draws_kms: np.ndarray


def _good_anchor_arrays(curve):
    required = {"Good", "Wavelength", "Flexure Correction"}
    if curve is None or len(curve) == 0 or not required.issubset(curve.columns):
        return np.array([]), np.array([]), np.array([])

    good = curve["Good"].fillna(False).values.astype(bool)
    wavelength = curve["Wavelength"].values.astype(float)
    correction = curve["Flexure Correction"].values.astype(float)
    good &= np.isfinite(wavelength) & np.isfinite(correction)
    if "Flexure Correction Error" in curve:
        error = curve["Flexure Correction Error"].values.astype(float)
    else:
        error = np.zeros(len(curve), dtype=float)
    error = np.where(np.isfinite(error) & (error > 0), error, 0.0)

    order = np.argsort(wavelength[good])
    return wavelength[good][order], correction[good][order], error[good][order]


def _anchor_covariance(errors, covariance=None, correlation=0.0):
    errors = np.asarray(errors, dtype=float)
    n_anchors = len(errors)
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (n_anchors, n_anchors):
            raise ValueError("anchor covariance must have shape (n_anchors, n_anchors)")
        if not np.all(np.isfinite(covariance)):
            raise ValueError("anchor covariance must be finite")
        return covariance

    correlation = float(correlation)
    if not -1.0 / max(n_anchors - 1, 1) < correlation < 1.0:
        raise ValueError("anchor correlation is outside the positive-definite range")
    covariance = correlation * np.outer(errors, errors)
    np.fill_diagonal(covariance, errors**2)
    return covariance


def draw_flexure_corrections(
    curve,
    wavelengths,
    *,
    n_draws=1000,
    random_state=None,
    fallback_flexure=np.nan,
    poly_degree=None,
    anchor_covariance=None,
    anchor_correlation=0.0,
):
    """Draw flexure curves and evaluate them at requested wavelengths.

    Parameters
    ----------
    curve : pandas.DataFrame
        Flexure-anchor table with ``Good``, ``Wavelength``,
        ``Flexure Correction``, and preferably ``Flexure Correction Error``.
    wavelengths : array-like
        Wavelengths in the same frame as the anchor-table ``Wavelength`` column.
    n_draws : int
        Number of Gaussian anchor draws.
    anchor_covariance : array-like, optional
        Full covariance matrix for the selected good anchors.  If omitted,
        formal anchor errors are treated as independent.
    anchor_correlation : float
        Equicorrelation used only when ``anchor_covariance`` is omitted.
    """
    wavelengths = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    n_draws = int(n_draws)
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2")

    anchor_wavelength, anchor_value, anchor_error = _good_anchor_arrays(curve)
    n_anchors = len(anchor_value)
    if n_anchors == 0:
        values = np.full((n_draws, len(wavelengths)), fallback_flexure, dtype=float)
        return values, n_anchors

    covariance = _anchor_covariance(
        anchor_error,
        covariance=anchor_covariance,
        correlation=anchor_correlation,
    )
    rng = np.random.default_rng(random_state)
    if np.all(covariance == 0):
        anchor_draws = np.repeat(anchor_value[None, :], n_draws, axis=0)
    else:
        anchor_draws = rng.multivariate_normal(
            anchor_value,
            covariance,
            size=n_draws,
            check_valid="raise",
        )

    if n_anchors == 1:
        values = np.repeat(anchor_draws, len(wavelengths), axis=1)
    elif poly_degree is not None and n_anchors > int(poly_degree):
        values = np.empty((n_draws, len(wavelengths)), dtype=float)
        for index, draw in enumerate(anchor_draws):
            values[index] = np.poly1d(np.polyfit(anchor_wavelength, draw, int(poly_degree)))(wavelengths)
    else:
        values = np.empty((n_draws, len(wavelengths)), dtype=float)
        for index, draw in enumerate(anchor_draws):
            values[index] = np.interp(
                wavelengths,
                anchor_wavelength,
                draw,
                left=draw[0],
                right=draw[-1],
            )
    return values, n_anchors


def estimate_combined_rv_flexure_uncertainty(
    chunk_rvs,
    flexure_curve,
    *,
    n_draws=1000,
    random_state=None,
    fallback_flexure=np.nan,
    poly_degree=None,
    obs_wavelengths_are_air=True,
    error_floor=3.0,
    anchor_covariance=None,
    anchor_correlation=0.0,
):
    """Estimate the combined-RV uncertainty induced by flexure anchors.

    The accepted chunk set and statistical weights are held fixed at their
    nominal values.  For each draw, the already-corrected chunk RV is shifted by
    ``nominal_flexure - drawn_flexure`` and the chunks are recombined.  This
    isolates the flexure-anchor contribution without double-counting the
    chunk-level template-fit uncertainty.
    """
    required = {"Wavelength Mid", "RV", "RV Error", "Flexure Correction"}
    if chunk_rvs is None or len(chunk_rvs) == 0 or not required.issubset(chunk_rvs.columns):
        empty = np.array([], dtype=float)
        return FlexureMonteCarloResult(np.nan, np.nan, np.nan, np.nan, int(n_draws), 0, 0, empty, empty)

    wavelength = chunk_rvs["Wavelength Mid"].values.astype(float)
    curve_wavelength = convert_air_to_vacuum(wavelength) if obs_wavelengths_are_air else wavelength
    nominal_flexure = chunk_rvs["Flexure Correction"].values.astype(float)
    rv = chunk_rvs["RV"].values.astype(float)
    rv_error = chunk_rvs["RV Error"].values.astype(float)
    if "Use in Combined RV" in chunk_rvs:
        use = chunk_rvs["Use in Combined RV"].fillna(False).values.astype(bool)
    else:
        use = np.ones(len(chunk_rvs), dtype=bool)
    use &= (
        np.isfinite(rv)
        & np.isfinite(rv_error)
        & (rv_error > 0)
        & np.isfinite(nominal_flexure)
        & np.isfinite(curve_wavelength)
    )
    if not np.any(use):
        empty = np.array([], dtype=float)
        return FlexureMonteCarloResult(np.nan, np.nan, np.nan, np.nan, int(n_draws), 0, 0, empty, empty)

    flexure_draws, n_anchors = draw_flexure_corrections(
        flexure_curve,
        curve_wavelength,
        n_draws=n_draws,
        random_state=random_state,
        fallback_flexure=fallback_flexure,
        poly_degree=poly_degree,
        anchor_covariance=anchor_covariance,
        anchor_correlation=anchor_correlation,
    )
    rv_draws = rv[None, :] + nominal_flexure[None, :] - flexure_draws
    weights = 1.0 / (rv_error[use] ** 2 + float(error_floor) ** 2)
    combined_draws = np.average(rv_draws[:, use], axis=1, weights=weights)

    nominal_combined = np.average(rv[use], weights=weights)
    shifts = combined_draws - nominal_combined
    finite = np.isfinite(shifts)
    if np.sum(finite) < 2:
        sigma = median = p16 = p84 = np.nan
    else:
        sigma = float(np.nanstd(shifts[finite], ddof=1))
        median = float(np.nanmedian(shifts[finite]))
        p16, p84 = np.nanpercentile(shifts[finite], [16, 84])

    chunk_sigma = np.nanstd(rv_draws, axis=0, ddof=1)
    return FlexureMonteCarloResult(
        sigma_rv_kms=sigma,
        median_shift_kms=median,
        p16_shift_kms=float(p16),
        p84_shift_kms=float(p84),
        n_draws=int(n_draws),
        n_anchors=int(n_anchors),
        n_chunks=int(np.sum(use)),
        chunk_sigma_kms=np.asarray(chunk_sigma, dtype=float),
        combined_rv_draws_kms=np.asarray(combined_draws, dtype=float),
    )


__all__ = [
    "FlexureMonteCarloResult",
    "draw_flexure_corrections",
    "estimate_combined_rv_flexure_uncertainty",
]
