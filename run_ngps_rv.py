#!/usr/bin/env python3
"""Command-line wrapper for the public NGPS radial-velocity tutorial code.

Example
-------
python run_ngps_rv.py observed_spectra/spec2d_ngps_260515_0087_R.fits \
    template_spectra/bosz2024_mp_t6000_g+4.0_m+0.00_a+0.00_c+0.00_v2_r5000_resam.txt.gz \
    --flexure-source absorption --output-dir rv_output
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import scienceplots
    plt.style.use(["science", "no-latex"])
except Exception:
    plt.style.use("default")
plt.rcParams.update({
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR / "code_scripts"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rv_diagnostics import plot_chunk_chi2_curves, plot_chunk_model_overlays, plot_emission_line_fits, plot_flexure_curve
from rv_helpers import convert_vacuum_to_air, read_reduced_2d_spectrum
from wavelength_dependent_rvs import DEFAULT_BOSZ_WAVELENGTHS, load_example_template, measure_ngps_rv


def _parse_bounds(values):
    if values is None:
        return None
    if len(values) != 2:
        raise argparse.ArgumentTypeError("rv_bounds needs two values: LOW HIGH")
    low, high = map(float, values)
    if high <= low:
        raise argparse.ArgumentTypeError("rv_bounds must have HIGH > LOW")
    return (low, high)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure an NGPS RV and save flexure/RV diagnostics.")
    parser.add_argument("fits_file", help="Path to reduced NGPS 2D spectrum FITS file.")
    parser.add_argument("template", help="Path to template spectrum. File can have one column (flux) or two columns (wavelength in Å and flux).")
    parser.add_argument("--flux-error", default=None, 
                        help="Path to optional flux-error file. File can have one column (flux error) or two columns (wavelength in Å and flux error).")
    parser.add_argument("--template-wavelength", default=None, help="Path to optional wavelength file for input template spectrum.")
    parser.add_argument("--template-resolution", type=float, default=5000.0, help="Intrinsic resolution of the stellar template (default: R = 5000).")
    parser.add_argument("--flexure-source", choices=["auto", "absorption", "emission"], default="auto", 
                        help="Flexure source to use. Auto uses emission for G and absorption for R/I.")
    parser.add_argument("--emission-line-dir", default=None, 
                        help="Path to directory with sky emission line tables. Files should be named table*.dat.txt and include air wavelength (Å) and strength.")
    parser.add_argument("--rv-bounds", nargs=2, metavar=("LOW", "HIGH"), help="Optional RV fitting window in Å.")
    parser.add_argument("--stellar-rv-guess", type=float, default=None, help="Optional stellar RV guess in km/s.")
    parser.add_argument("--output-dir", default="ngps_rv_output", help="Path to folder for output tables and plots.")
    parser.add_argument("--max-diagnostic-chunks", type=int, default=12, help="Maximum RV chunks to plot in each diagnostic figure.")
    return parser.parse_args()


def savefig(path):
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def read_numeric_table(path):
    data = np.genfromtxt(path, delimiter=None)
    arr = np.asarray(data, dtype=float)
    if np.size(arr) == 0 or np.all(~np.isfinite(arr)):
        data = np.genfromtxt(path, delimiter=",")
    return np.asarray(data, dtype=float)


def is_bosz_template(path):
    name = Path(path).name.lower()
    return name.startswith("bosz") or "bosz2024" in name


def load_template(args):
    if args.template is None:
        raise ValueError("template is required. Choose a stellar template appropriate for the science target.")

    template_path = Path(args.template)
    if is_bosz_template(template_path):
        wavelength_path = args.template_wavelength or DEFAULT_BOSZ_WAVELENGTHS
        return load_example_template(template_path=template_path, wavelength_path=wavelength_path)

    data = read_numeric_table(template_path)
    if data.ndim == 2 and data.shape[1] >= 2 and args.template_wavelength is None:
        return data[:, 0].astype(float), data[:, 1].astype(float)

    if args.template_wavelength is None:
        raise ValueError("Non-BOSZ templates need either two columns (wavelength, flux) or --template-wavelength.")

    wavelength = read_numeric_table(args.template_wavelength)
    wavelength = wavelength[:, 0] if wavelength.ndim == 2 else wavelength
    flux = data[:, 0] if data.ndim == 2 else data
    if len(wavelength) != len(flux):
        raise ValueError("Template wavelength and flux arrays must have the same length.")
    return wavelength.astype(float), flux.astype(float)


def load_flux_error(path, fits_file):
    if path is None:
        return None
    data = np.genfromtxt(path, delimiter=None)
    if np.size(data) == 0 or np.all(~np.isfinite(np.asarray(data, dtype=float))):
        data = np.genfromtxt(path, delimiter=",")
    if data.ndim == 1:
        return data.astype(float)
    if data.ndim == 2 and data.shape[1] >= 2:
        obs_wavelength, _, _ = read_reduced_2d_spectrum(fits_file)
        return np.interp(obs_wavelength, data[:, 0].astype(float), data[:, 1].astype(float), left=np.nan, right=np.nan)
    raise ValueError("--flux-error must be a one-column error vector or a two-column wavelength/error table.")


def public_flexure_table(result):
    table = result.flexure_curve.copy()
    if "Good" in table:
        table = table[table["Good"].fillna(False).astype(bool)]
    wavelength_column = "Wavelength" if "Wavelength" in table else "Air Wavelength [A]"
    columns = [
        wavelength_column,
        "Window Min",
        "Window Max",
        "Flexure Correction",
        "Flexure Correction Error",
    ]
    output = table[[col for col in columns if col in table.columns]].copy()
    if wavelength_column == "Wavelength":
        output[wavelength_column] = convert_vacuum_to_air(output[wavelength_column].values.astype(float))
    for column in ("Window Min", "Window Max"):
        if column in output:
            output[column] = convert_vacuum_to_air(output[column].values.astype(float))
    return output.rename(
        columns={
            wavelength_column: "Observed-frame air wavelength [Å]",
            "Window Min": "Window min (observed-frame air) [Å]",
            "Window Max": "Window max (observed-frame air) [Å]",
        }
    )


def public_flexure_source_name(source):
    source = str(source)
    if source in {"telluric_stellar", "telluric", "absorption"}:
        return "absorption"
    if source in {"sky_emission", "sky", "emission"}:
        return "emission"
    return source


def public_rv_chunk_table(result):
    columns = [
        "Wavelength Min",
        "Wavelength Max",
        "Wavelength Mid",
        "RV",
        "RV Error",
        "Flexure Correction",
        "Best Fit Chi2",
        "Reduced Chi2",
        "Use in Combined RV",
    ]
    return result.chunk_rvs[[col for col in columns if col in result.chunk_rvs.columns]].rename(
        columns={
            "Wavelength Min": "Observed-frame air wavelength min [Å]",
            "Wavelength Max": "Observed-frame air wavelength max [Å]",
            "Wavelength Mid": "Observed-frame air wavelength midpoint [Å]",
        }
    )


def main():
    args = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    template_wavelength, template_flux = load_template(args)
    rv_bounds = _parse_bounds(args.rv_bounds)
    obs_fluxerr = load_flux_error(args.flux_error, args.fits_file)

    measure_kwargs = {}
    if args.stellar_rv_guess is not None:
        measure_kwargs["stellar_rv_guess"] = args.stellar_rv_guess
        measure_kwargs["rv_prior"] = args.stellar_rv_guess
    if obs_fluxerr is not None:
        measure_kwargs["obs_fluxerr"] = obs_fluxerr
    if args.emission_line_dir is not None:
        measure_kwargs["emission_line_dir"] = args.emission_line_dir
        measure_kwargs["resolution_emission_lines"] = args.emission_line_dir

    result = measure_ngps_rv(
        args.fits_file,
        template_wavelength,
        template_flux,
        template_resolution=args.template_resolution,
        flexure_source=args.flexure_source,
        rv_bounds=rv_bounds,
        return_details=True,
        **measure_kwargs,
    )

    summary = pd.DataFrame([
        {
            "fits_file": str(args.fits_file),
            "channel": result.channel,
            "flexure_source": public_flexure_source_name(result.flexure_source),
            "barycentric_correction_kms": result.barycentric_correction,
            "initial_rv_guess_kms": result.initial_rv_guess,
            "initial_rv_source": result.initial_rv_source,
            "rv_kms": result.rv,
            "rv_error_kms": result.rv_error,
        }
    ])
    summary.to_csv(outdir / "rv_summary.csv", index=False)
    if result.channel != "G":
        public_flexure_table(result).to_csv(outdir / "flexure_curve.csv", index=False)
    public_rv_chunk_table(result).to_csv(outdir / "rv_chunks.csv", index=False)

    if result.channel != "G":
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        plot_flexure_curve(result, ax=ax, simple=True)
        savefig(outdir / "flexure_curve.png")

    fig, axes = plot_chunk_model_overlays(
        args.fits_file,
        result,
        template_wavelength,
        template_flux,
        max_chunks=args.max_diagnostic_chunks,
    )
    savefig(outdir / "rv_model_overlays.png")

    fig, axes = plot_chunk_chi2_curves(
        args.fits_file,
        result,
        template_wavelength,
        template_flux,
        max_chunks=args.max_diagnostic_chunks,
        rv_half_width=25,
        rv_step=0.5,
    )
    savefig(outdir / "rv_chi2_curves.png")

    if result.flexure_source in {"sky_emission", "emission", "sky"}:
        fig, axes = plot_emission_line_fits(args.fits_file, result, max_lines=args.max_diagnostic_chunks, emission_line_dir=args.emission_line_dir or "emission_lines")
        savefig(outdir / "emission_line_fits.png")

    print(f"RV = {result.rv:.3f} +/- {result.rv_error:.3f} km/s")
    print(f"Saved outputs to {outdir.resolve()}")


if __name__ == "__main__":
    main()
