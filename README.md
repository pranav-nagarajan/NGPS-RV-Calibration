# Measuring Radial Velocities with the Next Generation Palomar Spectrograph

This repository serves as a tutorial for computing flexure corrections and radial velocities (RVs) with the Next Generation Palomar Spectrograph (NGPS) at Palomar Observatory. The method is calibrated using RV standard stars and will be described in detail in Nagarajan et al. (forthcoming).

## How to Use

The tutorial notebook uses a 0.5 arcsec slit NGPS observation of a Gaia-ESO RV standard star in the G, R, and I bands as an example of how to measure flexure-corrected RVs with our pipeline. A user can either follow that tutorial or run `run_ngps_rv.py`, which provides a command-line interface that will print the measured RV and save output products in a specified folder:

```bash
python run_ngps_rv.py path/to/observed_spectrum \
  path/to/template_spectrum \
  --output-dir example_rv_output
```

The repository includes a HITRAN2020 telluric absorption model (`template_spectra/telluric_grid_airmass_default.npz`) and sky emission line tables from Hanuschik (2003) (`emission_lines/`), both of which are used by default.

### Inputs

`fits_file` is the first required positional argument. It should be a reduced NGPS 2D FITS spectrum from the QuickLook Data Reduction Pipeline. Our pipeline extracts the 1D spectrum from this file, estimates the flux uncertainty if one is not supplied, derives the flexure correction, and measures the RV.

`template`, which is the second required positional argument, supplies the stellar template spectrum. If the filename is recognized as a BOSZ template, the pipeline uses the bundled BOSZ wavelength grid (unless `--template-wavelength` is also supplied). Otherwise, the template file should either have two numeric columns, wavelength in Angstroms and flux, or one numeric flux column paired with a separate `--template-wavelength` file that supplies the 1D wavelength array.

`--flux-error` optionally supplies the 1D flux uncertainty. The file may either be one column with the same length as the extracted spectrum, or two columns containing wavelength in Angstroms and flux error. In the latter case, the errors are interpolated onto the observed wavelength grid.

`--flexure-source` chooses how the flexure correction is measured. The default, `auto`, uses sky emission lines in G and telluric absorption features in R/I. Use `absorption` to force telluric absorption, or `emission` to force sky emission lines.

`--emission-line-dir` optionally supplies a custom sky emission line catalog directory. The directory should contain one or more whitespace-delimited files named `table*.dat.txt`. Each row should provide either air wavelength in Angstroms and relative strength, or air wavelength in Angstroms, line width in Angstroms, and relative strength.

`--rv-bounds LOW HIGH` restricts the wavelength range used for the stellar RV chunks. The values should be in Angstroms. The flexure curve is still derived from the available telluric features, so a custom RV window may require interpolation or clamping of the flexure correction.

`--stellar-rv-guess` supplies an optional initial stellar RV guess in km/s. If it is omitted for an R or I band spectrum, the pipeline will estimate an initial guess for the stellar RV based on sky emission lines.

`--output-dir` names the folder where tables and diagnostic plots are saved. `--max-diagnostic-chunks` controls how many RV chunks or emission lines are shown in the diagnostic figures.

### Outputs

The terminal prints the final RV and uncertainty in km/s.

`rv_summary.csv` summarizes the following information: the input file, channel, flexure source, barycentric correction, any initial RV guess, the final RV, and the RV error.

`flexure_curve.csv` gives the derived flexure correction curve (i.e., measured flexure as a function of air wavelength in the observer frame). `flexure_curve.png` visualizes this curve (in the R or I channels).

`rv_chunks.csv` reports the stellar RV measurements in each wavelength chunk. It includes the observed-frame air-wavelength limits and midpoint in Angstroms, flexure correction applied at the midpoint, RV and RV error, corresponding chi-squared value, and whether each chunk was used in the robust combined RV. `rv_model_overlays.png` compares the normalized observed spectrum to the shifted stellar template on an observed-frame air-wavelength axis. `rv_chi2_curves.png` shows the corresponding chi-squared curves used to measure the best-fit chunk RVs.

`emission_line_fits.png` is saved if sky emission lines are used to derive the flexure correction. It shows the Gaussian fits to the selected emission lines.
