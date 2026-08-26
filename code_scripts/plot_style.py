"""Shared plotting typography for public tutorial and manuscript figures."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import matplotlib as mpl
from matplotlib import font_manager


def _cmu_serif_path():
    """Return a Unicode Computer Modern font path when TeX provides one."""
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich:
        try:
            result = subprocess.run(
                [kpsewhich, "cmunrm.otf"],
                check=False,
                capture_output=True,
                text=True,
            )
            candidate = Path(result.stdout.strip())
            if candidate.is_file():
                return candidate
        except OSError:
            pass

    texlive_root = Path("/usr/local/texlive")
    if texlive_root.is_dir():
        candidates = sorted(
            texlive_root.glob("*/texmf-dist/fonts/opentype/public/cm-unicode/cmunrm.otf"),
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


def configure_publication_font():
    """Use one Unicode-capable Computer Modern face for all plot text.

    Matplotlib's bundled ``cmr10`` lacks a native Angstrom glyph.  Mixing it
    with per-label LaTeX rendering makes wavelength labels look lighter than
    the remaining axis text in inline notebook SVGs.  CMU Serif is the
    Unicode extension of Computer Modern and contains a native ``Å`` glyph,
    so labels, titles, tick labels, and annotations share one renderer and
    stroke weight.  STIXGeneral is a portable serif fallback.
    """
    font_path = _cmu_serif_path()
    if font_path is not None:
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        family = "STIXGeneral"

    mpl.rcParams.update(
        {
            "font.family": family,
            "mathtext.fontset": "cm" if family == "CMU Serif" else "stix",
            "axes.formatter.use_mathtext": True,
            "text.usetex": False,
        }
    )
    return family


__all__ = ["configure_publication_font"]
