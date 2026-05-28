"""Shared matplotlib settings and figure size constants."""

from dataclasses import dataclass


@dataclass
class settings:
    matplotlib_params = {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": r"\usepackage{amsfonts,bm}",
        "font.serif": ["Computer Modern Serif"],
        "font.size": 8,
    }

    # path_output_figures = '../04_figures/01_code_generated/'
    figure_width = 6.4
    figure_height = 4.8
    figure_size = (figure_width, figure_height)
    figure_widesize = (figure_width, figure_height / 1.75)
    figure_halfsize = (figure_width / 2, figure_height / 2)

    @classmethod
    def figure_scaledsize(cls, w_divisor=1, h_divisor=1):
        return (cls.figure_width / w_divisor, cls.figure_height / h_divisor)

    figure_params = {"bbox_inches": "tight"}
