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

    figure_params = {"bbox_inches": "tight"}
