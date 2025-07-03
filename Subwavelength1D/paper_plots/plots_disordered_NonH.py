import numpy as np
import scipy as sci

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import matplotlib.cm as cm

from Utils.settings import settings as settings

from Utils.utils_general import *
import Utils.utils_propagation as utils_propagation

import Subwavelength1D.swp as swp
import Subwavelength1D.classic as classic
import Subwavelength1D.disordered as disordered
from Subwavelength1D.metaatom import *
from Subwavelength1D.quasiperiodic import *

from typing import Literal, Callable, Tuple, Self, List, override

import copy
from tqdm import tqdm


from itertools import product
from collections import deque
plt.rcParams.update(settings.matplotlib_params)
