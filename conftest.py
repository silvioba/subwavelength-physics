import sys
import os

# Ensure the library root is on sys.path so that tests can
# import Subwavelength1D, Subwavelength3D, Utils etc. directly.
sys.path.insert(0, os.path.dirname(__file__))
