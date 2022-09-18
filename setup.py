import numpy
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "mnk_game/board_algorithms.pyx",
    ], annotate=True),
    include_dirs=[numpy.get_include()]
)
