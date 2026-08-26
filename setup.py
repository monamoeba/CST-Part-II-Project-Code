from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension("scheduler", sorted(glob("native/*.cpp"))),
]

setup(
    name="colorcodecompiler-native",
    version="0.0.1",
    description="Native (C++/pybind11) scheduler extension for the color code compiler",
    ext_modules=ext_modules,
)