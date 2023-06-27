from setuptools import dist, setup, find_packages
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    dist.Distribution().fetch_build_eggs(["cython>=0.28.0"])
    from Cython.Build import cythonize
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
try:
    from numpy import get_include as np_get_include
except ImportError:
    dist.Distribution().fetch_build_eggs(["numpy"])
    from numpy import get_include as np_get_include

extensions = [
    Extension(
        "causalml.inference.tree.causal._criterion",
        ["causalml/inference/tree/causal/_criterion.pyx"],
        libraries=[],
        include_dirs=[np_get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "causalml.inference.tree.causal._builder",
        ["causalml/inference/tree/causal/_builder.pyx"],
        libraries=[],
        include_dirs=[np_get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "causalml.inference.tree.uplift",
        ["causalml/inference/tree/uplift.pyx"],
        libraries=[],
        include_dirs=[np_get_include()],
        extra_compile_args=["-O3"],
    ),
]

packages = find_packages(exclude=["tests", "tests.*"])

setup(
    packages=packages,
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[np_get_include()],
)
