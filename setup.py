from setuptools._vendor.packaging import version
from setuptools import dist, setup, find_packages
from setuptools.extension import Extension
import sklearn

try:
    from Cython.Build import cythonize
except ImportError:
    dist.Distribution().fetch_build_eggs(["cython<=0.29.36"])
    from Cython.Build import cythonize
import Cython.Compiler.Options


compile_time_env = {}
if version.parse(sklearn.__version__) >= version.parse("1.2"):
    SKLEARN_NEWER_12 = True
    compile_time_env["SKLEARN_NEWER_12"] = True
else:
    compile_time_env["SKLEARN_NEWER_12"] = False


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
        # cython_compile_time_env={"SKLEARN_NEWER_12": SKLEARN_NEWER_12},
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
    ext_modules=cythonize(extensions, annotate=True, compile_time_env=compile_time_env),
    include_dirs=[np_get_include()],
)
