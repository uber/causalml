import multiprocessing as mp
import os
from setuptools import dist, setup, find_packages
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    dist.Distribution().fetch_build_eggs(["cython<=0.29.34"])
    from Cython.Build import cythonize
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True
try:
    from numpy import get_include as np_get_include
except ImportError:
    dist.Distribution().fetch_build_eggs(["numpy"])
    from numpy import get_include as np_get_include

# fmt: off
cython_modules = [
    ("causalml.inference.tree._tree._tree", "causalml/inference/tree/_tree/_tree.pyx"),
    ("causalml.inference.tree._tree._criterion", "causalml/inference/tree/_tree/_criterion.pyx"),
    ("causalml.inference.tree._tree._splitter", "causalml/inference/tree/_tree/_splitter.pyx"),
    ("causalml.inference.tree._tree._utils", "causalml/inference/tree/_tree/_utils.pyx"),
    ("causalml.inference.tree.causal._criterion", "causalml/inference/tree/causal/_criterion.pyx"),
    ("causalml.inference.tree.causal._builder", "causalml/inference/tree/causal/_builder.pyx"),
    ("causalml.inference.tree.uplift", "causalml/inference/tree/uplift.pyx"),
]
# fmt: on

extensions = [
    Extension(
        name,
        [source],
        libraries=[],
        include_dirs=[np_get_include()],
        extra_compile_args=["-O3"],
    )
    for name, source in cython_modules
]

packages = find_packages(exclude=["tests", "tests.*"])

nthreads = mp.cpu_count()
if os.name == "nt":
    nthreads = 0
else:
    mp.set_start_method("fork", force=True)

setup(
    packages=packages,
    ext_modules=cythonize(extensions, annotate=True, nthreads=nthreads),
    include_dirs=[np_get_include()],
)
