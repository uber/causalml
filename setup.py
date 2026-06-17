import multiprocessing as mp
import os
from setuptools import dist, setup, find_packages
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    dist.Distribution().fetch_build_eggs(["cython"])
    from Cython.Build import cythonize
import Cython.Compiler.Options

Cython.Compiler.Options.annotate = True

try:
    import numpy as np
except ImportError:
    dist.Distribution().fetch_build_eggs(["numpy"])
    import numpy as np

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

# Cython line tracing is opt-in via the CYTHON_TRACE env var so that coverage
# builds keep instrumentation while release wheels stay uninstrumented (and fast).
# Hardcoding `# cython: linetrace=True` in the .pyx files would otherwise leak the
# tracing hooks into published wheels, crippling nogil tree-building performance.
linetrace = os.environ.get("CYTHON_TRACE", "0") not in ("0", "", "false", "False")
define_macros = (
    [("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")] if linetrace else []
)

extensions = [
    Extension(
        name,
        [source],
        libraries=[],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
        define_macros=define_macros,
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
    ext_modules=cythonize(
        extensions,
        annotate=True,
        nthreads=nthreads,
        compiler_directives={"linetrace": linetrace},
    ),
    include_dirs=[np.get_include()],
    setup_requires=[
        "cython",
        "numpy",
    ],
)
