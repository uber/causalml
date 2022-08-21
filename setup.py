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

import causalml

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = f.readlines()

with open("requirements-test.txt") as f:
    requirements_test = f.readlines()

extensions = [
    Extension(
        "causalml.inference.tree.causal.criterion",
        ["causalml/inference/tree/causal/criterion.pyx"],
        libraries=[],
        include_dirs=[np_get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "causalml.inference.tree.causal.builder",
        ["causalml/inference/tree/causal/builder.pyx"],
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
    name="causalml",
    version=causalml.__version__,
    author="Huigang Chen, Totte Harinen, Jeong-Yoon Lee, Yuchen Luo, Jing Pan, Mike Yung, Zhenyu Zhao",
    author_email="",
    description="Python Package for Uplift Modeling and Causal Inference with Machine Learning Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uber/causalml",
    packages=packages,
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        "setuptools>=18.0",
        "cython",
        "numpy",
        "scikit-learn>=0.22.0",
    ],
    install_requires=requirements,
    tests_require=requirements_test,
    ext_modules=cythonize(extensions, annotate=True),
    include_dirs=[np_get_include()],
    extras_require={"tf": ["tensorflow>=2.4.0"]},
)
