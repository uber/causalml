from setuptools import dist, setup, find_packages
from setuptools.extension import Extension
import causalml
try:
    from Cython.Build import cythonize
except ImportError:
    dist.Distribution().fetch_build_eggs(['cython'])
    from Cython.Build import cythonize
try:
    from numpy import get_include as np_get_include
except ImportError:
    dist.Distribution().fetch_build_eggs(['numpy'])
    from numpy import get_include as np_get_include


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


with open("requirements.txt") as f:
    requirements = f.readlines()

extensions = [
    Extension("causalml.inference.tree.causaltree",
              ["causalml/inference/tree/causaltree.pyx"],
              libraries=[],
              include_dirs=[np_get_include()],
              extra_compile_args=["-O3"])
]

packages = find_packages()

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
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        'setuptools>=18.0',
        'cython',
        'numpy'
    ],
    install_requires=requirements,
    ext_modules=cythonize(extensions),
    include_dirs=[np_get_include()]
)
