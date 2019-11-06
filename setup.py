from setuptools import setup, find_packages
from setuptools.extension import Extension
import causalml
try:
    from Cython.Build import cythonize
except ImportError:
    # create closure for deferred import
    def cythonize(*args, ** kwargs):
        from Cython.Build import cythonize
        return cythonize(*args, ** kwargs)
try:
    from numpy import get_include as np_get_include
except ImportError:
    # create closure for deferred import
    def np_get_include(*args, ** kwargs):
        from numpy import get_include
        return get_include(*args, ** kwargs)


with open("README.md", "r") as f:
    long_description = f.read()


with open("requirements.txt") as f:
    requirements = f.readlines()

extensions = [
    Extension("causalml.inference.tree.causaltree",
              ["causalml/inference/tree/causaltree.pyx"],
              libraries=[],
              include_dirs=[np_get_include()],
              extra_compile_args=["-O3"]
    )
]

packages = find_packages()

setup(
    name="causalml",
    version=causalml.__version__,
    author="Huigang Chen, Totte Harinen, Jeong-Yoon Lee, Mike Yung, Zhenyu Zhao",
    author_email="",
    description="Python Package for Uplift Modeling and Causal Inference with Machine Learning Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uber-common/causalml",
    packages=packages,
    python_requires=">=2.7",
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
