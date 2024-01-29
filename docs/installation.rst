============
Installation
============

Installation with ``conda`` is recommended.

``conda`` environment files for Python 3.7, 3.8 and 3.9 are available in the repository. To use models under the ``inference.tf`` module (e.g. ``DragonNet``), additional dependency of ``tensorflow`` is required. For detailed instructions, see below.

Install using ``conda``
-----------------------

Install ``conda``
^^^^^^^^^^^^^^^^^

.. code-block:: bash
    
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    source miniconda3/bin/activate
    conda init
    source ~/.bashrc

Install from ``conda-forge``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Directly install from the ``conda-forge`` channel using ``conda``.

.. code-block:: bash

    conda install -c conda-forge causalml

Install from the ``conda`` virtual environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This will create a new ``conda`` virtual environment named ``causalml-[tf-]py3x``, where ``x`` is in ``[7, 8, 9]``. e.g. ``causalml-py37`` or ``causalml-tf-py38``. If you want to change the name of the environment, update the relevant YAML file in ``envs/``.

.. code-block:: bash

    git clone https://github.com/uber/causalml.git
    cd causalml/envs/
    conda env create -f environment-py38.yml	# for the virtual environment with Python 3.8 and CausalML
    conda activate causalml-py38

Install ``causalml`` with ``tensorflow``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    git clone https://github.com/uber/causalml.git
    cd causalml/envs/
    conda env create -f environment-tf-py38.yml	# for the virtual environment with Python 3.8 and CausalML
    conda activate causalml-tf-py38
    pip install -U numpy			# this step is necessary to fix [#338](https://github.com/uber/causalml/issues/338)

Install from ``PyPI``
---------------------

.. code-block:: bash

    pip install causalml

Install ``causalml`` with ``tensorflow`` from ``PyPI``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install causalml[tf]
    pip install -U numpy                            # this step is necessary to fix [#338](https://github.com/uber/causalml/issues/338)


Install from source
-------------------

Create a clean ``conda`` environment.

.. code-block:: bash

    conda create -n causalml-py38 -y python=3.8
    conda activate causalml-py38
    conda install -c conda-forge cxx-compiler
    conda install python-graphviz
    conda install -c conda-forge xorg-libxrender

Then:

.. code-block:: bash

    git clone https://github.com/uber/causalml.git
    cd causalml
    pip install .
    python setup.py build_ext --inplace

with ``tensorflow``:

.. code-block:: bash

    pip install .[tf]

=======

Windows
-------

See content in https://github.com/uber/causalml/issues/678


Running Tests
-------------

Make sure pytest is installed before attempting to run tests.

Run all tests with:

.. code-block:: bash

    pytest -vs tests/ --cov causalml/

Add ``--runtf`` to run optional tensorflow tests which will be skipped by default.

You can also run tests via make:
 
.. code-block:: bash

    make test

