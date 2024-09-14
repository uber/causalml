============
Installation
============

Installation with ``conda`` or ``pip`` is recommended.  Developers can follow the **Install from source** instructions below.  If building from source, consider doing so within a conda environment and then exporting the environment for reproducibility.

To use models under the ``inference.tf`` or ``inference.torch`` module (e.g. ``DragonNet`` or ``CEVAE``), additional dependency of ``tensorflow`` or ``torch`` is required. For detailed instructions, see below.

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

Install from ``PyPI``
---------------------

.. code-block:: bash

    pip install causalml

Install ``causalml`` with ``tensorflow`` for ``DragonNet`` from ``PyPI``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install causalml[tf]
    pip install -U numpy                            # this step is necessary to fix [#338](https://github.com/uber/causalml/issues/338)

Install ``causalml`` with ``torch`` for ``CEVAE`` from ``PyPI``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install causalml[torch]


Install from source
-------------------

Create a clean ``conda`` environment.

.. code-block:: bash

    conda create -n causalml-py38 -y python=3.8
    conda activate causalml-py38
    conda install -c conda-forge cxx-compiler
    conda install python-graphviz
    conda install -c conda-forge xorg-libxrender
    conda install -c conda-forge libxcrypt

Then:

.. code-block:: bash

    git clone https://github.com/uber/causalml.git
    cd causalml
    pip install .
    python setup.py build_ext --inplace

with ``tensorflow`` for ``DragonNet``:

.. code-block:: bash

    pip install .[tf]

with ``torch`` for ``CEVAE``:

.. code-block:: bash

    pip install .[torch]

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

Add ``--runtf`` and/or ``--runtorch`` to run optional tensorflow/torch tests which will be skipped by default.

You can also run tests via make:

.. code-block:: bash

    make test
