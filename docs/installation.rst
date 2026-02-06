============
Installation
============

Installation with ``conda`` or ``pip`` is recommended.  Developers can follow the **Install from source** instructions below.  If building from source, consider doing so within a conda environment and then exporting the environment for reproducibility.

To use models under the ``inference.tf`` or ``inference.torch`` module (e.g. ``DragonNet`` or ``CEVAE``), additional dependency of ``tensorflow`` or ``torch`` is required. For detailed instructions, see below.

System Requirements
-------------------

**Python Version:** Python 3.11 or later is required.

**Linux Distributions:** Pre-built binary wheels require a Linux distribution with glibc 2.28 or later:

* **Ubuntu:** 20.04 LTS or later
* **RHEL/CentOS:** 8 or later
* **Debian:** 10 (Buster) or later
* **Fedora:** 32 or later

.. note::
   For older Linux distributions (e.g., RHEL 7, Ubuntu 16.04, Ubuntu 18.04), you will need to build CausalML from source. See **Install from source** below.

**macOS and Windows:** All recent versions are supported.

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

Install ``causalml`` with ``torch`` for ``CEVAE`` from ``PyPI``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install causalml[torch]


Install using `uv <https://github.com/astral-sh/uv/blob/main/README.md>`_
---------------------

.. code-block:: bash

    uv init
    uv add causalml

Install ``causalml`` with ``tensorflow`` for ``DragonNet`` using `uv <https://github.com/astral-sh/uv/blob/main/README.md>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    uv add "causalml[tf]"

Install ``causalml`` with ``torch`` for ``CEVAE`` using `uv <https://github.com/astral-sh/uv/blob/main/README.md>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    uv add "causalml[torch]"
    




    
Install from source
-------------------

[Optional] If you don't have Graphviz installed, you can install it using ``conda``, ``brew`` (on MacOS), or ``apt`` (on Linux).

.. code-block:: bash

    conda install python-graphviz
    brew install graphviz           # MacOS
    sudo apt-get install graphviz   # Linux

First, clone the repository and install the package:

.. code-block:: bash

    git clone https://github.com/uber/causalml.git
    cd causalml
    pip install -e .

with ``tensorflow`` for ``DragonNet``:

.. code-block:: bash

    pip install -e ".[tf]"

with ``torch`` for ``CEVAE``:

.. code-block:: bash

    pip install -e ".[torch]"

=======

Windows
-------

See content in https://github.com/uber/causalml/issues/678


Running Tests
-------------

Make sure pytest is installed before attempting to run tests.

.. code-block:: bash

    pip install -e ".[test]"

Run all tests with:

.. code-block:: bash

    pytest -vs tests/ --cov causalml/

Add ``--runtf`` and/or ``--runtorch`` to run optional tensorflow/torch tests which will be skipped by default.

You can also run tests via make:

.. code-block:: bash

    make test
