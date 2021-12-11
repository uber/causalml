============
Installation
============

Installation with ``conda`` is recommended. ``conda`` environment files for Python 3.6, 3.7, 3.8 and 3.9 are available in the repository. To use models under the ``inference.tf`` module (e.g. ``DragonNet``), additional dependency of ``tensorflow`` is required. For detailed instructions, see below.

Install using ``conda``
-----------------------

This will create a new ``conda`` virtual environment named ``causalml-[tf-]py3x``, where ``x`` is in ``[6, 7, 8, 9]``. e.g. ``causalml-py37`` or ``causalml-tf-py38``. If you want to change the name of the environment, update the relevant YAML file in ``envs/``.

.. code-block:: bash

    $ git clone https://github.com/uber/causalml.git
    $ cd causalml/envs/
    $ conda env create -f environment-py38.yml	# for the virtual environment with Python 3.8 and CausalML
    $ conda activate causalml-py38
    (causalml-py38)

To install ``causalml`` with ``tensorflow`` using ``conda``, use a relevant ``causalml-[tf-]py3x`` environment file as follows:

.. code-block:: bash

    $ git clone https://github.com/uber/causalml.git
    $ cd causalml/envs/
    $ conda env create -f environment-tf-py38.yml	# for the virtual environment with Python 3.8 and CausalML
    $ conda activate causalml-tf-py38
    (causalml-tf-py38) pip install -U numpy			# this step is necessary to fix [#338](https://github.com/uber/causalml/issues/338)

Install using ``pip``
---------------------

.. code-block:: bash

    $ git clone https://github.com/uber/causalml.git
    $ cd causalml
    $ pip install -r requirements.txt
    $ pip install causalml

To install ``causalml`` with ``tensorflow`` using ``pip``, use ``causalml[tf]`` as follows:

.. code-block:: bash

    $ git clone https://github.com/uber/causalml.git
    $ cd causalml
    $ pip install -r requirements-tf.txt
    $ pip install causalml[tf]
    $ pip install -U numpy							# this step is necessary to fix [#338](https://github.com/uber/causalml/issues/338)

Install from source
-------------------

.. code-block:: bash

    $ git clone https://github.com/uber/causalml.git
    $ cd causalml
    $ pip install -r requirements.txt
    $ python setup.py build_ext --inplace
    $ python setup.py install
