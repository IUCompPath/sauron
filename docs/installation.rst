Installation
============

Create a fresh environment:

.. code-block:: bash

   conda create -n "aegis" python=3.10
   conda activate aegis

Clone the repository:

.. code-block:: bash

   git clone https://github.com/siddhesh-thakur/aegis.git && cd aegis

Install the package locally:

.. code-block:: bash

   pip install -e .

Docker Installation
-------------------

Build the Docker image:

.. code-block:: bash

    docker build -t aegis .

Run the Docker container:

.. code-block:: bash

    docker run -it --gpus all -v /path/to/your/data:/data aegis /bin/bash

This will mount your local data directory into the container at `/data`.


.. warning::
   Some pretrained models require additional dependencies. AEGIS will guide you via error messages when needed.
