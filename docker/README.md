Docker images
=============

I made 2 docker images to run the relevant code.

icecube-tools
-------------

Has python 2.7 with ROOT bindings as well as Meerkat installed. Useful for running cascade_model and building KDEs.

hierarchical-nu
---------------

Has python 3.7 with pystan and basemap set up for sky plots. Useful for running the statistical_model code.

To use:

* `docker pull <docker-name>`

To launch a notebook in the docker, with the current directory mounted:

* ``docker run -v `pwd`:`pwd` -w `pwd` -p 3000:8080 -i -t <docker-name>``

Then simply open a browser to localhost:3000 and copy the access key over from the terminal. 
