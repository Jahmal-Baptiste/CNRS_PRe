#
# A Docker image for running mozaik simulations
#
# This image extends the "simulationx" image by adding mozaik dependencies
#
# Usage:
#
# docker build --no-cache -t mozaik .
# docker run -e DISPLAY=$DISPLAY -v `pwd`:`pwd` -w `pwd` -i -t mozaik /bin/bash
# (in the image)# python run_size_closed.py nest 8 param/defaults_mea 'data_size'

FROM neuralensemble/simulationx:py2

MAINTAINER domenico.guarino@cnrs.fr

##########################################################
# Xserver
#CMD export DISPLAY=:0
CMD export DISPLAY=0.0
#CMD export DISPLAY=:0.0
#ENV DISPLAY :0

#######################################################
# Additional prerequisite libraries

RUN $VENV/bin/pip install imagen param cycler

RUN apt-get autoremove -y && \
    apt-get clean

#######################################################
RUN git clone --branch=merged_JA_DG https://github.com/dguarino/mozaik /media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/Mozaik
WORKDIR /media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/Mozaik
RUN python setup.py install

# Simple test:
# cd examples/VogelsAbbott2005
# python run.py nest 2 param/defaults 'test'
# mpirun -np 2 python run.py nest 2 param/defaults 'test'

#######################################################
# T2
RUN git clone --branch=merged_JA_DG https://github.com/dguarino/T2 /media/Data/Scolarite/ENSTA/Stages/PRe_Windows/Environnement_de_travail/T2
