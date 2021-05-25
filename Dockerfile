#https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html
#https://hub.docker.com/r/jupyter/datascience-notebook/tags

FROM jupyter/datascience-notebook
WORKDIR $HOME/work
RUN pip install --quiet optuna pymc3 keras tensorflow mxnet torch flit coverage mkdocs mkdocs-autorefs mkdocstrings nbval nltk pytest pytkdocs 
RUN pip install --quiet --user --no-warn-script-location sktime gluonts seaborn tsfel
#ENV PYTHONPATH "${PYTHONPATH}:${HOME}/work/"
ENV JUPYTER_ENABLE_LAB=yes
#CMD jupyter lab
