FROM rocker/verse:4.2

# Create environment variables
ENV PATH=/opt/conda/condabin:/opt/conda/bin:${PATH}

# Path of RENV cache inside container, must be mounted
ENV RENV_PATHS_CACHE=/renv/cache  
ENV RENV_PATHS_ROOT=/renv/root
ENV RETICULATE_MINICONDA_PATH=/opt/conda/
ENV RETICULATE_PYTHON=/opt/conda/envs/biotools/bin/python

# Create arguments.  Values passed from docker-compose.yml
ARG RENV_VERSION=0.16.0
ARG CONDA_VERSION=py39_4.12.0

# Add environment variables to Renviron
RUN touch ${R_HOME}/etc/Renviron && \
    echo "RENV_PATHS_CACHE=${RENV_PATHS_CACHE}" >> ${R_HOME}/etc/Renviron && \
    echo "RENV_PATHS_ROOT=${RENV_PATHS_ROOT}" >> ${R_HOME}/etc/Renviron && \
    echo "RETICULATE_MINICONDA_PATH=${RETICULATE_MINICONDA_PATH}" >> ${R_HOME}/etc/Renviron && \
    echo "RETICULATE_PYTHON=${RETICULATE_PYTHON}" >> ${R_HOME}/etc/Renviron 

RUN apt update && apt -y install x11-apps

# Install miniconda
WORKDIR /tmp

# Install miniconda and create base environment
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh -O miniconda.sh && \
    mkdir -p /opt && \
    sh miniconda.sh -b -p /opt/conda && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/rstudio/.shinit && \
    echo "conda activate base" >> /home/rstudio/.shinit && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    conda update conda && \
    conda clean -afy  && \
    chgrp -R rstudio /opt/conda && \
    chmod 770 -R /opt/conda && \
    adduser rstudio rstudio

# create the environments from templates
COPY docker/environments/ environments/ 
RUN conda env create -f /tmp/environments/biotools.yml && \
    # conda env create -f /tmp/environments/nb-base.yml && \
    # conda env create -f /tmp/environments/cnapy.yml && \
    echo "conda activate biotools" >> /home/rstudio/.profile  \
    rm -rf environments

# Install the renv package and install packages from lockfile
RUN R -e "install.packages('remotes', repos = c(CRAN = 'https://cloud.r-project.org'))" && \
    R -e "remotes::install_github('rstudio/renv@${RENV_VERSION}')"

# restore from lockfile
COPY renv.lock renv.lock
# RUN R -e "renv::restore()"

# SKIP FOR NOW
# # # Copy Rstudio preferences
# COPY /rstudio-prefs.json /home/rstudio/.config/rstudio/rstudio-prefs.json
# RUN chown rstudio /home/rstudio/.config/rstudio/rstudio-prefs.json

# Create moutpoints
RUN mkdir -p /renv/cache && \
    mkdir -p /renv/root && \
    mkdir /home/rstudio/project/ && \
    mkdir -p /home/rstudio/.cache/R/R.cache && \
    chown -R rstudio:rstudio /home/rstudio/ && \
    chown -R rstudio:rstudio /renv/ 

RUN /opt/conda/envs/biotools/bin/python -m ipykernel install --prefix=/opt/conda/envs/biotools/ --name biotools
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate biotools && \
    jupyter nbextension install --py escher && \
    jupyter nbextension enable --py escher


# install cplex if available
COPY cplex/ cplex/ 
RUN cd cplex && chmod +x ./install_cplex.sh && ./install_cplex.sh && cd .. && rm -rf cplex && \
    bash -c '. /opt/conda/etc/profile.d/conda.sh && conda activate biotools && python /opt/ibm/ILOG/CPLEX_Studio201/python/setup.py install'
