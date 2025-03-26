FROM mcr.microsoft.com/devcontainers/miniconda:0-3

RUN apt-get update; \
    apt-get -yq install vim

COPY Project.toml Manifest.toml .
USER vscode

# Install Julia and Julia kernel
RUN curl -fsSL https://install.julialang.org | sh -s -- -y; \
    /home/vscode/.juliaup/bin/juliaup add 1.10; \
    /home/vscode/.juliaup/bin/juliaup default 1.10
RUN /home/vscode/.juliaup/bin/julia -e "using Pkg; Pkg.add(\"IJulia\"); Pkg.add(\"LanguageServer\"); Pkg.add(\"Revise\"); Pkg.activate(\".\"); Pkg.instantiate(); using MPI; MPI.install_mpiexecjl()"; \ 
    mkdir -p $HOME/.julia/config && echo "using Revise;" >> $HOME/.julia/config/startup.jl

USER root
RUN rm Project.toml Manifest.toml

