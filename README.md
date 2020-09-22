# honoursproject
Contains all work done for the honours project

## How to run in Docker
Firstly, start the docker daemon
```console
    systemctl start docker
```    
Secondly, create the docker build
```console
    docker build -t honours .
```
Thirdly, source the environment
```console
    source env.sh
```
Then to run an arbitrary python file use:
```console
    run <file.py>
```
