# honoursproject
Contains all work done for the honours project

## How to run in Docker
Firstly, start the docker daemon
```
    systemctl start docker
```    
Secondly, create the docker build
```
    docker build -t honours .
```
Thirdly, source the environment
```
    source env.sh
```
Then to run an arbitrary python file use:
```
    run <file.py>
```
