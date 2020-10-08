#!/bin/bash
# Run as user instead of root to prevent permission issues
case $1 in 
    runfile)
        docker run -it --rm -v $(realpath $(pwd)):/workspace -w /workspace -u $(id -u):$(id -g) -e TF_CPP_MIN_LOG_LEVEL=2 honours python ./$2
    ;;
    runfilei)
        docker run -it --rm -v $(realpath $(pwd)):/workspace -w /workspace -u $(id -u):$(id -g) -e TF_CPP_MIN_LOG_LEVEL=2 honours python -i ./$2
    ;;
    run)
        docker run -it --rm -v $(realpath $(pwd)):/workspace -w /workspace -u $(id -u):$(id -g) -e TF_CPP_MIN_LOG_LEVEL=2 honours python -m $2
    ;;
    runi)
        docker run -it --rm -v $(realpath $(pwd)):/workspace -w /workspace -u $(id -u):$(id -g) -e TF_CPP_MIN_LOG_LEVEL=2 honours python -i -m $2
    ;;
esac