#!/bin/bash
# Run as user instead of root to prevent permission issues
docker run -it --rm -v $(realpath $(pwd)):/tmp -w /tmp -u $(id -u):$(id -g) -e TF_CPP_MIN_LOG_LEVEL=2 honours python ./$1

