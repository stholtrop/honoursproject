#!/bin/bash
# Run as user instead of root to prevent permission issues
docker run -it --rm -v $(realpath $(pwd)):/tmp -w /tmp -u $(id -u):$(id -g) honours python -i ./$1

