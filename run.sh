#!/bin/bash

docker run -it --rm -v $(realpath $(pwd)):/tmp -w /tmp honours python ./$1

