#!/bin/sh

docker run --rm -it \
-p 8080:8080 \
-p 8081:8081 \
-p 8082:8082 \
-p 7070:7070 \
-p 7071:7071 \
-e install_py_dep_per_model=true \
--mount type=bind,source=/home/attila/PycharmProjects/MushroomProject/model-store,target=/tmp/models \
pytorch/torchserve:latest \
torchserve --model-store=/tmp/models --models mushnet18.mar

