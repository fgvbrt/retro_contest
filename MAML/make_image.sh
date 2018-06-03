#!/usr/bin/env bash
DOCKER_REGISTRY="retrocontestrtidfxqehvzsuwpo.azurecr.io"
docker build -f DockerFile.subm -t $DOCKER_REGISTRY/$1 .