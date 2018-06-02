#!/usr/bin/env bash
DOCKER_REGISTRY="retrocontestrtidfxqehvzsuwpo.azurecr.io"
docker build -f $1 -t $DOCKER_REGISTRY/$2 .