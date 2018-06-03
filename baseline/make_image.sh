#!/usr/bin/env bash
DOCKER_REGISTRY="retrocontestrtidfxqehvzsuwpo.azurecr.io"
docker build -f ppo2.subm.docker -t $DOCKER_REGISTRY/$1 .