#!/bin/bash

set -e

# Build Docker image
docker build \
  --tag=lawchen_droid_image:1.0 \
  .
