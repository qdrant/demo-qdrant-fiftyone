#!/bin/bash

docker run -p "6333:6333" -p "6334:6334" -d qdrant/qdrant:v0.8.0
