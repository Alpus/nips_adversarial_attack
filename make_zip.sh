#!/usr/bin/env bash

cd submit
zip -r ../submit_${1:-'untagged'}.zip ./* -x "*__pycache__*" -x *.swp

