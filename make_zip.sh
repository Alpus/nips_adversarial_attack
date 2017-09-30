#!/usr/bin/env bash

cd submit
zip -r ../submit_${1:-'untagged'}.zip ./* ../LICENSE -x "*__pycache__*" -x *.swp
