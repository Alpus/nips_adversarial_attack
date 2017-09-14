#!/usr/bin/env bash

cd submit
zip -r ../submit.zip ./* -x "*__pycache__*" -x *.swp

