#!/bin/bash
set -e

python freeze.py
tar cvzf neural-0.1.tar.gz neural-0.1/
