#!/bin/bash
 
# Variables
SIF_NAME="file_processing.sif"
SIF_DEF="file_processing.def"

echo "Apptaining from filemagic.sh"
# Comment this out if you don't want to build the sif

apptainer build $SIF_NAME $SIF_DEF

apptainer exec $SIF_NAME python3 file_magic.py