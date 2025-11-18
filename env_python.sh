#!/bin/sh

# You must source this script to activate the virtualenv in your shell.


# source my_script.sh

ENV_NAME=_venv


PYTHON=python3.10

# Disable the Pygame output
export PYGAME_HIDE_SUPPORT_PROMPT="hide"

# Create a virtualenv based on the ENV_NAME variable.
$PYTHON -m venv $ENV_NAME
echo "virtual environment has been created."
echo

# Activate the virtualenv.
source $ENV_NAME/bin/activate
echo "The virtual environment is activated."
echo

# Install the requirements in the virtualenv.
echo "Installing the requirements..."
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirement.txt
echo

$PYTHON ./srcs/main.py --help
