#!/bin/sh

# You must source this script to activate the virtualenv in your shell.


# source my_script.sh

ENV_NAME=_env

PYTHON=python3.10

# Disable the Pygame output
export PYGAME_HIDE_SUPPORT_PROMPT="hide"

# Create the virtualenv only if it doesn't exist
if [ ! -d "$ENV_NAME" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv $ENV_NAME
    echo "Virtual environment has been created."
else
    echo "Virtual environment already exists. Skipping creation."
fi
echo

# Activate the virtualenv.
source $ENV_NAME/bin/activate
echo "The virtual environment is activated."
echo

# Install the requirements in the virtualenv.
echo "Installing the requirements..."
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt
echo

alias norminette_python=flake8

$PYTHON main.py --help
