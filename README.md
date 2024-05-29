# Direct Use Geothermal Systems' simulation examples

[fair-software.nl](https://fair-software.nl) recommendations:

[![Github](https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue/target?=https://github.com/ychen1492/dugs_simulation_examples)](https://github.com/ychen1492/dugs_simulation_examples)
[![GitHub](https://img.shields.io/github/license/ychen1492/dugs_simulation_examples)](https://github.com/ychen1492/dugs_simulation_examples/blob/master/LICENSE.txt)


Code quality checks
<!--
[![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/ychen1492/reference-simulation/python-app.yml?branch=master)](https://github.com/ychen1492/reference-simulation/actions/workflows/python-app.yml)
[![codecov](https://codecov.io/gh/ychen1492/reference-simulation/branch/main/graph/badge.svg?token=W985RZZXSS)](https://codecov.io/gh/ychen1492/reference-simulation) -->

## What is DUGS?
Direct Use Geothermal Systems (DUGS), which are also known as low enthalpy geothermal systems, are mainly conduction mechanism dominated.
## System requirements
- Windows 10/11; Linux
- Language: Python

## Computational Dependencies
- Packages and libraries
    - Before running any py files, make sure the environment install `environment.yml`
    - Create a blank conda environment and activate it
    - Run `pip install -r <path to environment.yml>` in terminal

## Files explanation
1. `src/model.py`
    - It inherents from `DartsModel`, where the initial contion, boundary condition, reservoir type, simulation engine can be defined
    - To avoid effect of pressure and temperature on water density, mass rate control is chosen as boundary condition
    - `grav` option in `Geothermal` class is set to `True` by default.
2. `src/run_serial_resolution.py`
    - It is a main file to run multiple forward simultions to investigate the production temperature of different types of the reservoirs
    - The results of these files are csv files which have production temperature for each dx, dy and dz values
3. `src/run_serial_layers.py`
    - It is a main file to run multiple forward simulations to investigate the minimum confining layers
    - The results of these files are csv files which record the temperature and pressure of the top reservoir layer
4. `src/real_base.py`
    - It is the file which is used to generate the vtk results using the the resolution and confining layers information derived from `src/run_serial_resolution.py`.
