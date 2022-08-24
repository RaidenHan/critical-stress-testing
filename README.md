# Critical Stress Testing for Time Series Forecasting in Python

*Group Member: Alex Mink, Jiayi Liu, Naman Goel, Annie Gupta, Yi-Siou Feng, Raiden Han, Shraddha Kumari*

*Mentor: Jonathan Page*

The content of this repository is a course group project for the FIM 500 at North Carolina State University for the Fall 2021 semester.

The code in this repository differs somewhat from the code used to produce the report. raiden Han has modified a significant amount of the content to accommodate automation.

## Introduction

The project's purpose was to perform stress testing on US Equity and Debt Indices using certain macroeconomic variables that were shortlisted from a larger universe of available variables. The working group was able to leverage their learnings from subjects like Fixed Income, Credit Risk, and Statistics to research and apply the knowledge in a near real-world project scenario.

## Setup

- Use the requirements.txt file to configure the Python virtual environment
- Run the main.sh in **Git Bash Terminal**, or run the following Python scripts sequentially
  - update_data.py
  - select_features.py
  - predict_features.py
  - predict_target.py
  - data_visualization.py
- Optionally, if the user wishes to specify the macroeconomic variables multiple times manually, the Python script starting from predict_features can be run manually to shorten the runtime

## History

Last Update: Aug 23, 2022

- Gold Fixing Price (GOLDAMGBD228NLBM) is not available on FRED anymore. Replace it with Export Price Index (End Use): Nonmonetary Gold (IQ12260)
