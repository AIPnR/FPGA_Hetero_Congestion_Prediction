# FPGA Routing Congestion Prediction

This repository contains the experimental code for the paper "FPGA Routing Congestion Prediction via Graph Learning-Aided Conditional GAN".

## Overview

The project explores the use of graph learning techniques combined with Conditional Generative Adversarial Networks (CGAN) to predict routing congestion in FPGA designs. The approach leverages heterogeneous graph datasets constructed from multiple layouts and routing scenarios to train a predictive model that can effectively estimate congestion in unseen FPGA layouts.


## Repository Structure

### 1. arch_blif_source

This directory contains a subset of BLIF files from the VTR7 benchmarks. Additional BLIF files can be accessed within the official VTR project repository.

### 2. script

Contained within this directory are scripts essential for the automation and analysis of the VTR benchmarks:
- Multi-threading scripts for VTR execution
- Scripts for parsing outputs from the VTR
- Scripts for dataset generation

### 3. source

This directory hosts the training and model scripts for various machine learning models employed in this research.


## Getting Started

Due to the large size of the dataset, it has not been uploaded. Follow these steps to rebuild Datset and run the experiments. Visualizations of prediction results and evaluation metrics can be found in the source/check_points directory for reference.

1. **Install VTR 8.0.0:**
   - Follow the installation instructions available at [VTR's official documentation](https://docs.verilogtorouting.org/en/latest/).

2. **Generate Layout Instances:**
   - Run the script `vpr_run.py` located in the `/script` directory. This script executes 200 different layout instances for each circuit in the VTR7 benchmark.

3. **Create Heterogeneous Graph Dataset:**
   - Construct the graph dataset based on the layout and routing data. This involves processing the VTR output to extract relevant features and graph structures.

5. **Model Training and Testing:**
   - Navigate to the `/source` directory.
   - Run `train_<model>.py` to train and test the congestion prediction model. Ensure you have the necessary Python packages installed.
  
## Citations
If you find our paper/code useful in your research, please citing

https://doi.org/10.1145/3773770

