# FPGA Routing Congestion Prediction

This repository contains the experimental code for the paper "FPGA Routing Congestion Prediction via Graph Learning-Aided Conditional GAN".

## Overview

The project explores the use of graph learning techniques combined with Conditional Generative Adversarial Networks (CGAN) to predict routing congestion in FPGA designs. The approach leverages heterogeneous graph datasets constructed from multiple layouts and routing scenarios to train a predictive model that can effectively estimate congestion in unseen FPGA layouts.

## Getting Started

Follow these steps to set up and run the experiments from this repository.

### Prerequisites

Ensure you have the appropriate version of the VTR toolchain installed. This project is based on VTR version 8.0.0.

### Installation

1. **Install VTR 8.0.0:**
   - Follow the installation instructions available at [VTR's official documentation](https://docs.verilogtorouting.org/en/latest/).

### Usage

1. **Generate Layout Instances:**
   - Run the script `vpr_run.py` located in the `/script` directory. This script executes 200 different layout instances for each circuit in the VTR7 benchmark.

2. **Create Heterogeneous Graph Dataset:**
   - Construct the graph dataset based on the layout and routing data. This involves processing the VTR output to extract relevant features and graph structures.

3. **Model Training and Testing:**
   - Navigate to the `/source` directory.
   - Run `train_<model>.py` to train and test the congestion prediction model. Ensure you have the necessary Python packages installed.
