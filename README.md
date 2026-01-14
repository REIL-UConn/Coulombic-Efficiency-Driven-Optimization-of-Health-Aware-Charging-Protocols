# Coulombic Efficiency-Driven Optimization of Health-Aware Charging Protocols: An Experimental Investigation on LCO/Graphite Lithium-Ion Cells

This repository contains the code used in the paper **DOI to be added**

The workflow combines experimentally derived SOC-sweep Coulombic efficiency (CE) data with two complementary optimization strategies:

- **Threshold-based optimization**, where CE-threshold crossings extracted from SOC-sweep data are interpolated to define an SOC-dependent charging operating map.
- **Surrogate model-based optimization**, where CE is modeled as a continuous function of SOC, C-rate, and SOH using data-driven surrogates.

A physics-based single-particle model (eSPM) is also included for validation and plating-constrained charging analysis.

---
<p align="center">
  <img src="Graphical_Abstract.png" alt="Graphical Abstract" width="700"/>
</p>

---

## Repository Structure

### `data/`
SOC-sweep experimental datasets used for CE extraction and optimization.  
**Note:** Due to size limitation, the full datasets are **not included** in this repository. please refer to **datasets placeholder**

Expected directory structure:
- `SOC_Sweep_Data (Raw)/`  
  Raw experimental SOC-sweep measurements.
- `SOC_Sweep_Data (Processed)/`  
  Processed CE data used in the main analysis.
  
The processed datasets contain SOC-resolved CE values extracted from SOC-sweep experiments. 

### `notebooks/`
Jupyter notebooks used to generate key results and figures in the paper.
- `01_preprocessing.ipynb`: SOC-sweep preprocessing and CE extraction  
- `02_threshold_based_method.ipynb`: CE-threshold-based charging protocol optimization  
- `03_surrogate_model_based_method.ipynb`: surrogate-based CE modeling and optimization
- `04_benchmark_results.ipynb`: comparison of capacity trajectories and trade-off between charge speed and average lifetime

### `utils/`
Core Python utilities shared across notebooks.
- `soc_sweep_preprocess.py`: preprocessing utilities for SOC-sweep data  
- `soc_sweep_analysis.py`: CE computation, normalization, and SOC-based analysis  
- `threshold_boundaries.py`: CE-threshold extraction and SOC boundary definition  
- `surrogate_models.py`: surrogate fitting (polynomial, GPR), prediction, and plotting utilities
- `benchmark_capacity.py`: collecting capacity trajectory data for each strategy
- `benchmark_speed_lifetime.py`: collecting mean lifetime and charging speed for each strategy

### `single_particle_model/`
Physics-based model (PBM) implemented in MATLAB for plating-aware validation.
- `eSPM.m`: extended single-particle model  
- `current_optimizer.m`: iterative C-rate optimization under plating constraints  
- Supporting electrochemical, degradation, and voltage sub-models  
- Output files: anodic potential, plating current, and SOH evolution


---

## Citation

If you use this code, please cite:

> **Coulombic Efficiency-Driven Optimization of Health-Aware Charging Protocols: An Experimental Investigation on LCO/Graphite Lithium-Ion Cells**  
> *Authors*, Year. *(Journal / DOI TBD)*

---

## License

To be specified.
