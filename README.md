# Coulombic Efficiency-Driven Optimization of Health-Aware Charging Protocols  
### An Experimental Investigation on LCO/Graphite Lithium-Ion Cells

This repository contains the code used in the paper  
**“Coulombic Efficiency-Driven Optimization of Health-Aware Charging Protocols: An Experimental Investigation on LCO/Graphite Lithium-Ion Cells.”**

The workflow combines experimentally derived SOC-sweep Coulombic efficiency (CE) data with two complementary optimization strategies:

- **Threshold-based optimization**, where CE-threshold crossings extracted from SOC-sweep data are interpolated to define an SOC-dependent charging operating map.
- **Surrogate model-based optimization**, where CE is modeled as a continuous function of SOC, C-rate, and SOH using data-driven surrogates.

A physics-based single-particle model (eSPM) is also included for validation and plating-constrained charging analysis.

---

## Graphical Abstract (placeholder)

<p align="center">
  <img src="assets/Graphical_Abstract.png" alt="Graphical Abstract" width="900"/>
</p>

---

## Repository Structure

### `data/`
SOC-sweep experimental datasets.
- `SOC sweep data (raw)/`: raw experimental SOC-sweep measurements  
- `SOC sweep data (processed)/`: processed CE data used in the main analysis  
- `SOC sweep data (processed)_v2/`: updated/alternative processed dataset

### `notebooks/`
Jupyter notebooks used to generate all results and figures in the paper.
- `01_preprocessing.ipynb`: SOC-sweep preprocessing and CE extraction  
- `02_threshold_based_method.ipynb`: CE-threshold-based charging protocol optimization  
- `03_surrogate_model_based_method.ipynb`: surrogate-based CE modeling and optimization

### `utils/`
Core Python utilities shared across notebooks.
- `soc_sweep_preprocess.py`: preprocessing utilities for SOC-sweep data  
- `soc_sweep_analysis.py`: CE computation, normalization, and SOC-based analysis  
- `threshold_boundaries.py`: CE-threshold extraction and SOC boundary definition  
- `surrogate_models.py`: surrogate fitting (polynomial, GPR), prediction, and plotting utilities

### `single_particle_model/`
Physics-based model (PBM) implemented in MATLAB for plating-aware validation.
- `eSPM.m`: extended single-particle model  
- `current_optimizer.m`: iterative C-rate optimization under plating constraints  
- Supporting electrochemical, degradation, and voltage sub-models  
- Output files: anodic potential, plating current, and SOH evolution

---

## Notes

- SOC and SOH are expressed in **percent (%)**, C-rate in **C**.
- CE is reported in **percent (%)** and shifted to start at 100% for visualization consistency.
- The repository reflects the exact structure and analysis used in the manuscript.

---

## Citation

If you use this code, please cite:

> **Coulombic Efficiency-Driven Optimization of Health-Aware Charging Protocols: An Experimental Investigation on LCO/Graphite Lithium-Ion Cells**  
> *Authors*, Year. *(Journal / DOI TBD)*

---

## License

To be specified.
