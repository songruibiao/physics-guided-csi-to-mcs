# Structure-Preserving Physics-Guided CSI-to-MCS Learning under Uncalibrated Link Logs

This repository contains the code, sample data, and experiment scripts for the paper:

**Structure-Preserving Physics-Guided CSI-to-MCS Learning under Uncalibrated Link Logs**

## Overview

In operational broadband wireless links, link adaptation logs often contain only channel state information (CSI) and a small set of implementation-layer statistics, while the BLER curves, coding configurations, and offline calibration information required by conventional EESM/MI-EESM methods are unavailable.

To address this issue, this repository implements a **structure-preserving physics-guided learning framework** for CSI-to-MCS prediction under uncalibrated link logs. The framework retains an MI-EESM-type log-sum-exp aggregation as a physical interface and enables restricted sample-wise adaptation through low-dimensional degradation features.

## Main Features

- Structure-preserving MI-EESM-type aggregation
- Sample-wise adaptive parameters \((\beta_t, w_t, b_t)\)
- Physics-guided ordinal prior over MCS levels
- Additive logit fusion of physical and data-driven branches
- Evaluation under random and distribution-shift settings

## Repository Structure

```text
physics-guided-csi-to-mcs/
├── README.md
├── LICENSE
├── data/
│   ├── sample/
│   └── README.md
├── scripts/
├── src/
├── results/
├── figs/
├── docs/
└── models/
