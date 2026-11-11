# BMS Sensor Fusion for E-Mobility

A simple, self-contained Python project demonstrating **multi-sensor data fusion** for an E-mobility Battery Management System (BMS). The demo fuses redundant temperature sensors, redundant current sensors, and voltage/current measurements to estimate the battery **State of Charge (SOC)** using an Extended Kalman Filter (EKF).

This project is intended for learning, prototyping, and GitHub portfolios. It is **not** production safety software.

---

## Features

- **Robust temperature fusion** using median/MAD outlier rejection and inverse-variance weighted averaging.
- **Redundant current fusion** between a shunt and a Hall sensor, including stuck-sensor and disagreement diagnostics.
- **SOC estimation** with an Extended Kalman Filter (EKF) based on a 1-RC equivalent circuit cell model.
- **Synthetic e-scooter drive cycle** with acceleration, cruise, regenerative braking, and standstill phases.
- **Fault injection demo**: one NTC temperature sensor becomes biased after 1500 s and is automatically rejected.
- **Console summary** with SOC RMSE, temperature RMSE, and fault-detection statistics.
- **Optional Matplotlib plot** saved as `bms_fusion_results.png`.

---

## Fusion Modules

### 1. Robust Temperature Fusion

Fuses four NTC sensor readings. A biased sensor is rejected using a robust median/MAD test, and the remaining sensors are combined with inverse-variance weighting.

### 2. Current Sensor Fusion

Fuses a precise shunt reading with a noisier Hall-effect sensor reading. The module also flags stuck sensors and implausible disagreements between channels.

### 3. SOC Estimation with EKF

Uses the fused current and measured terminal voltage to estimate SOC through a 1-RC Thevenin cell model:

```text
V_terminal = OCV(SOC) - I * R0 - V1
dSOC/dt    = -I / (3600 * Q_nom)
dV1/dt     = -V1 / (R1*C1) + I / C1
