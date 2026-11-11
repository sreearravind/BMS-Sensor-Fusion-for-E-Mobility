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
````

The EKF combines coulomb counting with the OCV-based voltage measurement, correcting an initially wrong SOC estimate.

---

## Requirements

- Python 3.8+
- NumPy
- Matplotlib (optional, only for plotting)

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## Usage

Clone the repository and run the main script:

```bash
python bms_fusion.py
```

The script runs a 3000-second simulation at 1 Hz and prints a summary similar to:

```text
==============================================================
 BMS SENSOR FUSION  -  RUN SUMMARY
==============================================================
 Simulation length          : 3000 s @ 1 Hz
 Cell capacity              : 2.50 Ah
 Final true / estimated SOC :  18.4 % /  18.6 %
--------------------------------------------------------------
 SOC RMSE  (t > 300 s)      :  0.45 %
 SOC max |error|            :  1.20 %
 Pack temperature RMSE      :  0.18 degC
--------------------------------------------------------------
 Faulty NTC4 rejected       : 1498 / 1500 samples after fault (99.9 %)
 Current plausibility flags : 0
==============================================================
```

If Matplotlib is installed, the script also saves:

```text
bms_fusion_results.png
```

---

## Project Structure

```text
.
├── bms_fusion.py          # Main simulation and fusion algorithms
├── README.md              # This file
└── bms_fusion_results.png # Generated plot (optional)
```

---

## Customization

You can easily modify the demo:

- Change the random seed:

  ```python
  rng = np.random.default_rng(42)
  ```

- Change when the faulty NTC sensor starts biasing:

  ```python
  fault_started_at = 1500
  ```

- Tune sensor noise levels in:

  ```python
  RobustTemperatureFusion(sensor_var={...})
  CurrentFusion(var_shunt=..., var_hall=...)
  ```

- Tune EKF process and measurement noise:

  ```python
  SOCEKF(model, soc0=0.85, p0_soc=0.10, q_soc=0.002, r_volt=0.010)
  ```

---

## Disclaimer

This project is an educational simulation. It is **not** a certified BMS implementation and must not be used as-is in real battery systems, vehicles, or safety-critical applications. Real BMS designs require cell-level protection, functional safety analysis, hardware validation, and compliance with applicable standards.

---

## License

MIT License. See `LICENSE` for details.

---

## Contributing

Issues and pull requests are welcome. For major changes, please open an issue first to discuss the proposed change.

````
