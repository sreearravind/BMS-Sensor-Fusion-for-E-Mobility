#!/usr/bin/env python3
"""
bms_fusion.py
=============
Multi-sensor data fusion for an E-mobility Battery Management System (BMS).

A self-contained demo (NumPy only, Matplotlib optional) that solves three
classic BMS fusion problems:

  1. Redundant temperature sensing  -> robust median/MAD fusion + fault rejection
  2. Redundant current sensing      -> inverse-variance fusion + stuck-sensor diagnostics
  3. Voltage + current + cell model -> Extended Kalman Filter (EKF) for State-of-Charge

Run:
    python bms_fusion.py

Outputs:
    - Console summary (RMSE, max error, diagnostics)
    - bms_fusion_results.png  (if matplotlib is installed)

License: MIT
"""

from __future__ import annotations

from collections import deque

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")          # headless-safe
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:                  # pragma: no cover
    HAVE_MPL = False


# =============================================================================
# 1. Battery cell model  --  1-RC Thevenin equivalent circuit
# =============================================================================
class CellModel:
    """
    Simple 1-RC equivalent circuit model of a single Li-ion cell.

        V_terminal = OCV(SOC) - I * R0 - V1
        dSOC/dt    = -I / (3600 * Q_nom)
        dV1/dt     = -V1 / (R1*C1) + I / C1

    Convention: I > 0  ->  discharge,  I < 0  ->  charge (regen).
    """

    def __init__(self, capacity_ah: float = 2.5, dt: float = 1.0):
        self.Q = float(capacity_ah)      # [Ah]
        self.dt = float(dt)              # [s]

        # Ohmic + polarisation parameters (typical 18650 class cell)
        self.R0 = 0.045                  # [Ohm]
        self.R1 = 0.030                  # [Ohm]
        self.C1 = 1500.0                 # [F]
        self.tau = self.R1 * self.C1     # [s]

        # OCV(SOC) lookup table  (NMC-ish curve)
        self._soc_bp = np.array(
            [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
             0.60, 0.70, 0.80, 0.90, 0.95, 1.00]
        )
        self._ocv_bp = np.array(
            [3.00, 3.35, 3.48, 3.58, 3.63, 3.67, 3.71,
             3.75, 3.80, 3.87, 3.96, 4.05, 4.15]
        )

    # ---- model helpers -----------------------------------------------------
    def ocv(self, soc: float) -> float:
        """Open-circuit voltage as a function of SOC."""
        return float(np.interp(np.clip(soc, 0.0, 1.0), self._soc_bp, self._ocv_bp))

    def docv_dsoc(self, soc: float, h: float = 1e-3) -> float:
        """Numerical derivative dOCV/dSOC (used by the EKF Jacobian)."""
        return (self.ocv(soc + h) - self.ocv(soc - h)) / (2.0 * h)

    def step(self, soc: float, v1: float, current: float):
        """Advance the plant by one sample and return the measured terminal voltage."""
        a = np.exp(-self.dt / self.tau)
        soc_next = soc - current * self.dt / (3600.0 * self.Q)
        v1_next = a * v1 + (1.0 - a) * self.R1 * current
        v_term = self.ocv(soc_next) - current * self.R0 - v1_next
        return soc_next, v1_next, v_term


# =============================================================================
# 2. Robust fusion of redundant temperature sensors
# =============================================================================
class RobustTemperatureFusion:
    """
    Median / MAD outlier rejection followed by an inverse-variance weighted mean.

    A single biased NTC must not be able to poison the pack temperature that
    feeds the BMS derating logic -> this is a classic safety-critical fusion step.
    """

    def __init__(self, sensor_var: dict[str, float],
                 z_max: float = 3.5,
                 hard_limit: float = 8.0,
                 mad_floor: float = 0.3):
        self.sensor_var = dict(sensor_var)
        self.z_max = z_max
        self.hard_limit = hard_limit
        self.mad_floor = mad_floor

    def update(self, readings: dict[str, float]):
        """Returns (fused_temp, fused_variance, list_of_rejected_sensor_names)."""
        names = [n for n in readings if n in self.sensor_var]
        if not names:
            raise ValueError("No valid temperature sensors configured")

        vals = np.array([readings[n] for n in names], dtype=float)

        med = float(np.median(vals))
        mad = 1.4826 * float(np.median(np.abs(vals - med)))   # robust sigma
        scale = max(mad, self.mad_floor)

        keep = [i for i in range(vals.size)
                if abs(vals[i] - med) <= self.z_max * scale
                and abs(vals[i] - med) <= self.hard_limit]

        if not keep:                                  # never return empty
            keep = [int(np.argmin(np.abs(vals - med)))]

        w = np.array([1.0 / self.sensor_var[names[i]] for i in keep])
        v = vals[keep]
        est = float(np.sum(w * v) / np.sum(w))
        var = float(1.0 / np.sum(w))

        rejected = [names[i] for i in range(vals.size) if i not in keep]
        return est, var, rejected


# =============================================================================
# 3. Fusion of redundant current sensors
# =============================================================================
class CurrentFusion:
    """
    Fuses a shunt (accurate, low noise) with a Hall sensor (noisy, gain/offset
    error) using inverse-variance weighting.

    Adds two simple diagnostics:
      * stuck-sensor detection (flat output over a sliding window)
      * disagreement detection (plausibility check between channels)
    """

    def __init__(self,
                 var_shunt: float = 0.03 ** 2,
                 var_hall: float = 0.15 ** 2,
                 disagreement: float = 1.5,
                 stuck_window: int = 25,
                 stuck_tol: float = 1e-3):
        self.var_shunt = var_shunt
        self.var_hall = var_hall
        self.disagreement = disagreement
        self.stuck_window = stuck_window
        self.stuck_tol = stuck_tol

        self.buf_shunt: deque = deque(maxlen=stuck_window)
        self.buf_hall: deque = deque(maxlen=stuck_window)

        self.last = 0.0
        self.last_var = var_shunt

    def _stuck(self, buf: deque) -> bool:
        return (len(buf) == self.stuck_window
                and (max(buf) - min(buf)) < self.stuck_tol)

    def update(self, i_shunt: float, i_hall: float):
        """Returns (fused_current, fused_variance, status_string)."""
        self.buf_shunt.append(i_shunt)
        self.buf_hall.append(i_hall)

        ok_shunt = not self._stuck(self.buf_shunt)
        ok_hall = not self._stuck(self.buf_hall)
        status = "ok"

        # Plausibility check: if they disagree, trust the more accurate channel
        if ok_shunt and ok_hall and abs(i_shunt - i_hall) > self.disagreement:
            ok_hall = False
            status = "disagreement"

        if ok_shunt and ok_hall:
            ws, wh = 1.0 / self.var_shunt, 1.0 / self.var_hall
            est = (ws * i_shunt + wh * i_hall) / (ws + wh)
            var = 1.0 / (ws + wh)
        elif ok_shunt:
            est, var = i_shunt, self.var_shunt
            if status == "ok":
                status = "degraded"      # Hall dropped out
        elif ok_hall:
            est, var = i_hall, self.var_hall
            status = "degraded"          # Shunt dropped out
        else:
            est, var = self.last, self.last_var
            status = "invalid"           # Hold last known good value

        self.last, self.last_var = est, var
        return est, var, status


# =============================================================================
# 4. Extended Kalman Filter for State-of-Charge
# =============================================================================
class SOCEKF:
    """
    EKF that fuses coulomb counting (current sensor) with the OCV measurement
    (voltage sensor) through the 1-RC cell model.

        x = [SOC, V1]^T
        f(x, i) : model-based prediction   (coulomb counting + RC dynamics)
        h(x, i) : predicted terminal voltage
    """

    def __init__(self, model: CellModel,
                 soc0: float = 0.85,
                 p0_soc: float = 0.10,
                 p0_v1: float = 0.01,
                 q_soc: float = 0.002,
                 q_v1: float = 0.002,
                 r_volt: float = 0.010):
        self.m = model
        self.x = np.array([float(soc0), 0.0])
        self.P = np.diag([p0_soc ** 2, p0_v1 ** 2])
        self.Q = np.diag([q_soc ** 2, q_v1 ** 2])
        self.R = r_volt ** 2
        self.innovation = 0.0

    # ---- process model -----------------------------------------------------
    def _f(self, x: np.ndarray, i: float) -> np.ndarray:
        a = np.exp(-self.m.dt / self.m.tau)
        soc = x[0] - i * self.m.dt / (3600.0 * self.m.Q)
        v1 = a * x[1] + (1.0 - a) * self.m.R1 * i
        return np.array([soc, v1])

    # ---- measurement model -------------------------------------------------
    def _h(self, x: np.ndarray, i: float) -> float:
        return self.m.ocv(x[0]) - i * self.m.R0 - x[1]

    # ---- EKF steps ---------------------------------------------------------
    def predict(self, current: float) -> None:
        a = np.exp(-self.m.dt / self.m.tau)
        F = np.array([[1.0, 0.0], [0.0, a]])
        self.x = self._f(self.x, current)
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, current: float, v_meas: float):
        H = np.array([[self.m.docv_dsoc(self.x[0]), -1.0]])
        y = v_meas - self._h(self.x, current)

        S = float(H @ self.P @ H.T) + self.R
        K = (self.P @ H.T).ravel() / S

        self.x = self.x + K * y
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)

        # Joseph form for numerical stability
        I = np.eye(2)
        A = I - np.outer(K, H)
        self.P = A @ self.P @ A.T + np.outer(K, K) * self.R

        self.innovation = y
        return y, S

    @property
    def soc(self) -> float:
        return float(self.x[0])

    @property
    def soc_std(self) -> float:
        return float(np.sqrt(max(self.P[0, 0], 0.0)))


# =============================================================================
# 5. Synthetic e-scooter / e-bike drive cycle
# =============================================================================
def build_drive_cycle(n_steps: int, rng: np.random.Generator) -> np.ndarray:
    """
    Deterministic-but-jittered urban drive cycle for a light electric vehicle.

    Positive current = traction (discharge), negative = regenerative braking.
    """
    segments = [
        (6.5, 8),     # hard acceleration
        (2.0, 30),    # cruise
        (-3.0, 6),    # regen braking
        (0.0, 12),    # standstill
        (9.0, 6),     # hill climb
        (1.5, 35),    # low-speed cruise
        (-2.5, 8),    # regen
        (0.0, 20),    # stop
    ]

    profile = np.empty(n_steps, dtype=float)
    k = 0
    while k < n_steps:
        base, dur = segments[int(rng.integers(len(segments)))]
        dur = max(1, int(dur * rng.uniform(0.6, 1.4)))
        size = min(dur, n_steps - k)
        seg = np.full(size, base, dtype=float) + rng.normal(0.0, 0.15, size)
        profile[k:k + size] = seg
        k += size
    return profile


# =============================================================================
# 6. Main simulation
# =============================================================================
def main() -> None:
    # ---- configuration -----------------------------------------------------
    dt = 1.0
    n_steps = 3000                       # 3000 s @ 1 Hz
    rng = np.random.default_rng(42)

    model = CellModel(capacity_ah=2.5, dt=dt)
    profile = build_drive_cycle(n_steps, rng)

    # Fusion blocks
    temp_fusion = RobustTemperatureFusion(
        sensor_var={"NTC1": 0.25 ** 2, "NTC2": 0.25 ** 2,
                    "NTC3": 0.25 ** 2, "NTC4": 0.25 ** 2}
    )
    current_fusion = CurrentFusion()

    # EKF deliberately starts with a wrong SOC (0.85 vs true 0.95)
    ekf = SOCEKF(model, soc0=0.85, p0_soc=0.10)

    # ---- true plant state --------------------------------------------------
    soc_true = 0.95
    v1_true = 0.0
    temp_true = 25.0

    # ---- logging -----------------------------------------------------------
    keys = ["soc_true", "soc_est", "soc_std", "v_true", "v_meas",
            "i_true", "i_shunt", "i_hall", "i_fused",
            "temp_true", "temp_fused", "temp_ntc4"]
    log = {k: np.zeros(n_steps) for k in keys}

    rejected_after_fault = 0
    disagreement_count = 0
    fault_started_at = 1500

    # ---- simulation loop ---------------------------------------------------
    for k in range(n_steps):
        i_true = float(profile[k])

        # --- ground truth ---------------------------------------------------
        soc_true, v1_true, v_term = model.step(soc_true, v1_true, i_true)

        heat = i_true * i_true * model.R0
        ambient = 25.0 + 2.5 * np.sin(2.0 * np.pi * k / 1200.0)
        temp_true += dt * ((ambient + 2.0 * heat) - temp_true) / 150.0

        # --- sensors (with realistic noise and faults) ----------------------
        v_meas = v_term + rng.normal(0.0, 0.010)                 # 10 mV noise
        i_shunt = i_true + rng.normal(0.0, 0.03)                 # accurate
        i_hall = 1.02 * i_true + 0.15 + rng.normal(0.0, 0.15)    # gain + offset

        ntc4_bias = 12.0 if k >= fault_started_at else 0.0
        ntc = {
            "NTC1": temp_true + rng.normal(0.0, 0.25),
            "NTC2": temp_true + rng.normal(0.0, 0.25),
            "NTC3": temp_true + rng.normal(0.0, 0.25),
            "NTC4": temp_true + rng.normal(0.0, 0.25) + ntc4_bias,
        }

        # --- FUSION ---------------------------------------------------------
        t_fused, _, rejected = temp_fusion.update(ntc)
        i_fused, _, c_status = current_fusion.update(i_shunt, i_hall)

        ekf.predict(i_fused)
        ekf.update(i_fused, v_meas)

        # --- diagnostics ----------------------------------------------------
        if k >= fault_started_at and "NTC4" in rejected:
            rejected_after_fault += 1
        if c_status in ("disagreement", "invalid"):
            disagreement_count += 1

        # --- logging --------------------------------------------------------
        log["soc_true"][k] = soc_true
        log["soc_est"][k] = ekf.soc
        log["soc_std"][k] = ekf.soc_std
        log["v_true"][k] = v_term
        log["v_meas"][k] = v_meas
        log["i_true"][k] = i_true
        log["i_shunt"][k] = i_shunt
        log["i_hall"][k] = i_hall
        log["i_fused"][k] = i_fused
        log["temp_true"][k] = temp_true
        log["temp_fused"][k] = t_fused
        log["temp_ntc4"][k] = ntc["NTC4"]

    # ---- metrics -----------------------------------------------------------
    skip = 300                                        # ignore EKF convergence
    err_soc = log["soc_est"][skip:] - log["soc_true"][skip:]
    rmse_soc = float(np.sqrt(np.mean(err_soc ** 2)))
    max_soc = float(np.max(np.abs(err_soc)))

    err_t = log["temp_fused"] - log["temp_true"]
    rmse_t = float(np.sqrt(np.mean(err_t ** 2)))

    n_fault_window = n_steps - fault_started_at
    detect_rate = 100.0 * rejected_after_fault / max(n_fault_window, 1)

    print("\n" + "=" * 62)
    print(" BMS SENSOR FUSION  -  RUN SUMMARY")
    print("=" * 62)
    print(f" Simulation length          : {n_steps} s @ {dt:.0f} Hz")
    print(f" Cell capacity              : {model.Q:.2f} Ah")
    print(f" Final true / estimated SOC : {log['soc_true'][-1]*100:5.1f} % /"
          f" {log['soc_est'][-1]*100:5.1f} %")
    print("-" * 62)
    print(f" SOC RMSE  (t > {skip} s)     : {rmse_soc*100:5.2f} %")
    print(f" SOC max |error|            : {max_soc*100:5.2f} %")
    print(f" Pack temperature RMSE      : {rmse_t:5.2f} degC")
    print("-" * 62)
    print(f" Faulty NTC4 rejected       : {rejected_after_fault}"
          f" / {n_fault_window} samples after fault ({detect_rate:.1f} %)")
    print(f" Current plausibility flags : {disagreement_count}")
    print("=" * 62)

    # ---- plot --------------------------------------------------------------
    if not HAVE_MPL:
        print("\n[info] matplotlib not installed - skipping plot.")
        return

    t = np.arange(n_steps) * dt
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("BMS multi-sensor data fusion (E-mobility)", fontsize=14)

    # SOC
    a = ax[0, 0]
    a.plot(t, log["soc_true"] * 100, "k-", lw=2, label="True SOC")
    a.plot(t, log["soc_est"] * 100, "r--", lw=1.6, label="EKF estimate")
    band = 2 * log["soc_std"] * 100
    a.fill_between(t, log["soc_est"] * 100 - band, log["soc_est"] * 100 + band,
                   color="red", alpha=0.15, label="±2σ")
    a.set_title("State of Charge (EKF fusion)")
    a.set_xlabel("time [s]"); a.set_ylabel("SOC [%]")
    a.grid(alpha=0.3); a.legend(loc="lower left")

    # Voltage
    a = ax[0, 1]
    a.plot(t, log["v_meas"], color="0.7", lw=0.8, label="Measured")
    a.plot(t, log["v_true"], "k-", lw=1.0, label="True terminal V")
    a.set_title("Terminal voltage")
    a.set_xlabel("time [s]"); a.set_ylabel("voltage [V]")
    a.grid(alpha=0.3); a.legend(loc="upper right")

    # Current
    a = ax[1, 0]
    a.plot(t, log["i_true"], "k-", lw=1.2, label="True")
    a.plot(t, log["i_shunt"], color="tab:blue", lw=0.7, alpha=0.7, label="Shunt")
    a.plot(t, log["i_hall"], color="tab:orange", lw=0.7, alpha=0.7, label="Hall")
    a.plot(t, log["i_fused"], "r--", lw=1.3, label="Fused")
    a.set_title("Current sensor fusion")
    a.set_xlabel("time [s]"); a.set_ylabel("current [A]")
    a.grid(alpha=0.3); a.legend(loc="upper right", ncol=2, fontsize=8)

    # Temperature
    a = ax[1, 1]
    a.plot(t, log["temp_true"], "k-", lw=1.6, label="True cell temp")
    a.plot(t, log["temp_ntc4"], ":", color="tab:red", lw=1.2,
           label="NTC4 (faulty after 1500 s)")
    a.plot(t, log["temp_fused"], color="tab:green", lw=1.6, label="Fused")
    a.axvline(fault_started_at, color="red", ls="--", alpha=0.5)
    a.set_title("Robust temperature fusion (median/MAD)")
    a.set_xlabel("time [s]"); a.set_ylabel("temperature [°C]")
    a.grid(alpha=0.3); a.legend(loc="lower right", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = "bms_fusion_results.png"
    fig.savefig(out, dpi=130)
    print(f"\n[info] Plot saved to ./{out}")


if __name__ == "__main__":
    main()
