"""1D double-shell: heat from 20 °C, then sweep total hoop strain.

Reuses the user's interpolation.py (Granta 316 CSV data stay local).
Also exports the common material and strain-controlled shell functions
for known_initial_pressurize.py. Pressure is calculated, not prescribed,
during the strain sweep. Heating still solves equilibrium at p_g=0.

Assumptions: thin membrane, common radius, perfectly compatible hoop strain,
small strain, prescribed uniform temperature per shell, 1D rate-independent
isotropic hardening, no axial/radial stresses or geometric updating.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from interpolation import E_at, flow_stress, plastic_strain_max_at, thermal_strain

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Research presets. Temperature CANNOT be inferred from geometry alone.
PRESET_T_INNER_C = 538.0
PRESET_T_OUTER_C = 20.0
INITIAL_TEMPERATURE_C = 20.0
HEATING_MAX_STEP_C = 5.2  # automatic, roughly 101 steps for 20 -> 538 C

# Automatic numerical viewing policy, NOT a limit-load or failure criterion.
# 2% absolute hoop strain: declared small-strain study cap
AUTO_MAX_TOTAL_STRAIN = 0.020
AUTO_MIN_SWEEP = 0.005        # show at least 0.5% extra hoop strain
# extra strain after both shells develop new plasticity
AUTO_POST_BOTH_PLASTIC = 0.005
AUTO_MAX_STEP = 2e-4
AUTO_MIN_STEP = 1e-6
AUTO_PRESSURE_CURVE_TOL_PA = 3000.0  # midpoint vs straight-line deviation
AUTO_MAX_POINTS = 5000


def ask_float(label: str, default: float) -> float:
    response = input(f"{label} [{default:g}]: ").strip()
    try:
        value = float(response) if response else float(default)
    except ValueError as exc:
        raise ValueError(f"{label}: 請輸入有效數字") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: 數值必須為有限值")
    return value


def ask_int(label: str, default: int, minimum: int = 2) -> int:
    response = input(f"{label} [{default}]: ").strip()
    value = int(response) if response else default
    if value < minimum:
        raise ValueError(f"{label} 必須 >= {minimum}")
    return value


def validate_geometry(R: float, h_i: float, h_o: float) -> None:
    if min(R, h_i, h_o) <= 0:
        raise ValueError("半徑與厚度都必須大於 0")
    if max(h_i, h_o) / R >= 0.1:
        print("警告：h/R 不小，薄壁近似可能不再可靠。")


def validate_state(T: float, state: tuple[float, float]) -> None:
    eps_p, kappa = state
    if not math.isfinite(eps_p) or not math.isfinite(kappa):
        raise ValueError("塑性狀態必須為有限值")
    if kappa < 0 or abs(eps_p) > kappa + 1e-10:
        raise ValueError("需滿足 κ >= 0 且 |εp| <= κ")
    if kappa > plastic_strain_max_at(T) + 1e-12:
        raise ValueError(f"T={T:g}°C 的 κ 超過硬化資料範圍")
    if E_at(T) <= 0:
        raise ValueError("彈性模數必須為正")


def material_step(
    T: float, eps_mech: float, eps_p_old: float, kappa_old: float
) -> tuple[float, float, float]:
    """Single 1D strain increment with isotropic hardening and history.

    Returns (sigma_Pa, signed_eps_p_new, accumulated_kappa_new).
    No history is mutated during the trial evaluation.
    """
    validate_state(T, (eps_p_old, kappa_old))
    E = E_at(T)
    trial = E * (eps_mech - eps_p_old)
    yield_now = flow_stress(T, kappa_old)
    if abs(trial) <= yield_now:
        return trial, eps_p_old, kappa_old

    kappa_max = plastic_strain_max_at(T)
    delta_max = kappa_max - kappa_old
    if delta_max <= 0:
        raise ValueError("塑性增量超出原始硬化曲線範圍")

    def consistency(delta: float) -> float:
        return abs(trial) - E * delta - flow_stress(T, min(kappa_max, kappa_old + delta))

    f_max = consistency(delta_max)
    tolerance_pa = max(1e-3, 1e-10 * abs(trial))
    if f_max > tolerance_pa:
        raise ValueError("所需塑性應變超過 Granta 材料資料範圍")
    delta = delta_max if f_max >= 0 else brentq(consistency, 0.0, delta_max)

    sign = float(np.sign(trial))
    sigma = trial - sign * E * delta
    return sigma, eps_p_old + sign * delta, kappa_old + delta


def admissible_epsilon_range(T: float, state: tuple[float, float]) -> tuple[float, float]:
    """Feasible total hoop strain interval for this trial state and T."""
    validate_state(T, state)
    eps_p_old, kappa_old = state
    kappa_max = plastic_strain_max_at(T)
    eps_limit = (kappa_max - kappa_old) + flow_stress(T, kappa_max) / E_at(T)
    eps_th = thermal_strain(T)
    return eps_th + eps_p_old - eps_limit, eps_th + eps_p_old + eps_limit


def evaluate_shell(
    eps_theta: float,
    T_i: float,
    T_o: float,
    state_i_old: tuple[float, float],
    state_o_old: tuple[float, float],
):
    inner = material_step(T_i, eps_theta - thermal_strain(T_i), *state_i_old)
    outer = material_step(T_o, eps_theta - thermal_strain(T_o), *state_o_old)
    return inner, outer


def shell_step(
    p_g: float, T_i: float, T_o: float,
    R: float, h_i: float, h_o: float,
    state_i_old: tuple[float, float],
    state_o_old: tuple[float, float],
) -> dict:
    """Solve one prescribed-pressure equilibrium increment.

    Every root evaluation starts from the SAME old plastic states.
    New plastic states are returned only after equilibrium converges.
    """
    validate_geometry(R, h_i, h_o)
    lo_i, hi_i = admissible_epsilon_range(T_i, state_i_old)
    lo_o, hi_o = admissible_epsilon_range(T_o, state_o_old)
    lo, hi = max(lo_i, lo_o), min(hi_i, hi_o)
    if lo > hi:
        raise ValueError("兩殼的材料可計算應變範圍沒有交集")

    def residual(eps_theta: float) -> float:
        inner, outer = evaluate_shell(
            eps_theta, T_i, T_o, state_i_old, state_o_old)
        return h_i * inner[0] + h_o * outer[0] - p_g * R  # N/m

    f_lo, f_hi = residual(lo), residual(hi)
    balance_tol = max(1e-5, 1e-10 * max(abs(p_g * R), abs(f_lo), abs(f_hi)))
    if abs(f_lo) <= balance_tol:
        eps_theta = lo
    elif abs(f_hi) <= balance_tol:
        eps_theta = hi
    elif f_lo * f_hi > 0:
        raise ValueError(
            "指定壓力的平衡解超出材料有效範圍；"
            f"可用殘差端點為 ({f_lo:.3g}, {f_hi:.3g}) N/m"
        )
    else:
        eps_theta = brentq(residual, lo, hi, xtol=1e-13)

    inner, outer = evaluate_shell(
        eps_theta, T_i, T_o, state_i_old, state_o_old)
    result = {
        "p_g": p_g, "T_i": T_i, "T_o": T_o,
        "epsilon_theta": eps_theta, "delta_R": R * eps_theta,
        "sigma_i": inner[0], "sigma_o": outer[0],
        "epsilon_p_i": inner[1], "epsilon_p_o": outer[1],
        "kappa_i": inner[2], "kappa_o": outer[2],
        "state_i": (inner[1], inner[2]),
        "state_o": (outer[1], outer[2]),
    }
    if abs(residual(eps_theta)) > max(1e-4, 1e-9 * abs(p_g * R)):
        raise ArithmeticError("環向平衡殘差過大，求解未通過檢查")
    return result


def strain_history(
    start_epsilon: float,
    end_epsilon: float,
    steps: int,
    T_i: float,
    T_o: float,
    R: float,
    h_i: float,
    h_o: float,
    state_i: tuple[float, float],
    state_o: tuple[float, float],
) -> list[dict]:
    """Feodosyev-style strain sweep, retaining incremental plastic history.

    At each prescribed total hoop strain eps_theta, calculate the two stresses,
    then p_g from eq. (9.6)'s *hoop* equilibrium. No pressure root is solved.
    The stored state updates after each successful step.
    This is NOT the full biaxial Feodosyev method: axial equilibrium is omitted.
    """
    validate_geometry(R, h_i, h_o)
    if not math.isfinite(start_epsilon) or not math.isfinite(end_epsilon):
        raise ValueError("環向應變必須是有限值")
    if steps < 2:
        raise ValueError("應變步數需至少 2")
    if end_epsilon <= start_epsilon:
        raise ValueError(
            f"應變終點須大於起點 {start_epsilon*100:.6f}%（從初始狀態往拉伸方向掃描）"
        )
    history = []
    for j, eps_theta in enumerate(np.linspace(start_epsilon, end_epsilon, steps)):
        try:
            inner, outer = evaluate_shell(
                float(eps_theta), T_i, T_o, state_i, state_o)
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(
                f"第 {j+1}/{steps} 步，ΔR/R={eps_theta*100:.6f}%：{exc}"
            ) from exc
        sigma_i, sigma_o = inner[0], outer[0]
        p_g = (h_i * sigma_i + h_o * sigma_o) / R
        result = {
            "p_g": p_g, "T_i": T_i, "T_o": T_o,
            "epsilon_theta": float(eps_theta), "delta_R": R * float(eps_theta),
            "sigma_i": sigma_i, "sigma_o": sigma_o,
            "epsilon_p_i": inner[1], "epsilon_p_o": outer[1],
            "kappa_i": inner[2], "kappa_o": outer[2],
            "state_i": (inner[1], inner[2]),
            "state_o": (outer[1], outer[2]),
        }
        if not np.isfinite(p_g):
            raise ArithmeticError(f"第 {j+1} 步求得非有限壓力")
        history.append(result)
        # Commit the history ONLY after completing the step.
        state_i, state_o = result["state_i"], result["state_o"]
    return history


def automatic_strain_history(
    start_epsilon: float,
    T_i: float,
    T_o: float,
    R: float,
    h_i: float,
    h_o: float,
    state_i: tuple[float, float],
    state_o: tuple[float, float],
) -> tuple[list[dict], str]:
    """Adaptive *strain-controlled* loading for a reproducible display curve.

    Stop after a visible post-(both-new-plastic) segment, when possible.
    Otherwise stop at the fixed study cap or the original Granta domain.
    Neither stopping rule indicates structural failure or Feodosyev point D.
    """
    validate_geometry(R, h_i, h_o)
    validate_state(T_i, state_i)
    validate_state(T_o, state_o)
    # Both layers must remain inside the raw hardening-data domain.
    cap_i = admissible_epsilon_range(T_i, state_i)[1]
    cap_o = admissible_epsilon_range(T_o, state_o)[1]
    cap = min(AUTO_MAX_TOTAL_STRAIN, cap_i - 1e-9, cap_o - 1e-9)
    if cap <= start_epsilon + AUTO_MIN_STEP:
        raise ValueError(
            "初始應變已接近材料資料上限或 2% 小應變研究上限，無法建立向外載入曲線"
        )

    def state_at(eps: float, si: tuple[float, float], so: tuple[float, float]) -> dict:
        inner, outer = evaluate_shell(eps, T_i, T_o, si, so)
        sigma_i, sigma_o = float(inner[0]), float(outer[0])
        p_g = (h_i * sigma_i + h_o * sigma_o) / R
        if not np.isfinite(p_g):
            raise ArithmeticError("計算壓力不是有限數值")
        return {
            "p_g": p_g, "T_i": T_i, "T_o": T_o,
            "epsilon_theta": eps, "delta_R": R * eps,
            "sigma_i": sigma_i, "sigma_o": sigma_o,
            "epsilon_p_i": inner[1], "epsilon_p_o": outer[1],
            "kappa_i": inner[2], "kappa_o": outer[2],
            "state_i": (inner[1], inner[2]), "state_o": (outer[1], outer[2]),
        }

    first = state_at(float(start_epsilon), state_i, state_o)
    history = [first]
    state_i, state_o = first["state_i"], first["state_o"]
    kappa_i_initial, kappa_o_initial = state_i[1], state_o[1]
    both_plastic_at = None
    step = min(AUTO_MAX_STEP, (cap - start_epsilon) / 20)
    reason = "達到研究用最大總環向應變 2%"
    if cap < AUTO_MAX_TOTAL_STRAIN - 1e-7:
        reason = "達到 Granta 硬化資料的有效範圍"

    while history[-1]["epsilon_theta"] < cap - 1e-12:
        prev = history[-1]
        current = prev["epsilon_theta"]
        step = min(step, cap - current)
        target = current + step
        midpoint = current + step / 2
        try:
            middle = state_at(midpoint, state_i, state_o)
            endpoint = state_at(target, middle["state_i"], middle["state_o"])
        except (ValueError, ArithmeticError):
            if step <= AUTO_MIN_STEP:
                reason = "接近材料可計算範圍（自動縮步後停止）"
                break
            step = max(AUTO_MIN_STEP, step / 2)
            continue
        bend = abs(middle["p_g"] - 0.5 * (prev["p_g"] + endpoint["p_g"]))
        if bend > AUTO_PRESSURE_CURVE_TOL_PA and step > AUTO_MIN_STEP:
            step = max(AUTO_MIN_STEP, step / 2)
            continue
        history.append(endpoint)
        state_i, state_o = endpoint["state_i"], endpoint["state_o"]
        if (both_plastic_at is None
            and state_i[1] > kappa_i_initial + 1e-8
                and state_o[1] > kappa_o_initial + 1e-8):
            both_plastic_at = target
        if (both_plastic_at is not None
            and target >= max(start_epsilon + AUTO_MIN_SWEEP,
                              both_plastic_at + AUTO_POST_BOTH_PLASTIC)):
            reason = "兩殼皆有新增塑性後，已保留至少 0.5% 的後續應變觀察區間"
            break
        if bend < AUTO_PRESSURE_CURVE_TOL_PA / 8:
            step = min(AUTO_MAX_STEP, step * 1.6)
        if len(history) >= AUTO_MAX_POINTS:
            reason = "達到自動取樣點數上限；請檢查自動網格收斂"
            break

    if len(history) < 2:
        raise ValueError("自動取樣未取得足夠有效資料點")
    return history, reason


def write_pressure_csv(history: list[dict], filename: Path) -> None:
    fieldnames = [
        "pressure_MPa", "deltaR_over_R", "deltaR_over_R_percent",
        "deltaR_mm", "inner_stress_MPa", "outer_stress_MPa",
        "inner_signed_plastic_strain", "outer_signed_plastic_strain",
        "inner_accumulated_plastic_strain", "outer_accumulated_plastic_strain",
    ]
    with filename.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for r in history:
            writer.writerow({
                "pressure_MPa": r["p_g"] / 1e6,
                "deltaR_over_R": r["epsilon_theta"],
                "deltaR_over_R_percent": r["epsilon_theta"] * 100,
                "deltaR_mm": r["delta_R"] * 1000,
                "inner_stress_MPa": r["sigma_i"] / 1e6,
                "outer_stress_MPa": r["sigma_o"] / 1e6,
                "inner_signed_plastic_strain": r["epsilon_p_i"],
                "outer_signed_plastic_strain": r["epsilon_p_o"],
                "inner_accumulated_plastic_strain": r["kappa_i"],
                "outer_accumulated_plastic_strain": r["kappa_o"],
            })


def plot_history(history: list[dict], label: str, filename: Path) -> None:
    x = [r["epsilon_theta"] * 100 for r in history]
    y = [r["p_g"] / 1e6 for r in history]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, y, linewidth=2)
    ax.set_xlabel(r"Total hoop strain $\Delta R/R$ (%)")
    ax.set_ylabel(r"Calculated gas pressure $p_g$ (MPa)")
    ax.set_title(label)
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(filename, dpi=180)
    plt.show()
    plt.close(fig)


def print_state(result: dict) -> None:
    print(f"  p_g: {result['p_g']/1e6:.6f} MPa")
    print(
        f"  εθ = ΔR/R: {result['epsilon_theta']:.9f} (={result['epsilon_theta']*100:.6f} %)")
    print(f"  σθ,inner: {result['sigma_i']/1e6:.5f} MPa")
    print(f"  σθ,outer: {result['sigma_o']/1e6:.5f} MPa")
    print(
        f"  Inner: εp={result['epsilon_p_i']:.9f}, κ={result['kappa_i']:.9f}")
    print(
        f"  Outer: εp={result['epsilon_p_o']:.9f}, κ={result['kappa_o']:.9f}")


def main() -> None:
    print("\n=== 20°C 升溫後，以 ΔR/R 為控制量建立 p_g 曲線：一維雙筒 ===")
    print("先在 p_g=0 下逐步升溫，再固定各殼均勻溫度，從熱平衡位置增加環向總應變。")
    print("曲線上的 p_g 由環向力平衡計算，不預設壓力終點。")
    print("注意：尚未加入 Feodosyev 的軸向平衡，不可視為完整 §9.3 模型。")
    print("只需輸入三個幾何尺寸；溫度、範圍與數值密度使用程式內建設定。\n")

    R_mm = ask_float("共同半徑 R (mm)", 100)
    hi_mm = ask_float("Inner 厚度 hi (mm)", 1)
    ho_mm = ask_float("Outer 厚度 ho (mm)", 2)
    # Geometry only. Material, thermal loading and numerical settings are presets.
    T_i_end, T_o_end = PRESET_T_INNER_C, PRESET_T_OUTER_C
    n_heat = max(2, math.ceil(max(abs(T_i_end - INITIAL_TEMPERATURE_C),
                                  abs(T_o_end - INITIAL_TEMPERATURE_C)) /
                              HEATING_MAX_STEP_C) + 1)
    print(f"已採用固定溫度 Ti={T_i_end:g}°C, To={T_o_end:g}°C；"
          f"升溫自動取樣 {n_heat} 點。")

    R, h_i, h_o = R_mm * 1e-3, hi_mm * 1e-3, ho_mm * 1e-3
    validate_geometry(R, h_i, h_o)
    # Reject inadmissible temperatures before any state computation.
    for T in (20.0, T_i_end, T_o_end):
        _ = E_at(T)
        _ = thermal_strain(T)
        _ = flow_stress(T, 0.0)

    state_i = (0.0, 0.0)
    state_o = (0.0, 0.0)
    heating = []
    for j, fraction in enumerate(np.linspace(0.0, 1.0, n_heat)):
        Ti = 20.0 + fraction * (T_i_end - 20.0)
        To = 20.0 + fraction * (T_o_end - 20.0)
        try:
            result = shell_step(0.0, Ti, To, R, h_i, h_o, state_i, state_o)
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(
                f"升溫步驟 {j+1}/{n_heat}, Ti={Ti:g}, To={To:g}°C：{exc}"
            ) from exc
        heating.append(result)
        state_i, state_o = result["state_i"], result["state_o"]

    initial = heating[-1]
    print("\n升溫後、環向應變掃描前的平衡狀態 (p_g=0)：")
    print_state(initial)
    eps_start = initial["epsilon_theta"]
    history, stop_reason = automatic_strain_history(
        eps_start, T_i_end, T_o_end, R, h_i, h_o, state_i, state_o
    )
    print(f"\n自動取樣：{len(history)} 點；總環向應變 "
          f"{eps_start*100:.5f}% → {history[-1]['epsilon_theta']*100:.5f}%")
    print(f"停止原因：{stop_reason}")
    p0_err = abs(history[0]["p_g"] - initial["p_g"])
    if p0_err > 100.0:
        raise ArithmeticError(f"初始壓力未重現升溫平衡：差 {p0_err:.3g} Pa")

    OUTPUT_DIR.mkdir(exist_ok=True)
    initial_json = {
        "format": "double_shell_initial_state_v1",
        "description": "Result after 20°C-to-target heating at zero gas pressure; no heat-transfer PDE.",
        "R_m": R, "h_i_m": h_i, "h_o_m": h_o,
        "T_i_C": T_i_end, "T_o_C": T_o_end,
        "p_start_Pa": initial["p_g"],
        "epsilon_theta0": eps_start,
        "epsilon_p_i": state_i[0], "kappa_i": state_i[1],
        "epsilon_p_o": state_o[0], "kappa_o": state_o[1],
    }
    state_file = OUTPUT_DIR / "thermal_preload_state.json"
    state_file.write_text(json.dumps(
        initial_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n應變掃描終點：")
    print_state(history[-1])
    csv_path = OUTPUT_DIR / "heat_then_pressurize_pg_strain.csv"
    png_path = OUTPUT_DIR / "heat_then_pressurize_pg_strain.png"
    write_pressure_csv(history, csv_path)
    plot_history(
        history,
        "1D double-shell: heating followed by strain-controlled pressurization",
        png_path,
    )
    print(f"\n曲線 CSV：{csv_path}")
    print(f"曲線圖片：{png_path}")
    print(f"升溫後初始狀態 JSON：{state_file}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, ImportError, ArithmeticError) as err:
        print(f"\n計算無法完成：{err}")
        raise SystemExit(1) from err
