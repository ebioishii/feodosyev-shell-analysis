from pathlib import Path
import csv
import numpy as np

DATA_DIR = (
    Path(__file__).resolve().parent
    / "database"
    / "Granta_SS316_annealed_csv"
)


def read_csv(filename):
    with (DATA_DIR / filename).open(
        encoding="utf-8-sig", newline=""
    ) as f:
        return list(csv.DictReader(f))


def linear_interp(x, x_data, y_data):
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    order = np.argsort(x_data)
    x_data = x_data[order]
    y_data = y_data[order]

    # 基本資料檢查
    if len(x_data) < 2 or np.any(np.diff(x_data) <= 0):
        raise ValueError("插值節點不足或存在重複的 x")

    if not np.all(np.isfinite(x_data)) or not np.all(np.isfinite(y_data)):
        raise ValueError("材料資料包含無效數值")

    # 禁止外插
    if not x_data[0] <= x <= x_data[-1]:
        raise ValueError(f"{x} 超出資料範圍")

    return float(np.interp(x, x_data, y_data))


# 讀取彈性模數
elastic_rows = read_csv("Elasticity.csv")

T_E = np.array([
    float(r["Temperature [C]"]) for r in elastic_rows
])

E_Pa = np.array([
    float(r["Young's Modulus [Pa]"]) for r in elastic_rows
])


def E_at(T_C):
    return linear_interp(T_C, T_E, E_Pa)


# 讀取割線熱膨脹係數
cte_rows = read_csv("Coefficient of Thermal Expansion.csv")

T_alpha = np.array([
    float(r["Temperature [C]"]) for r in cte_rows
])

alpha_values = np.array([
    float(r["Coefficient of Thermal Expansion [C^-1]"])
    for r in cte_rows
])

# 讀取零熱應變參考溫度
ref_rows = read_csv(
    "Zero-Thermal-Strain Reference Temperature.csv"
)

if len(ref_rows) != 1:
    raise ValueError("預期只有一個零熱應變參考溫度")

T_ref = float(
    ref_rows[0]["Zero-Thermal-Strain Reference Temperature [C]"]
)


def alpha_at(T_C):
    return linear_interp(T_C, T_alpha, alpha_values)


def thermal_strain(T_C):
    return alpha_at(T_C) * (T_C - T_ref)


hardening_rows = read_csv("Isotropic Hardening.csv")

# 依照溫度分組
hardening = {}

for r in hardening_rows:
    T = float(r["Temperature [C]"])
    eps_p = float(r["Plastic Strain [m m^-1]"])
    stress = float(r["Stress [Pa]"])

    hardening.setdefault(T, []).append((eps_p, stress))

# 將各溫度的節點整理成 numpy array
for T, points in hardening.items():
    data = np.asarray(sorted(points), dtype=float)

    eps_p = data[:, 0]
    stress = data[:, 1]

    # 檢查初始降伏點及節點順序
    if not np.isclose(eps_p[0], 0, atol=1e-12):
        raise ValueError(f"{T}°C 沒有零塑性應變節點")

    if np.any(np.diff(eps_p) <= 0):
        raise ValueError(f"{T}°C 塑性應變節點異常")

    if np.any(np.diff(stress) < 0):
        raise ValueError(f"{T}°C 流動應力節點非單調")

    hardening[T] = (eps_p, stress)

T_hard = np.array(sorted(hardening))


def flow_stress(T_C, eps_p):
    # 禁止溫度外插
    if not T_hard[0] <= T_C <= T_hard[-1]:
        raise ValueError("溫度超出硬化曲線資料範圍")

    upper = int(np.searchsorted(T_hard, T_C))

    # 情況 1：材料庫剛好有這個溫度
    if upper < len(T_hard) and T_hard[upper] == T_C:
        eps, stress = hardening[T_hard[upper]]
        return linear_interp(eps_p, eps, stress)

    # 情況 2：需要溫度插值
    T_low = T_hard[upper - 1]
    T_high = T_hard[upper]

    eps_low, stress_low = hardening[T_low]
    eps_high, stress_high = hardening[T_high]

    # 先沿塑性應變插值
    s_low = linear_interp(eps_p, eps_low, stress_low)
    s_high = linear_interp(eps_p, eps_high, stress_high)

    # 再沿溫度插值
    return linear_interp(
        T_C,
        [T_low, T_high],
        [s_low, s_high]
    )


def plastic_strain_max_at(T_C):

    if not T_hard[0] <= T_C <= T_hard[-1]:
        raise ValueError("溫度超出硬化資料範圍")

    idx = np.searchsorted(T_hard, T_C)

    # 原始資料剛好有這個溫度
    if idx < len(T_hard) and T_hard[idx] == T_C:
        return float(hardening[T_hard[idx]][0][-1])

    # 需要在兩條溫度曲線間插值
    T_low = T_hard[idx - 1]
    T_high = T_hard[idx]

    max_low = hardening[T_low][0][-1]
    max_high = hardening[T_high][0][-1]

    return float(min(max_low, max_high))
