# Feodosyev Shell Analysis

用 Python 建立可追溯的再生冷卻結構分析工具。目前包含材料資料管理、**單調載入**應力–應變曲線、Phase 3A 的**僅環向線彈性**雙殼模型，以及 Phase 3B 的**雙軸線彈性與合成材料增量**模型。這些簡化模型尚非實際引擎強度或壽命預測。

## 快速開始

安裝 [uv](https://docs.astral.sh/uv/) 後，在此資料夾執行：

```powershell
uv python install 3.12
uv sync --locked --extra dev
uv run --locked pytest
uv run --locked ruff check .
uv run --locked jupyter lab
```

開啟 `notebooks/01_material_curves.ipynb`，從第一格依序執行。範例資料在 `examples/synthetic/`，全部是人工數值，供操作與程式測試，**不能作為 SS316 材料證據**。圖面可旋轉、縮放、查值；Notebook 會將離線 HTML 與重現紀錄寫入 `outputs/`。

如已有 Python 3.12，uv 可以使用現有安裝。`python` 指令若指向 Windows Store 但無法啟動，仍可透過 uv 管理專案環境。

## 資料與單位

內部統一使用 K、Pa、m、無因次應變。圖面使用 °C、MPa、應變百分比，軸標清楚區分。每一資料集由 `material.json` 加 CSV 表格組成；檔案保留材料條件、來源狀態、處理方式、單位及曲線定義。資料缺漏會明確顯示，不由程式猜測或填補。

目前外部參考資料放在 `D:\LAB\regen\references`，不屬於 Git Repository。若需匯入現有三份 SS316 CSV，可在 Python 環境中執行：

```python
from feodosyev_shell.materials.legacy import import_legacy_ss316
material = import_legacy_ss316(
    r"D:\LAB\regen\references\SS316_data", "data/processed/ss316_local"
)
print(material.completeness_report())
```

匯入會處理彈性模數、降伏強度、泊松比，以及以 0 °C 為參考的**平均熱膨脹係數**。這個匯入函式對 `SS316_plastic_StressStrain.csv` 僅產生品質摘要。若要使用暫用曲線，請使用下面的獨立流程。來源、材料狀態及使用權仍待核對。`data/raw/`、`data/processed/` 與 `outputs/` 不提交 GitHub。

### Phase 2：暫用 SS316 曲線

在 VS Code 開啟 `notebooks/02_provisional_ss316.ipynb`，選擇專案 `.venv` 的 Python 核心，從第一格依序執行。不需要另外啟動 JupyterLab。也可以在你的 `test.ipynb` 執行：

```python
from feodosyev_shell.materials.provisional import prepare_provisional_ss316
from feodosyev_shell.constitutive import curve_strain_intervals, evaluate_curve

curves = prepare_provisional_ss316(
    r"D:\LAB\regen\references\SS316_data\SS316_plastic_StressStrain.csv",
    "data/processed/ss316_provisional_v1",
)
temperature = 400 + 273.15
print(curve_strain_intervals(curves, temperature))
stress = evaluate_curve(curves, temperature, 0.2,
                       kind="engineering", allow_provisional=True)
print(stress / 1e6, "MPa")  # 0.2 是 20% 工程應變
```

程式只保留可辨識、沒有重複應變座標的連續區段。每段至少兩點；不平均重複值、不補初始段、不跨越排除區域。查詢缺口或範圍外會報錯。`allow_provisional=True` 明確表示接受暫用資料作探索；不代表材料已驗證。

Notebook 會輸出二維圖、三維離線 HTML、插值 CSV 與重現紀錄至 `outputs/phase2_provisional/`。每筆原始資料的去留原因保存在新資料集的 `selection_audit.csv`。既有 `ss316_local` 與新曲線分開，避免把未確認相容的來源混合。

工程量轉真量的公式已在合成範例驗證；暫用 SS316 的均勻變形範圍未確認，因此不自動轉成真塑性曲線。完整處理規則與驗收狀態見 [Phase 2 說明](docs/phase2.md)。

### 平均熱膨脹係數的使用流程

在 VS Code 的 PowerShell 終端機，於專案根目錄執行 `uv run --locked python`，再逐行輸入：

```python
from feodosyev_shell.materials import load_material, accumulated_thermal_strain, thermal_strain
from feodosyev_shell.constitutive import mechanical_hoop_strain

material = load_material("data/processed/ss316_local")
beta_100 = accumulated_thermal_strain(material, 100 + 273.15)
beta_315 = accumulated_thermal_strain(material, 315 + 273.15)
eps_th = thermal_strain(material, 315 + 273.15, 100 + 273.15)
print(beta_100, beta_315, eps_th)
# 0.00159, 0.005103, 0.003513 (約略值；無因次)
```

原始表格的平均係數以 **0 °C** 為來源基準：先在每個資料點算 `β(T) = ᾱ(0→T) × (T − 0 °C)`，再對 `β` 做分段線性插值。`thermal_strain` 另扣除案例選定的**無應力參考溫度** `T_ref`：`ε_th = β(T) − β(T_ref)`。上述 100 °C 只是操作示例，**不是已確認的 SS316 無應力溫度**。函式輸入一律用 K；溫差的數值與 °C 相同。

若薄殼已有指定的徑向位移 `ΔR`、參考半徑 `R` 和熱應變，可計算 `mechanical_hoop_strain(ΔR, R, eps_th)`，即 `ΔR/R − ε_th`。這只計算小變形環向機械應變；它不會求出壓力造成的位移、殼間平衡或塑性歷史。`R` 和 `ΔR` 使用 m，且須屬於同一參考狀態。

目前係數資料點為 100–815 °C。程式知道 `β(0 °C)=0`，但**不會**據此補出 0–100 °C 間的曲線；例如把 20 °C 設為 `T_ref` 會報錯，需先取得有依據的低溫資料。SS316 表格本身的來源與材料條件仍未驗證。

## 物理假設與限制

### Phase 3A 線彈性雙殼範例

在專案根目錄執行：

```powershell
uv run --locked python examples/phase3a_demo.py
```

這使用 `examples/synthetic_shell/` 的**人工合成材料**。結果會寫入本機 `outputs/phase3a_demo/`：`pressure_sweep.csv`、壓力–位移圖、內外殼應力與熱應變比較圖及 `manifest.json`。該資料夾不會提交 GitHub。

在 Python 或 VS Code Notebook 中單次求解：

```python
from feodosyev_shell.materials import load_material
from feodosyev_shell.solvers import LinearShellCase, solve_linear_shell

material = load_material("examples/synthetic_shell")
case = LinearShellCase(
    radius_m=0.1, inner_thickness_m=0.001, outer_thickness_m=0.002,
    inner_temperature_k=400, outer_temperature_k=300,
    stress_free_temperature_k=300, gas_pressure_pa=1e6,
)
result = solve_linear_shell(case, material)
print(result.radial_displacement_m, result.inner.hoop_stress_pa,
      result.outer.hoop_stress_pa, result.elastic_prediction_credible)
```

`result` 另含內外殼的總／熱／機械環向應變。輸入固定使用 m、K、Pa。內壓、拉應力及向外位移為正。`elastic_prediction_credible=False` 表示至少一殼超過資料表所選的降伏強度，**不代表已證實結構失效**；若沒有降伏資料，該欄為 `None`。SS316 的材料狀態與這個界限尚待驗證。[Phase 3A 公式與假設](docs/phase3a_plan.md)、[驗證紀錄](docs/phase3a_validation.md)

### Phase 3B 雙軸與合成塑性範例

3B-1 加入共同軸向應變、軸向力平衡及泊松效應；`BiaxialLinearCase` 另有 `axial_force_ratio`，理想閉端圓筒示例使用 `0.5`。3B-2 用**人工合成**雙線性 J2 材料逐步加壓及卸壓：

```powershell
uv run --locked python examples/phase3b_synthetic_cycle.py
```

本機 `outputs/phase3b_synthetic_cycle/` 會有壓力–位移循環圖、每一步的 CSV 與重現紀錄。卸載至零壓後的殘餘位移是**合成模型示範**，不是 SS316 實驗結果。塑性狀態要按步驟傳遞，不能用最終壓力和溫度重新推算。[公式、手算案例及限制](docs/phase3b_validation.md)

第一版插值先沿應變方向、再沿溫度方向做分段線性插值，僅在曲線共同有效範圍內計算。工程應力應變轉為真量時，須確認資料位於均勻變形區間。單調曲線沒有載入歷史，不能回答卸載、殘留應變、疲勞或壽命問題。[完整假設](docs/assumptions.md)

## 驗證與開發

`tests/` 使用人工可手算資料，檢查原始節點、溫度中點、單位、轉換和非法輸入。Phase 3A 另用獨立手算值核對雙殼環向平衡。測試通過代表程式符合這些案例；SS316 資料的物理有效性及真實引擎的結構行為仍需另外驗證。[階段路線](docs/roadmap.md)

`main` 保存已驗證成果。改動使用 `feature/...` 或 `fix/...` 短期分支，經 PR 與 CI 檢查後合併。規劃中的階段標籤為 `v0.1.0`、`v0.2.0`、`v0.3.0`，僅在對應階段完成驗收後建立。尚未選擇開源授權。
