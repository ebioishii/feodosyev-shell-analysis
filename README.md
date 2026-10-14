# Feodosyev Shell Analysis

用 Python 建立可追溯的再生冷卻結構分析工具。目前完成的範圍是材料資料管理與**單調載入**應力–應變曲線。雙層薄殼求解器是下一階段工作；目前尚無可用的 `p_g–ΔR` 結果。

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

匯入會處理彈性模數、降伏強度、泊松比，以及以 0 °C 為參考的**平均熱膨脹係數**。新加入的 `SS316_plastic_StressStrain.csv` 目前只有品質摘要與雜湊紀錄，沒有被轉成正式曲線：圖像擷取資料仍有重複應變與缺少初始段。來源、材料狀態、使用權及曲線整理方式仍待核對。`data/raw/`、`data/processed/` 與 `outputs/` 不提交 GitHub。

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

第一版插值先沿應變方向、再沿溫度方向做分段線性插值，僅在曲線共同有效範圍內計算。工程應力應變轉為真量時，須確認資料位於均勻變形區間。單調曲線沒有載入歷史，不能回答卸載、殘留應變、疲勞或壽命問題。[完整假設](docs/assumptions.md)

## 驗證與開發

`tests/` 使用人工可手算資料，檢查原始節點、溫度中點、單位、轉換和非法輸入。測試通過代表程式行為符合這些案例；SS316 資料的物理有效性與 Feodosyev 文獻方程仍需另外驗證。[階段路線](docs/roadmap.md)

`main` 保存已驗證成果。改動使用 `feature/...` 或 `fix/...` 短期分支，經 PR 與 CI 檢查後合併。規劃中的階段標籤為 `v0.1.0`、`v0.2.0`、`v0.3.0`，僅在對應階段完成驗收後建立。尚未選擇開源授權。
