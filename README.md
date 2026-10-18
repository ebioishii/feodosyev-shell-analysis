# Feodosyev Shell Analysis

本專案使用同一個 GitHub repository，逐步研究 Feodosyev Chapter 9 §9.3 的雙薄殼方法。

## 目前內容

- `heat_then_pressurize.py`：既有的一維雙殼升溫後加壓模型。
- `known_initial_pressurize.py`：由已知 JSON 初始狀態開始的一維加壓模型。
- `interpolation.py`：讀取本機 Granta SS316 資料並做溫度／硬化曲線插值。
- `1D html/`：本機 HTML 介面，支援 Heating ON/OFF、Load/Save JSON、Auto/Manual Strain Range，以及 PNG/CSV 匯出。
- `tests/test_existing_1d.py`：既有一維模型的回歸測試。
- `feodosyev_single_point.py`：獨立的 Feodosyev §9.3 式 (9.1)–(9.6) 單點求解器。
- `tests/test_feodosyev_single_point.py`：解析簡化案例、Granta 應變轉換及平衡殘差測試。
- `feodosyev_curve.py`：在指定環向總應變範圍內，以連續分支預測建立完整壓力－應變曲線。
- `tests/test_feodosyev_curve.py`：曲線延續、逐點檢查，以及 CSV／PNG 輸出測試。
- `feodosyev_diagnostics.py`：環向應力符號、溫度對應降伏參考與局部壓力斜率診斷。
- `tests/test_feodosyev_diagnostics.py`：解析切線、非線性應力微擾與獨立平衡重解驗證。
- `feodosyev_temperature_study.py`：階段 4A 的 Inner 固定工作溫度參數研究，含零壓力起點求解及三張比較圖。
- `tests/test_feodosyev_temperature_study.py`：解析起點/低點、七種溫度、538 C 基準及明確失敗回報測試。
- `feodosyev_thickness_study.py`：階段 4B-1 固定總壁厚的厚度比研究，含有號膜力與同壓力承載比較。
- `tests/test_feodosyev_thickness_study.py`：厚度換算、膜力解析檢查、六組案例、η=2 基準及共同壓力求根驗證。

物理條件的對照、基準結果及限制見 [docs/feodosyev_physical_diagnostics.md](docs/feodosyev_physical_diagnostics.md)。本例內殼環向應力雖由壓縮轉為拉伸，其材料圖狀態始終在塑性分支；不以曲線外觀直接認定完整四階段機制。

Inner 溫度敏感度研究的計算方式與輸出格式見 [docs/feodosyev_temperature_study.md](docs/feodosyev_temperature_study.md)。

固定總壁厚的厚度比分析見 [docs/feodosyev_thickness_study.md](docs/feodosyev_thickness_study.md)。

既有一維模型只考慮共同環向應變與環向平衡，**不是**完整的 Feodosyev §9.3 雙軸模型。兩者保持為分開的程式模組；單點求解器的方程、材料應變轉換及限制見 [docs/feodosyev_single_point.md](docs/feodosyev_single_point.md)，階段 2 曲線延續與輸出規則見 [docs/feodosyev_curve.md](docs/feodosyev_curve.md)。

## 本機材料資料

Granta 原始材料資料受來源與授權限制，不納入 Git。執行既有一維模型前，本機需有：

```text
database/Granta_SS316_annealed_csv/Elasticity.csv
database/Granta_SS316_annealed_csv/Coefficient of Thermal Expansion.csv
database/Granta_SS316_annealed_csv/Zero-Thermal-Strain Reference Temperature.csv
database/Granta_SS316_annealed_csv/Isotropic Hardening.csv
```

缺少這些檔案時，材料相關測試會清楚顯示為 skipped，而不是假裝已驗證。

## 執行既有 HTML 介面

在專案根目錄建立環境並安裝套件後：

```powershell
python -m pip install -e .
python "1D html/app.py"
```

再用瀏覽器開啟 `http://127.0.0.1:8765`。

## 執行測試

```powershell
python -m unittest discover -s tests -v
```

研究範圍與模型邊界見 [docs/research_scope.md](docs/research_scope.md)。原始文獻、Granta 資料、ANSYS 專案、Notebook、設計過程截圖與計算輸出都只保留於本機，不納入 Git。
