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

既有一維模型只考慮共同環向應變與環向平衡，**不是**完整的 Feodosyev §9.3 雙軸模型。兩者保持為分開的程式模組；單點求解器的方程、材料應變轉換及限制見 [docs/feodosyev_single_point.md](docs/feodosyev_single_point.md)。

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
