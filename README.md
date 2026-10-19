# Feodosyev §9.3 HTML

本專案提供 Feodosyev §9.3 雙薄殼的本機操作介面，可計算壓力－總環向應變曲線，並匯出 PNG、CSV 與可再次匯入的設定 JSON。

## 保留的程式

- `Feodosyev html/`：網頁、Python 後端與操作狀態圖片。
- `feodosyev_single_point.py`：Feodosyev 單點求解器。
- `feodosyev_curve.py`：連續掃描與 CSV 輸出。
- `feodosyev_origin.py`：Auto 模式使用的零壓力起點搜尋。
- `interpolation.py`：讀取材料資料並進行插值。
- `pyproject.toml`：Python 版本與套件需求。

## 啟動

需要 Python 3.12 以上。在專案資料夾開啟 PowerShell。

已有本機 `.venv` 時，直接執行：

```powershell
.\.venv\Scripts\python.exe "Feodosyev html/app.py"
```

新環境先建立環境並安裝套件：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe "Feodosyev html/app.py"
```

用瀏覽器開啟 <http://127.0.0.1:8766>。停止時，在啟動程式的終端機按 Ctrl+C。
若埠號已被使用，可加上 `--port 8767`，再開啟對應網址。
請透過 Python 後端啟動；只開啟 `index.html` 無法進行計算。

## 本機材料資料

`interpolation.py` 需要下列四個檔案，放在專案內的 `database/Granta_SS316_annealed_csv/`：

```text
Elasticity.csv
Coefficient of Thermal Expansion.csv
Zero-Thermal-Strain Reference Temperature.csv
Isotropic Hardening.csv
```

Granta 原始資料受來源與授權限制，不隨 GitHub 專案提供。請自行準備有權使用的資料；缺少資料時無法啟動計算。本機的材料原始檔不因程式整理而刪除。

## 操作

1. 輸入半徑、兩殼厚度、兩殼溫度與無因次 Axial Load Factor（k），或用 Load JSON 載入設定。
2. Auto Strain Range 開啟時，程式搜尋非負壓力起點；掃描設定留白時使用終點 3%、481 點。已匯入或手動填入的終點與點數會沿用。
3. 關閉 Auto 後，必須填寫 Start Strain (%)、End Strain (%) 與 Sample Count（2–5000，包含兩端）。例如 3 表示總環向應變 3%。
4. 按 Generate。成功後，在 EXPORT 填入檔名，再按對應的 Save 按鈕。File Path 必須是已存在的資料夾，程式不覆寫同名檔案。

k 必須是有限數值；材料範圍、非負壓力、求解分支與平衡檢查無法通過時，畫面會顯示原因。掃描終點不代表承載極限。

- PNG：保存主圖。
- CSV：保存各計算點的壓力、共同軸向／環向應變、兩殼軸向／環向應力及求解欄位。應變沒有單位，壓力與應力的 Pa／MPa 依欄位名稱標示。
- JSON：保存本次的參數設定，可用 Load JSON 再次載入。

設定 JSON 的 `format` 為 `feodosyev_section_9_3_settings`；`units` 必須是 `length=mm`、`temperature=degC`、`strain=%`、`k=dimensionless`。`settings` 必須包含半徑、兩殼厚度、兩殼溫度、k、Auto 開關、掃描起終點與取樣數。建議先用 Save JSON 產生完整設定，再匯入重用。
