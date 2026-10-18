# Feodosyev §9.3 本機 HTML 介面

在專案根目錄執行 `python "Feodosyev html/app.py"`，瀏覽器開啟
`http://127.0.0.1:8766`；已有 `.venv` 時可改用
`.venv\Scripts\python.exe "Feodosyev html/app.py"`。
需要 README 列出的本機 Granta SS316 材料檔。這些資料不隨 repository 發布。
使用 `--port 另一個埠號` 可避開埠號衝突。

## 計算

輸入半徑、兩殼厚度、兩殼溫度及無因次軸向載荷係數 k。
初次開啟時，除了 File Path 以外的數字與檔名欄位全部空白。
模型參數（包括 k）必須自行輸入，或由 Load JSON 一次填入。
下方 JSON 範例為 R=100 mm、Inner=1 mm／538°C、Outer=2 mm／20°C、k=0.5。
k 僅要求有限數值，不額外指定任意上下限；若某個 k 無法通過既有求解器的
材料範圍、非負壓力、軸向根或平衡檢查，Generate 會顯示原始原因。
不進行推力換算、升溫歷史積分或塑性初始狀態載入。

- Auto ON：重用 `feodosyev_temperature_study.py` 的零壓力起點搜尋。
  在兩殼自由熱應變之間檢查唯一的負到正壓力交點，使用數值包圍區間的非負側，
  不截掉負壓力、不將數值壓力硬設為零。若搜尋失敗會停止並顯示錯誤。
- Auto OFF：顯示 Start Strain (%)、End Strain (%) 及 Sample Count。
  起終點均為「共同總環向應變」百分比，例如 3 表示 0.03。
  取樣點均勻分布，包含兩端，數量限 2–5000。
- Auto 下掃描欄位若空白，內部使用終點 3%、481 點，不將預設值填回輸入框。
  Auto 只決定起點；已有手動／匯入設定時，使用設定中的終點與點數。
  Manual 下三個掃描欄位均須填寫。
- `feodosyev_curve.py` 呼叫 `feodosyev_single_point.py` 做逐點計算與連續分支選擇，
  沒有在網頁或後端重新實作方程、插值及本構。分支不明、材料範圍超出、
  負壓力或平衡失敗時，不輸出不完整的成功曲線。

掃描終點只代表這次指定的計算範圍，**不代表承載極限**。
主圖只顯示壓力－總環向應變，不分類 A–D 或判定失效。
設定變更後需要重新按 Generate；舊結果不再可從畫面匯出。
貓咪使用原動畫中的完整旋轉片段，移除開頭站立等待，保留原片段像素及播放速度。
一圈為 1.93 秒；計算或儲存完成後，等當圈播完才顯示成功／失敗圖片。
因此快速案例的畫面等待時間約為一圈，較慢案例會繼續旋轉。

## Import Data 與 Save JSON

Load JSON 先驗證整份文件，成功才一次更新所有模型與掃描欄位。
匯入錯誤時，現有設定不變，畫面顯示錯誤圖片與原因。
匯入後可直接手動改值，再按 Generate。

格式如下（`summary` 可省略；Save JSON 會附上數值求解摘要）：

```json
{
  "format": "feodosyev_section_9_3_settings",
  "units": {
    "length": "mm",
    "temperature": "degC",
    "strain": "%",
    "k": "dimensionless"
  },
  "settings": {
    "radius_mm": 100,
    "inner_thickness_mm": 1,
    "outer_thickness_mm": 2,
    "inner_temperature_c": 538,
    "outer_temperature_c": 20,
    "k": 0.5,
    "auto_strain_range": true,
    "start_strain_percent": 0.025,
    "max_strain_percent": 3,
    "sample_count": 481
  }
}
```

全部 `settings` 欄位都必須存在，數字不可用字串或布林值代替。
單位必須完全相符，不進行猜測或自動換算；設定檔限 4 MB。
Auto 下仍保存手動起點，但計算時會重新搜尋實際起點，並寫入摘要。
JSON 保存的是輸入條件，`summary` 不會被當成求解輸入；不接受舊的
`double_shell_initial_state_v1`。重現計算還需要使用相同的程式及本機材料資料。

## 匯出

先按 Generate，成功後在 File Path 填入已存在的資料夾，填寫各格式的檔名後按 Save。
預設資料夾在 Python 啟動時由 `Path.cwd()` 取得，沒有固定儲存某台電腦的路徑。
這裡的「啟動資料夾」是執行啟動指令時所在的工作資料夾，不一定是 `app.py` 所在處。
沿用 1D 的檔名與路徑操作，拒絕覆寫既有檔案。

- PNG：與原 1D 相同的簡潔主圖，無額外標題或圖例。
- CSV：使用既有 `write_curve_csv`，包含每點的壓力、共同軸向／環向應變、
  Inner／Outer 軸向及環向應力，以及原求解器的平衡與分支欄位。
  欄位名稱標示 Pa／MPa，無後綴的應變為無因次，另有環向應變百分比欄。
- JSON：保存「產生這次成功結果時」的設定及摘要，可直接由 Load JSON 再次載入。

生成檔案只留在本機，不提交至 Git。

## 驗證

```powershell
python -m unittest discover -s tests -v
node --check "Feodosyev html/app.js"
node tests/test_feodosyev_html_animation.js
```

`tests/test_feodosyev_html.py` 比對階段 2 的 17 點完整結果、預設 Auto 的 481 點、
k 改值、JSON 驗證與完整曲線重現、PNG／CSV／JSON 實際寫檔、HTTP 與貓咪資源。
材料資料不存在時，相關測試明確 skipped。
原有 1D 及其他階段測試仍分開保留；瀏覽器實測需另外確認顯示和操作。

計算加速方式與新舊結果對照見 [效能驗證](feodosyev_performance.md)。
