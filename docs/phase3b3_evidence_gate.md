# Phase 3B-3：SS316 真塑性資料的進場檢查

## 目前查到的資料

2026-10-15 檢查本機 `D:/LAB/regen/references/SS316_data/`：`SS316_plastic_StressStrain.csv` 仍是 Lv et al. (2022) Fig. 2(a) 的暫用圖像數位化資料，SHA-256 為 `147ea6fec0e848a4875ea58c11e74159b143b2a5f7d11154daa07b4067f91e9d`。另有 `x_P...csv`，欄位是位置 `x_mm` 與壓力 `P_MPa`，不是缺少的拉伸應力–應變原始資料。沒有發現可辨識為作者新增拉伸數據的檔案。

既有 [Gate B/C1 紀錄](gate_b_c1_plan.md)指出六個溫度的正式均勻變形截止點均未驗證；350°C 有兩段、450°C 有七段可辨識曲線，所有溫度都缺少零應變附近的初始段。峰值應變只能作暫用參考，不能自動視為頸縮起點。

## 如何執行

在專案根目錄的 PowerShell 執行：

```powershell
uv run --locked python scripts/audit_phase3b3_evidence.py
```

工具讀取本機 `data/processed/ss316_provisional_v1/`，在 Git 忽略的 `outputs/phase3b3_evidence/` 產生：

- `evidence_template.json`：按材料曲線的溫度列出待填證據欄位；初始值全部為 `null`。
- `readiness_report.json`：每個溫度的資料範圍、缺口、阻擋原因，以及輸入檔與程式版本的雜湊。

目前六個溫度都是 `blocked`，各有 10 個尚缺項目。最關鍵的是來源仍為暫用數位化、材料狀態未知、初始應變段缺失、沒有獲核准的均勻變形截止點，以及缺乏可在同一材料資料集中查詢的相容 E(T)。`readiness_report.json` 將這些項目逐溫度列出，不能把阻擋數量當成資料品質分數。

收到作者的新資料後，應先建立**另一份**有來源的材料資料集，保留原始檔不修改，再將每個溫度的證據填入模板的副本。填寫時需記錄：均勻變形界限的數值、定義與來源；初始段來源；試驗條件；E(T) 與試片的相容依據；以及屈服或 0.2% proof stress 的定義。工程應變以 `1` 為單位，例如 20% 填 `0.20`；溫度以 K 識別。若資料未給出某欄，保留 `null`，不要推定或補造。

可用下列指令讀取已填的證據副本；模板本身在重跑時不會被覆寫：

```powershell
uv run --locked python scripts/audit_phase3b3_evidence.py --material data/processed/NEW_DATASET --evidence outputs/phase3b3_evidence/my_evidence.json --output outputs/phase3b3_new_review
```

工具會拒絕資料集版本、來源雜湊或溫度不符的證據表，並檢查核准界限之前是否有資料缺口。即使將來每個軟體檢查都通過，狀態仍是 `manual_review_required`：研究者必須再核對論文、作者數值、試驗條件及不確定性，才能決定是否進入真應力／真塑性應變轉換。此工具不會自動產生 SS316 塑性本構，也不判定結構安全。

本次只新增進場檢查程式與使用人工資料的自動測試；原始 CSV、處理資料和報告均留在本機。
