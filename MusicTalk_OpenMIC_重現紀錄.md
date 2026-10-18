# MusicTalk OpenMIC-2018 重現紀錄

更新日期：2026-10-15

## 對應論文實驗

本次對應 Lin、Cheng、Chiu 的 MusicTalk 論文第 VI 節 OpenMIC-2018 benchmark。作者在該段把評估結果指向 Table 1；但論文另一處又說 Table 1 是 Section II 的相關工作摘要，且表中 MusicTalk 的資料列列出 URMP、MedleyDB、INSTDB，和 OpenMIC 段落的指向互相矛盾。因此本紀錄以「第 VI 節 OpenMIC benchmark」作為實驗名稱，不把 Table 1 的數字直接當成已確認同一 protocol 的比較目標。這不是 Table 4 的 MedleyDB/URMP/INSTDB 強標籤實驗，也不是 Table 5 的《彼得與狼》案例。

論文指定的資料安排和訓練規則包括：官方 split01 training 中取 15% 作 validation（12,677 train、2,238 validation），遮蔽缺失標籤，以 inverse-frequency weighted BCE 處理類別不平衡，validation loss 連續 10 epochs 未改善時早停，保存最低 validation loss 的 checkpoint。評估使用 macro ROC-AUC、mAP，以及以 validation ROC 曲線最接近 (0,1) 的 per-class threshold 計算 macro F1。

## 本次實驗

- 資料：OpenMIC-2018 全量 NPZ 與原始 OGG；archive MD5 `e4ccf187e2bb5ab2e115416e8aafe7f4`。
- Split：split01 的 12,677 train、2,238 validation、5,085 official test；20 類別。
- 模型：PaSST/ViT + BCP-WP，搭配 PANNs ResNet38；兩個 AudioSet pretrained backbone 都參與微調。
- 訓練：seed 17、batch size 4、AdamW；backbone LR 1e-5、head LR 3e-4、weight decay 1e-4；最多 20 epochs、patience 10。
- 實際停止：從頭連續跑 15 epochs，因最佳點 epoch 5 後連續 10 epochs 沒有更低 validation loss 而早停。
- 環境：Ubuntu 22.04 WSL、Python 3.12.15、PyTorch 2.10.0+cu128、RTX 3090。

### 結果

| 資料分割／checkpoint | macro ROC-AUC | mAP | macro F1 |
|---|---:|---:|---:|
| Validation，最佳 epoch 5 | 0.92466 | 0.86883 | 0.82734 |
| Official test split01，最佳 epoch 5 | 0.91494 | 0.84970 | 0.79959 |

Test F1 使用 validation 選出的每類 threshold，沒有用 test 標籤調 threshold。官方 test 之前已在本專案另一個開發 checkpoint 上評估過；因此本次結果不能稱為完全未接觸的盲測，這個 exposure 已保留在 test metrics 中。

## 重現程度與限制

這是 **單 seed、論文方法導向的 OpenMIC 主實驗重實作**，具備完整資料讀取、核心分類器訓練、masked/weighted loss、validation 選模、早停和 test 評估記錄；不等同作者原始碼的逐位元重現，也不代表整篇論文所有方法和實驗都已重做。

目前沒有找到 MusicTalk 專屬的作者程式碼。實作使用本地 pinned 的 PaSST/PANNs 上游程式碼和權重，並自行整合 BCP-WP 與分支融合。論文沒有完整指定 seed、optimizer、learning rate、weight decay、batch size、輸入時長、checkpoint 版本和 validation sample IDs；本次這些設定需視為重實作選擇。另外，本地融合層是將兩組 20 類別分支機率送入 40-to-20 線性層；需再對照論文描述，不能聲稱融合公式與作者實作逐項相同。執行環境也不同於論文年代的工具鏈。

尚未重現的範圍：

- Table 4：MedleyDB、URMP、INSTDB 強標籤訓練／測試。
- Table 5：《彼得與狼》小提琴／大提琴案例。
- MusicTalk 完整 microservice、Audio Shaper、服務整合與可視化系統。
- 多 seed 重複實驗與論文全部 baseline 的統一重跑。

## 可檢查檔案

本次訓練目錄：`runs/paper_aligned_openmic_pat10_cap20_seed17_20261015/`

包含 `config.json`、`split_ids.json`、`history.json`、`metrics.json` 和 `checkpoint_best.pt`。最佳 checkpoint SHA-256：`26ad86b8908fe547a30ce5e853854605bccf4af89c182d1ea7f4e941042e2ef4`。

測試目錄：`runs/official_test_pat10_cap20_seed17_20261015/`

包含 `metrics.json`、`metrics.csv` 和 `test_predictions.npz`。

### 重跑指令（Ubuntu WSL）

```bash
cd /home/user/TATQAQOAO/0_ntsc_project
./MusicTalk/run_musictalk_wsl.sh ./MusicTalk/train_paper_aligned.py \
  --output-dir ./MusicTalk/runs/paper_aligned_openmic_pat10_cap20_seed17_20261015 \
  --seed 17 --batch-size 4 --epochs 20 --patience 10
```

本次結果目錄已存在，腳本會拒絕覆寫；另跑時請改用新的 output-dir，並在比較時保留 seed 與設定。

## 2026-10-17：從最佳 checkpoint 分支訓練至 epoch 30

這次不是把 10/15 的訓練程序無縫接續。以該 run 的 epoch 5 最佳權重作為初始化，沿用同一組 train/validation IDs 與 seed 17，再訓練 epochs 6–30，共新增 25 個 epoch。原 checkpoint 沒有 optimizer state 或 RNG state，因此建立新的 AdamW optimizer 並重新設定 RNG；這是 best-checkpoint fine-tuning branch，不能視為精確 resume。

| 項目 | 結果 |
|---|---|
| 訓練終點 | epoch 30 cap；本階段執行 25 epochs（6–30） |
| 最佳 checkpoint | epoch 5（來源權重；本階段沒有改善 validation loss） |
| 最佳 validation weighted masked BCE | 0.347015 |
| Validation macro ROC-AUC | 0.923003 |
| Validation mAP | 0.867215 |
| Validation ROC-nearest macro-F1 | 0.830072 |
| Official test | 未評估 |

結果是單 seed 的 validation 開發紀錄，且 F1 與 mAP/AUC 的評估指標只描述本次分支的 validation。Official test 沒有在這個分支上重新評估；不要把本結果當作新的 test score 或與不同設定的 run 混為一談。

設定、逐 epoch history、metrics、split IDs 與完整訓練 log 位於 `runs/paper_aligned_openmic_ckpt5_resume_from005_cap30_seed17_20261017/` 及 `retrain_ckpt5_resume_cap30_seed17_20261017.log`。權重檔保留在本機但不納入 GitHub 推送，以避免大型 checkpoint 超出一般 Git 檔案限制。

