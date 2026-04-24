# SimLens 系統研究架構圖

## 完整系統架構

```mermaid
flowchart TB
    %% ===== 離線訓練階段 =====
    subgraph Offline["🔧 離線訓練階段（一次性，完全自動化）"]
        direction TB
        
        subgraph Step1["步驟 1：影片特徵提取"]
            A1[(MicroLens-100K<br/>19,738 個影片<br/>100,000 觀眾<br/>719,405 交互記錄)]
            
            A1 --> T1[PySceneDetect<br/>場景檢測]
            A1 --> T2[OpenCV<br/>Farneback 光流]
            A1 --> T3[Whisper<br/>語音識別]
            A1 --> T4[CLIP ViT-B/32<br/>視覺特徵]
            
            T1 --> F1[ASD<br/>平均鏡頭時長]
            T1 --> F2[SCR<br/>鏡頭切換率]
            T2 --> F3[OFM<br/>光流強度]
            T3 --> F4[SD<br/>語音密度]
            T4 --> F5[CC<br/>內容複雜度]
            
            F1 --> STD[標準化<br/>映射到 1-10 分制<br/>基於全數據集統計分佈]
            F2 --> STD
            F3 --> STD
            F4 --> STD
            F5 --> STD
            
            STD --> FDB[(影片特徵數據庫<br/>19,738 個特徵向量)]
        end

        subgraph Step2["步驟 2：觀眾權重學習（100,000 個獨立線性回歸）"]
            A1 --> WL[對每個觀眾<br/>學習權重向量]
            FDB --> WL
            
            WL --> LR[Ridge Regression<br/>w* = argmin ∑（y - ŷ）² + λ‖w‖²<br/>λ = 0.01]
            
            LR --> W1[w_ASD]
            LR --> W2[w_SCR]
            LR --> W3[w_OFM]
            LR --> W4[w_SD]
            LR --> W5[w_CC]
            
            W1 --> WDB[(觀眾權重數據庫<br/>100,000 個權重向量)]
            W2 --> WDB
            W3 --> WDB
            W4 --> WDB
            W5 --> WDB
        end

        subgraph Step3["步驟 3：自動推斷觀眾特質（完全自動化）"]
            A1 --> TI[行為特質推斷]
            WDB --> TI
            
            TI --> BT1[Activity<br/>T = ∑ y_ui<br/>觀看影片總數]
            TI --> BT2[Conformity<br/>T = 1/N ∑ ｜r_ui - R_i｜²<br/>評分與平均差異]
            TI --> BT3[Diversity<br/>T = ｜∪ G_i｜<br/>觀看類型總數]
            TI --> BT4[年齡區間<br/>基於內容類型<br/>統計規則推斷]
            
            BT1 --> TDB[(觀眾特質數據庫<br/>100,000 個特質向量)]
            BT2 --> TDB
            BT3 --> TDB
            BT4 --> TDB
        end

        subgraph Step4["步驟 4：訓練映射模型（特質 → 權重）"]
            TDB --> MM[XGBoost 回歸模型<br/>輸入：特質向量<br/>輸出：權重向量]
            WDB --> MM
            
            MM --> EVAL[模型評估<br/>Weight MAE < 0.15<br/>Weight Correlation > 0.5]
            
            MM --> MODEL[(訓練好的映射模型<br/>mapping_model.pkl)]
        end

        subgraph Step5["步驟 5：建立 Few-shot 範例庫（RAG）"]
            PC[(PersonaChat<br/>8,000+ persona 描述)] --> EMB[Sentence Transformer<br/>生成 Embedding]
            B5[(Big5-Chat<br/>100,000 對話)] --> EMB
            
            EMB --> FAISS[(FAISS 語義索引<br/>persona_index.faiss)]
        end
    end

    %% ===== 線上使用階段 =====
    subgraph Online["🚀 線上使用階段（實時）"]
        direction TB
        
        subgraph OStep1["步驟 1-2：影片上傳與特徵提取（1-2 分鐘）"]
            UP[創作者上傳影片] --> ASYNC[異步特徵提取<br/>Celery Worker]
            
            ASYNC --> OT1[PySceneDetect → ASD, SCR]
            ASYNC --> OT2[OpenCV → OFM]
            ASYNC --> OT3[Whisper → SD]
            ASYNC --> OT4[CLIP → CC]
            
            OT1 --> OSTD[標準化到 1-10 分制]
            OT2 --> OSTD
            OT3 --> OSTD
            OT4 --> OSTD
            
            OSTD --> CACHE[(Redis 緩存<br/>影片特徵向量)]
        end

        subgraph OStep3["步驟 3：創建代理人觀眾（3-5 秒）"]
            INPUT[創作者輸入<br/>「25歲學生，喜歡快節奏」] --> SEARCH[3.1 FAISS 語義搜索<br/>檢索 3-5 個相似範例<br/>< 100ms]
            
            FAISS -.-> SEARCH
            
            SEARCH --> LLM[3.2 LLM 提取特質<br/>GPT-4 / Qwen<br/>參考範例 + 創作者描述<br/>2-3 秒]
            
            LLM --> TRAITS[特質向量<br/>age, activity,<br/>conformity, diversity, ...]
            
            TRAITS --> MAP[3.3 映射模型預測權重<br/>mapping_model.predict<br/>< 100ms]
            
            MODEL -.-> MAP
            
            MAP --> WEIGHTS[權重向量<br/>w_ASD, w_SCR,<br/>w_OFM, w_SD, w_CC, b]
        end

        subgraph OStep4["步驟 4：留存率計算（< 10ms）"]
            CACHE --> CALC[Retention = ∑ Feature_i × Weight_i + b]
            WEIGHTS --> CALC
            
            CALC --> SIG[Sigmoid 歸一化<br/>映射到 0-1]
            
            SIG --> SCORE[留存率預測<br/>例：0.82（82%）]
        end

        subgraph OStep5["步驟 5：可解釋分析報告（< 150ms）"]
            SCORE --> EX1[第一層：線性分解<br/>每個特徵的貢獻<br/>SCR: +8.1, OFM: +4.9, ...]
            
            SCORE --> EX2[第二層：SHAP 解釋<br/>每個特質對權重的貢獻<br/>外向性→w_SCR +0.35]
            
            EX1 --> REPORT[完整分析報告<br/>特徵貢獻排序<br/>權重來源解釋<br/>優化建議]
            EX2 --> REPORT
        end

        subgraph OStep6["步驟 6：反事實與時序分析（可選）"]
            SCORE --> CF[反事實分析<br/>如何修改影片<br/>達到目標留存率]
            
            CACHE --> TS[時序分析<br/>分段計算留存率<br/>找出問題片段]
            WEIGHTS --> TS
            
            CF --> ADV[優化建議<br/>「SCR 從 5 提升到 7<br/>預期提升 +15%」]
            
            TS --> CURVE[留存率曲線<br/>識別高潮與低谷<br/>精準定位問題時間段]
        end
    end

    %% ===== 實驗驗證 =====
    subgraph Experiments["📊 實驗驗證"]
        direction LR
        
        subgraph CoreExp["核心系統驗證"]
            EXP1[實驗 1：特徵有效性<br/>QVHighlights t-test<br/>MicroLens 相關性]
            EXP2[實驗 2：權重學習<br/>MAE < 0.2<br/>Pearson r > 0.4]
            EXP3[實驗 3：消融研究<br/>移除特徵影響分析]
            EXP4[實驗 4：基線比較<br/>vs CF / VRAgent-R1]
        end
        
        subgraph PersonaExp["Persona 框架驗證"]
            EXP5[實驗 5：特質提取<br/>MAE < 10<br/>Pearson r > 0.6]
            EXP6[實驗 6：特質映射<br/>Weight MAE < 0.15]
            EXP7[實驗 7：代理人一致性<br/>Taste Accuracy > 65%]
            EXP8[實驗 8：真實世界差距<br/>A/B Agreement > 80%]
        end
    end

    %% 離線到線上的連接
    FDB -.->|特徵標準化參數| OSTD
    MODEL -.->|映射模型| MAP
    FAISS -.->|語義索引| SEARCH

    %% 實驗連接
    FDB -.-> EXP1
    WDB -.-> EXP2
    WDB -.-> EXP3
    SCORE -.-> EXP4
    LLM -.-> EXP5
    MODEL -.-> EXP6
    MODEL -.-> EXP7
    MODEL -.-> EXP8

    %% 樣式
    style Offline fill:#f0f8ff,stroke:#4a90d9,stroke-width:2px
    style Online fill:#f0fff0,stroke:#4a9d4a,stroke-width:2px
    style Experiments fill:#fff8f0,stroke:#d9904a,stroke-width:2px
    style Step1 fill:#e8f4fd
    style Step2 fill:#e8fde8
    style Step3 fill:#fde8f4
    style Step4 fill:#f4fde8
    style Step5 fill:#f4e8fd
    style OStep1 fill:#e8f4fd
    style OStep3 fill:#fde8f4
    style OStep4 fill:#f4fde8
    style OStep5 fill:#e8fde8
    style OStep6 fill:#f4e8fd
    style CoreExp fill:#fde8e8
    style PersonaExp fill:#e8e8fd
```


## 系統流程說明

### 離線訓練階段（一次性，2-3 天，完全自動化）

| 步驟 | 輸入 | 處理 | 輸出 | 時間 |
|------|------|------|------|------|
| 步驟 1 | 19,738 個影片 | PySceneDetect, OpenCV, Whisper, CLIP | 影片特徵數據庫 | 1-2 天 |
| 步驟 2 | 100,000 觀眾觀看歷史 | Ridge Regression（每人一個） | 觀眾權重數據庫 | 2-3 小時 |
| 步驟 3 | 觀看歷史 + 權重向量 | 統計推斷（Activity, Conformity, Diversity） | 觀眾特質數據庫 | 2-3 小時 |
| 步驟 4 | 特質向量 + 權重向量 | XGBoost 回歸 | 映射模型 | 1-2 小時 |
| 步驟 5 | PersonaChat + Big5-Chat | Sentence Transformer + FAISS | Few-shot 範例庫 | 2-3 小時 |

### 線上使用階段（實時）

| 步驟 | 輸入 | 處理 | 輸出 | 時間 |
|------|------|------|------|------|
| 步驟 1-2 | 創作者上傳影片 | 異步特徵提取 | 影片特徵向量 | 1-2 分鐘 |
| 步驟 3.1 | 創作者描述 | FAISS 語義搜索 | 相似 Persona 範例 | < 100ms |
| 步驟 3.2 | 描述 + 範例 | LLM 提取特質 | 特質向量 | 2-3 秒 |
| 步驟 3.3 | 特質向量 | 映射模型預測 | 權重向量 | < 100ms |
| 步驟 4 | 特徵 × 權重 | 線性計算 + Sigmoid | 留存率預測 | < 10ms |
| 步驟 5 | 預測結果 | 線性分解 + SHAP | 可解釋分析報告 | < 150ms |
| 步驟 6 | 預測結果 | 反事實推理 + 時序分析 | 優化建議 + 留存率曲線 | 2-3 秒 |

### 數據集用途

| 數據集 | 角色 | 用途 | 階段 |
|--------|------|------|------|
| MicroLens-100K | 訓練數據 | 特徵提取、權重學習、特質推斷、映射模型訓練 | 離線 |
| PersonaChat | 教科書 | Few-shot 範例庫，教 LLM 提取特質 | 離線建立 + 線上使用 |
| Big5-Chat | 教科書 | Few-shot 範例庫，補充人格特質範例 | 離線建立 + 線上使用 |
| QVHighlights | 驗證數據 | 驗證特徵有效性（實驗 1） | 實驗 |
| TVSum | 驗證數據 | 驗證特徵有效性（實驗 1） | 實驗 |

### 模型清單

| 模型 | 類型 | 用途 | 位置 |
|------|------|------|------|
| Ridge Regression × 100,000 | 線性回歸 | 學習每個觀眾的權重 | 離線步驟 2 |
| XGBoost | 樹模型 | 特質 → 權重映射 | 離線步驟 4 + 線上步驟 3.3 |
| Sentence Transformer | 語言模型 | 生成 Persona Embedding | 離線步驟 5 + 線上步驟 3.1 |
| GPT-4 / Qwen | LLM | 從文字描述提取特質 | 線上步驟 3.2 |
| SHAP TreeExplainer | 可解釋性 | 解釋映射模型的權重來源 | 線上步驟 5 |

### 可解釋性架構

```
第一層（內建）：線性分解
  Retention = w₁×ASD + w₂×SCR + w₃×OFM + w₄×SD + w₅×CC + b
  → 直接分解每個特徵的貢獻
  → 時間: < 10ms

第二層（SHAP）：權重來源解釋
  SHAP TreeExplainer 分析映射模型
  → 解釋每個特質對權重的貢獻
  → 例：外向性(70分) → w_SCR +0.35
  → 時間: < 100ms

第三層（反事實）：優化建議
  反向計算特徵變化
  → 告訴創作者如何修改影片達到目標留存率
  → 例：SCR 從 5 提升到 7 → 預期提升 +15%
  → 時間: < 50ms
```

### 核心公式

```
特徵標準化:
  ASD_score = 10 - min(9, (ASD - 1) / 1.0)
  SCR_score = min(10, (SCR - 2) / 2.8)
  OFM_score = min(10, OFM / 10)
  SD_score = SD × 10
  CC_score = min(10, CC × 20)

權重學習:
  w_u = argmin [∑(retention_actual - w·features)² + λ‖w‖²]

留存率預測:
  Retention_Score = sigmoid(∑ Feature_i × Weight_i + b)

行為特質計算:
  Activity = ∑ y_ui (觀看影片總數)
  Conformity = (1/N) ∑ |r_ui - R_i|²
  Diversity = |∪ G_i| (觀看類型總數)
```
