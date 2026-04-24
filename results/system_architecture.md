# SimLens 系統架構圖（以線上使用流程為主）

## 線上流程系統架構

```mermaid
flowchart TB
    %% ===== 創作者入口 =====
    subgraph Entry["👤 創作者入口"]
        U1[創作者上傳影片]
        U2[創作者輸入目標觀眾描述<br/>「25歲學生，喜歡快節奏」]
    end

    %% ===== 第一層：影片特徵提取服務 =====
    subgraph Layer1["第一層：影片特徵提取服務（1-2 分鐘）"]
        direction TB
        
        subgraph VideoProcessing["影片處理"]
            VP1[影片接收<br/>FastAPI Endpoint]
            VP2[影片轉碼與切片<br/>FFmpeg]
            VP3[異步任務分發<br/>Celery + RabbitMQ]
        end

        subgraph FeatureWorkers["特徵提取 Workers（並行）"]
            FW1[Worker 1<br/>PySceneDetect<br/>→ ASD, SCR]
            FW2[Worker 2<br/>OpenCV Farneback<br/>→ OFM]
            FW3[Worker 3<br/>Whisper base<br/>→ SD]
            FW4[Worker 4<br/>CLIP ViT-B/32<br/>→ CC]
        end

        subgraph Normalize["特徵標準化"]
            NORM[標準化引擎<br/>映射到 1-10 分制<br/>使用離線統計參數]
        end

        VP1 --> VP2 --> VP3
        VP3 --> FW1
        VP3 --> FW2
        VP3 --> FW3
        VP3 --> FW4
        FW1 --> NORM
        FW2 --> NORM
        FW3 --> NORM
        FW4 --> NORM
    end

    %% ===== 特徵緩存 =====
    NORM --> CACHE[(Redis<br/>影片特徵緩存<br/>TTL: 24h)]

    %% ===== 第二層：觀眾權重建模服務 =====
    subgraph Layer2["第二層：觀眾權重建模服務（3-5 秒）"]
        direction TB

        subgraph ColdStart["冷啟動路徑（新觀眾）"]
            CS1[FAISS 語義搜索<br/>檢索 3-5 個相似 Persona<br/>< 100ms]
            CS2[LLM 特質提取<br/>GPT-4 / Qwen<br/>參考範例 + 描述<br/>2-3 秒]
            CS3[XGBoost 映射模型<br/>特質向量 → 權重向量<br/>< 100ms]
            
            CS1 --> CS2 --> CS3
        end

        subgraph WarmPath["暖啟動路徑（有歷史觀眾）"]
            WP1[查詢觀眾權重數據庫<br/>< 10ms]
        end

        CS3 --> WEIGHTS[權重向量<br/>w_ASD, w_SCR, w_OFM,<br/>w_SD, w_CC, b]
        WP1 --> WEIGHTS
    end

    %% ===== 第三層：留存率預測與分析服務 =====
    subgraph Layer3["第三層：留存率預測與分析服務"]
        direction TB

        subgraph Prediction["留存率計算（< 10ms）"]
            CALC[線性計算引擎<br/>Retention = Σ Feature_i × Weight_i + b]
            SIG[Sigmoid 歸一化<br/>映射到 0-1]
            CALC --> SIG
        end

        subgraph Explain["可解釋分析（< 150ms）"]
            EX1[線性分解<br/>每個特徵的貢獻值<br/>SCR: +8.1, OFM: +4.9]
            EX2[SHAP 解釋<br/>每個特質對權重的貢獻<br/>外向性 → w_SCR +0.35]
        end

        subgraph Advanced["進階分析（2-3 秒，可選）"]
            CF1[反事實分析<br/>「SCR 5→7，留存率 +15%」]
            TS1[時序分析<br/>每 10 秒分段計算<br/>留存率曲線]
        end

        SIG --> EX1
        SIG --> EX2
        SIG --> CF1
        SIG --> TS1
    end

    %% ===== 報告輸出 =====
    subgraph Output["📊 分析報告"]
        RPT1[留存率預測分數<br/>例：0.82（82%）]
        RPT2[特徵貢獻排序<br/>SCR > OFM > CC > SD > ASD]
        RPT3[權重來源解釋<br/>為什麼這個觀眾重視 SCR]
        RPT4[優化建議<br/>具體改進方向與預期效果]
        RPT5[留存率曲線<br/>逐段分析，定位問題時間段]
    end

    %% ===== 離線產物（虛線連接）=====
    subgraph OfflineArtifacts["🔧 離線訓練產物"]
        OA1[(FAISS 語義索引<br/>PersonaChat + Big5-Chat)]
        OA2[(XGBoost 映射模型<br/>mapping_model.pkl)]
        OA3[(觀眾權重數據庫<br/>100K 權重向量)]
        OA4[(特徵標準化參數<br/>全數據集統計分佈)]
    end

    %% ===== 主流程連接 =====
    U1 --> VP1
    U2 --> CS1

    CACHE --> CALC
    WEIGHTS --> CALC

    EX1 --> RPT1
    EX1 --> RPT2
    EX2 --> RPT3
    CF1 --> RPT4
    TS1 --> RPT5

    %% ===== 離線產物到線上的連接（虛線）=====
    OA1 -.->|語義索引| CS1
    OA2 -.->|映射模型| CS3
    OA3 -.->|歷史權重| WP1
    OA4 -.->|標準化參數| NORM

    %% ===== 特徵緩存到時序分析 =====
    CACHE --> TS1

    %% ===== 樣式 =====
    style Entry fill:#fff3e0,stroke:#e65100,stroke-width:2px
    style Layer1 fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Layer2 fill:#fce4ec,stroke:#c62828,stroke-width:2px
    style Layer3 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style Output fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style OfflineArtifacts fill:#f5f5f5,stroke:#9e9e9e,stroke-width:1px,stroke-dasharray: 5 5
    style VideoProcessing fill:#e1f5fe
    style FeatureWorkers fill:#e1f5fe
    style Normalize fill:#e1f5fe
    style ColdStart fill:#ffebee
    style WarmPath fill:#ffebee
    style Prediction fill:#e8f5e9
    style Explain fill:#e8f5e9
    style Advanced fill:#e8f5e9
```

## 線上流程步驟說明

### 整體流程時間線

```
創作者操作                    系統處理                         時間
─────────────────────────────────────────────────────────────────
上傳影片 ──────────────→ 影片接收 + 轉碼                    即時
                         異步特徵提取（4 Workers 並行）      1-2 分鐘
輸入觀眾描述 ──────────→ FAISS 語義搜索                     < 100ms
                         LLM 提取特質                        2-3 秒
                         映射模型預測權重                     < 100ms
                         ─────────────────────────────────
                         留存率計算                           < 10ms
                         線性分解 + SHAP 解釋                 < 150ms
                         反事實分析 + 時序分析                 2-3 秒
                         ─────────────────────────────────
收到完整分析報告 ←──────── 報告生成                          即時
─────────────────────────────────────────────────────────────────
總計：約 2 分鐘（瓶頸在特徵提取）
```

### 第一層：影片特徵提取服務

對應三層架構的第一層「客觀影片特徵提取」。

| 組件 | 技術選型 | 職責 | 延遲 |
|------|---------|------|------|
| 影片接收 | FastAPI | 接收上傳、驗證格式 | 即時 |
| 影片轉碼 | FFmpeg | 統一格式、提取幀 | 5-10 秒 |
| 任務分發 | Celery + RabbitMQ | 將特徵提取任務分發到 Workers | 即時 |
| 場景檢測 Worker | PySceneDetect 0.6.x | 計算 ASD、SCR | 20-30 秒 |
| 光流計算 Worker | OpenCV 4.x Farneback | 計算 OFM | 30-40 秒 |
| 語音檢測 Worker | Whisper base/small | 計算 SD | 30-60 秒 |
| 視覺特徵 Worker | CLIP ViT-B/32 | 計算 CC | 20-30 秒 |
| 標準化引擎 | NumPy | 映射到 1-10 分制 | < 1ms |
| 特徵緩存 | Redis | 緩存特徵向量，TTL 24h | < 1ms |

四個 Workers 並行執行，總時間取決於最慢的 Worker（Whisper，約 30-60 秒）。

### 第二層：觀眾權重建模服務

對應三層架構的第二層「觀眾權重學習」。線上階段有兩條路徑：

**冷啟動路徑**（新觀眾，無歷史數據）：

| 步驟 | 組件 | 技術選型 | 延遲 |
|------|------|---------|------|
| 3.1 | FAISS 語義搜索 | Sentence Transformer + FAISS | < 100ms |
| 3.2 | LLM 特質提取 | GPT-4 / Qwen + Few-shot Prompt | 2-3 秒 |
| 3.3 | 權重映射 | XGBoost 回歸模型 | < 100ms |

**暖啟動路徑**（有歷史數據的觀眾）：

| 步驟 | 組件 | 技術選型 | 延遲 |
|------|------|---------|------|
| 直接查詢 | 權重數據庫 | PostgreSQL / Redis | < 10ms |

### 第三層：留存率預測與分析服務

對應三層架構的第三層「留存率計算」，加上可解釋性分析。

| 功能 | 計算方式 | 延遲 | 輸出 |
|------|---------|------|------|
| 留存率計算 | Σ Feature_i × Weight_i + b → Sigmoid | < 10ms | 0-1 分數 |
| 線性分解 | 直接拆解每個 Feature × Weight | < 1ms | 特徵貢獻排序 |
| SHAP 解釋 | TreeExplainer 分析 XGBoost | < 100ms | 特質→權重貢獻 |
| 反事實分析 | 反向求解 ΔFeature = ΔRetention / Weight | < 50ms | 優化建議 |
| 時序分析 | 分段提取特徵 → 逐段計算留存率 | 2-3 秒 | 留存率曲線 |

## 離線訓練產物

線上流程依賴以下離線訓練產物（一次性生成，定期更新）：

| 產物 | 來源 | 大小 | 更新頻率 |
|------|------|------|---------|
| FAISS 語義索引 | PersonaChat + Big5-Chat | ~500MB | 每月 |
| XGBoost 映射模型 | MicroLens-100K 訓練 | ~10MB | 每月 |
| 觀眾權重數據庫 | 100K 用戶 Ridge Regression | ~50MB | 每週 |
| 特徵標準化參數 | 全數據集統計分佈 | < 1KB | 每月 |

## 技術棧總覽

```
┌─────────────────────────────────────────────────────────┐
│                     前端 (Frontend)                      │
│  React + TypeScript                                      │
│  創作者儀表板 / 影片上傳 / 報告視覺化                    │
├─────────────────────────────────────────────────────────┤
│                   API 網關 (Gateway)                     │
│  FastAPI (Python) + Nginx                                │
│  JWT 認證 / Rate Limiting / SSL                          │
├─────────────────────────────────────────────────────────┤
│               線上服務層 (Online Services)                │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 影片處理服務  │  │ 觀眾建模服務  │  │ 預測分析服務  │  │
│  │              │  │              │  │              │  │
│  │ FFmpeg       │  │ FAISS        │  │ 線性計算     │  │
│  │ Celery       │  │ LLM API      │  │ SHAP         │  │
│  │ 4x Workers   │  │ XGBoost      │  │ 反事實引擎   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│              模型服務層 (Model Serving)                   │
│                                                          │
│  PySceneDetect │ OpenCV │ Whisper │ CLIP │ XGBoost      │
│  (各自 Docker 容器，GPU 加速)                            │
├─────────────────────────────────────────────────────────┤
│               數據層 (Data Layer)                        │
│                                                          │
│  Redis          │ PostgreSQL      │ MinIO               │
│  特徵緩存       │ 權重/元數據     │ 影片文件存儲        │
├─────────────────────────────────────────────────────────┤
│              消息隊列 (Message Queue)                    │
│  RabbitMQ + Celery Workers                               │
│  異步特徵提取任務                                        │
├─────────────────────────────────────────────────────────┤
│              基礎設施 (Infrastructure)                   │
│  Docker Compose (開發) / Kubernetes (生產)               │
│  Prometheus + Grafana (監控)                             │
└─────────────────────────────────────────────────────────┘
```

