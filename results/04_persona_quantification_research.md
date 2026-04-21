# Persona 量化與評估研究報告

## 摘要

本文檔補充 SimLens 系統中關於 persona (代理人) 量化和評估的研究基礎。我們探討如何將個性、工作、興趣、MBTI、Big Five 等人格特質量化為可測量的指標,並提出客觀評估代理人行為準確性的方法。

## 1. Persona 特質量化框架

### 1.1 Big Five 人格特質 (Five-Factor Model)

Big Five 是心理學中最廣泛接受的人格模型,包含五個維度:

**1. Openness (開放性)**: 對新體驗的開放程度
- 測量範圍: 0-100 分
- 高分特徵: 好奇、創造力強、喜歡探索
- 低分特徵: 保守、傳統、偏好熟悉事物
- 與影片偏好關聯: 高開放性用戶偏好多樣化、創新內容

**2. Conscientiousness (盡責性)**: 組織性和目標導向程度
- 測量範圍: 0-100 分
- 高分特徵: 有條理、負責任、自律
- 低分特徵: 隨性、靈活、自發
- 與影片偏好關聯: 高盡責性用戶偏好教育性、結構化內容

**3. Extraversion (外向性)**: 社交活躍度和能量來源
- 測量範圍: 0-100 分
- 高分特徵: 外向、健談、尋求刺激
- 低分特徵: 內向、安靜、偏好獨處
- 與影片偏好關聯: 高外向性用戶偏好社交、娛樂性內容

**4. Agreeableness (親和性)**: 合作和同理心程度
- 測量範圍: 0-100 分
- 高分特徵: 友善、信任他人、樂於助人
- 低分特徵: 競爭性強、懷疑、直接
- 與影片偏好關聯: 高親和性用戶偏好溫馨、正面情感內容

**5. Neuroticism (神經質)**: 情緒穩定性
- 測量範圍: 0-100 分
- 高分特徵: 焦慮、情緒波動、敏感
- 低分特徵: 冷靜、情緒穩定、抗壓
- 與影片偏好關聯: 高神經質用戶可能避免高壓力、緊張內容

### 1.2 MBTI 人格類型

MBTI 將人格分為 16 種類型,基於四個維度:

**維度 1: Extraversion (E) vs Introversion (I)**
- 量化方式: 二元分類或 0-100 連續分數
- 影響: 社交內容偏好

**維度 2: Sensing (S) vs Intuition (N)**
- 量化方式: 二元分類或 0-100 連續分數
- 影響: 具體 vs 抽象內容偏好

**維度 3: Thinking (T) vs Feeling (F)**
- 量化方式: 二元分類或 0-100 連續分數
- 影響: 邏輯 vs 情感內容偏好

**維度 4: Judging (J) vs Perceiving (P)**
- 量化方式: 二元分類或 0-100 連續分數
- 影響: 結構化 vs 開放式內容偏好

### 1.3 Agent4Rec 社交特質

基於 Agent4Rec 論文 (arXiv:2310.10108v3),定義三個可量化的社交特質:

**1. Activity (活躍度)**

```
T_activity(u) = Σ y_ui (用戶 u 觀看的影片總數)
```
- 測量: 用戶與推薦系統的互動頻率
- 分層: 低活躍 (< 33 percentile), 中活躍 (33-66 percentile), 高活躍 (> 66 percentile)
- 影響: 高活躍用戶會觀看更多影片,容忍度更高

**2. Conformity (從眾性)**
```
T_conformity(u) = (1/N) Σ |r_ui - R_i|²
```
其中 r_ui 是用戶評分, R_i 是影片平均評分
- 測量: 用戶評分與大眾評分的一致性
- 分層: 低從眾 (獨特品味), 中從眾, 高從眾 (跟隨主流)
- 影響: 低從眾用戶對小眾內容接受度更高

**3. Diversity (多樣性)**
```
T_diversity(u) = |∪ G_i| (用戶觀看的影片類型總數)
```
- 測量: 用戶觀看內容的類型廣度
- 分層: 低多樣性 (專注特定類型), 中多樣性, 高多樣性 (廣泛興趣)
- 影響: 高多樣性用戶對不同風格內容接受度更高

### 1.4 人口統計學特徵

**年齡 (Age)**
- 量化: 連續變量 (18-65+)
- 分組: 18-24, 25-34, 35-44, 45-54, 55+
- 研究依據: 不同年齡層對影片節奏偏好不同 (年輕用戶偏好快節奏)

**職業 (Occupation)**
- 量化: 類別變量
- 分類: 學生、教育工作者、科技業、創意產業、服務業等
- 影響: 職業背景影響內容主題偏好

**興趣領域 (Interests)**
- 量化: 多標籤分類
- 類別: 科技、藝術、運動、音樂、旅遊、美食等
- 影響: 直接影響內容主題偏好

## 2. Persona 到影片特徵權重的映射

### 2.1 基於數據驅動的映射方法


**方法 1: 從 PersonaChat/Big5-Chat 數據集學習**

PersonaChat 數據集包含 8,000+ 個 persona 描述,Big5-Chat 包含 100,000 個基於 Big Five 的對話。

```python
# 步驟 1: 從對話數據集提取 persona 特質
def extract_persona_traits(persona_description):
    """
    輸入: "I am a 25-year-old student who loves action movies..."
    輸出: {
        'age': 25,
        'occupation': 'student',
        'big5_openness': 75,
        'big5_extraversion': 60,
        ...
    }
    """
    # 使用 LLM 或訓練好的分類器提取特質
    traits = llm_extract_traits(persona_description)
    return traits

# 步驟 2: 將 persona 特質映射到影片特徵權重
def map_traits_to_weights(persona_traits):
    """
    輸入: persona 特質向量
    輸出: 影片特徵權重向量 [w_ASD, w_SCR, w_OFM, w_SD, w_CC]
    """
    # 使用從 MicroLens-100K 學習的映射函數
    weights = learned_mapping_function(persona_traits)
    return weights
```

**方法 2: 基於 MicroLens-100K 的聚類映射**

```python
# 步驟 1: 對 MicroLens-100K 用戶進行聚類
from sklearn.cluster import KMeans

# 基於觀看行為學習的權重向量進行聚類
user_weights = [learn_weights(user) for user in microlens_users]
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit(user_weights)

# 步驟 2: 為每個聚類標註 persona 特質
cluster_personas = {
    0: {'age_range': '18-24', 'activity': 'high', 'big5_extraversion': 70},
    1: {'age_range': '25-34', 'activity': 'medium', 'big5_openness': 80},
    ...
}

# 步驟 3: 新 persona 映射到最近的聚類
def map_new_persona(new_persona_traits):
    nearest_cluster = find_nearest_cluster(new_persona_traits, cluster_personas)
    weights = cluster_centers[nearest_cluster]
    return weights
```

### 2.2 特質到權重的理論映射

基於心理學研究和影片分析文獻,建立理論映射關係:

**Big Five Openness → 影片特徵權重**

- 高開放性 → 高 Content Complexity (CC) 權重 (偏好多樣化、創新內容)
- 高開放性 → 高 Shot Change Rate (SCR) 權重 (接受快速變化)

**Big Five Extraversion → 影片特徵權重**
- 高外向性 → 高 Optical Flow Magnitude (OFM) 權重 (偏好動態、刺激內容)
- 高外向性 → 高 Speech Density (SD) 權重 (偏好社交、對話內容)

**Big Five Conscientiousness → 影片特徵權重**
- 高盡責性 → 高 Speech Density (SD) 權重 (偏好信息密集、教育內容)
- 高盡責性 → 低 Shot Change Rate (SCR) 權重 (偏好結構化、穩定節奏)

**Age → 影片特徵權重**
- 年輕用戶 (18-24) → 高 SCR 和 OFM 權重 (偏好快節奏、動態內容)
- 中年用戶 (35-50) → 高 SD 權重 (偏好信息密集內容)
- 老年用戶 (55+) → 低 SCR 權重 (偏好慢節奏內容)

**Activity (活躍度) → 行為模式**
- 高活躍度 → 更長的觀看時間,更高的容忍度
- 低活躍度 → 更快退出,更挑剔

## 3. Persona 代理人評估框架

### 3.1 PersonaGym 評估框架

基於 arXiv:2407.18416v4 的 PersonaGym 框架,評估 persona 代理人的一致性。

**PersonaScore 指標**

PersonaScore 基於決策理論,評估代理人行為與 persona 描述的一致性:

```
PersonaScore = Σ w_i × consistency(action_i, persona)
```

其中:
- action_i: 代理人在情境 i 的行為
- consistency(): 行為與 persona 一致性函數
- w_i: 情境 i 的權重

**評估維度**

1. **Preference Consistency (偏好一致性)**
   - 測量: 代理人對影片的選擇是否符合其 persona 描述的偏好
   - 指標: Accuracy, Precision, Recall, F1 Score
   - 方法: 給定 persona 和影片對,預測代理人是否會觀看

2. **Rating Consistency (評分一致性)**

   - 測量: 代理人的評分分佈是否符合真實用戶的評分模式
   - 指標: KL Divergence, Jensen-Shannon Divergence
   - 方法: 比較代理人評分分佈與真實用戶評分分佈

3. **Behavioral Consistency (行為一致性)**
   - 測量: 代理人的觀看時長、退出行為是否符合 persona 特質
   - 指標: MAE (Mean Absolute Error), Pearson Correlation
   - 方法: 預測觀看時長,與真實數據比較

4. **Trait Coherence (特質連貫性)**
   - 測量: 代理人在多輪互動中是否保持 persona 特質一致
   - 指標: Trait Stability Score
   - 方法: 在長期互動中檢測特質漂移

### 3.2 Agent Alignment 評估 (基於 Agent4Rec)

**評估方法 1: User Taste Alignment**

測試代理人是否能準確識別符合其 persona 的影片:

```python
# 設置: 給代理人 20 個影片,其中 m 個是符合其偏好的
# 評估: 代理人能否正確識別這些影片

def evaluate_taste_alignment(agent, test_videos, ratio='1:1'):
    """
    ratio: 符合偏好 vs 不符合偏好的比例
    返回: Accuracy, Recall, Precision, F1 Score
    """
    predictions = [agent.would_watch(video) for video in test_videos]
    ground_truth = [video.matches_persona(agent.persona) for video in test_videos]
    
    return compute_metrics(predictions, ground_truth)
```

**預期結果** (基於 Agent4Rec 論文):
- Accuracy: ~65-70%
- Recall: ~75%
- Precision: 隨比例變化 (1:1 時 ~70%, 1:9 時 ~25%)

**評估方法 2: Rating Distribution Alignment**

測試代理人的評分分佈是否符合真實用戶模式:

```python
def evaluate_rating_distribution(agent_ratings, real_user_ratings):
    """
    比較代理人評分分佈與真實用戶評分分佈
    """
    # KL Divergence: 越小越好
    kl_div = compute_kl_divergence(agent_ratings, real_user_ratings)
    
    # Chi-square test: p > 0.05 表示分佈無顯著差異
    chi2, p_value = chi2_test(agent_ratings, real_user_ratings)
    
    return {'kl_divergence': kl_div, 'p_value': p_value}
```

**評估方法 3: Social Trait Alignment**

測試代理人的社交特質 (activity, conformity, diversity) 是否符合設定:

```python
def evaluate_social_traits(agent, expected_traits):
    """
    測量代理人實際表現的社交特質與預期的差異
    """
    actual_activity = count_videos_watched(agent)
    actual_conformity = compute_conformity_score(agent)
    actual_diversity = count_unique_genres(agent)
    
    # 計算與預期的差異
    trait_error = {
        'activity': abs(actual_activity - expected_traits['activity']),
        'conformity': abs(actual_conformity - expected_traits['conformity']),
        'diversity': abs(actual_diversity - expected_traits['diversity'])
    }
    
    return trait_error
```

### 3.3 與真實世界的差距評估

**方法 1: Holdout Validation**

```python
# 步驟 1: 將 MicroLens-100K 分為訓練集和測試集
train_users, test_users = train_test_split(microlens_users, test_size=0.2)

# 步驟 2: 從訓練集學習 persona 到權重的映射
mapping_function = learn_mapping(train_users)

# 步驟 3: 為測試集用戶創建代理人
test_agents = [create_agent(user.persona, mapping_function) for user in test_users]

# 步驟 4: 比較代理人行為與真實用戶行為
for agent, real_user in zip(test_agents, test_users):
    # 預測觀看時長
    predicted_retention = agent.predict_retention(test_videos)
    actual_retention = real_user.actual_retention(test_videos)
    
    # 計算誤差
    mae = mean_absolute_error(predicted_retention, actual_retention)
    correlation = pearsonr(predicted_retention, actual_retention)
```

**預期結果**:
- MAE < 0.2 (留存率預測誤差小於 20%)
- Pearson r > 0.4 (中等相關性)

**方法 2: A/B Testing Simulation**

```python
# 模擬 A/B 測試,比較代理人反饋與真實用戶反饋的一致性

def simulate_ab_test(recommendation_algorithm_A, recommendation_algorithm_B):
    # 使用代理人評估兩個推薦算法
    agent_preference_A = evaluate_with_agents(algorithm_A)
    agent_preference_B = evaluate_with_agents(algorithm_B)
    
    # 使用真實用戶評估兩個推薦算法
    real_preference_A = evaluate_with_real_users(algorithm_A)
    real_preference_B = evaluate_with_real_users(algorithm_B)
    
    # 檢查代理人和真實用戶是否得出相同結論
    agent_winner = 'A' if agent_preference_A > agent_preference_B else 'B'
    real_winner = 'A' if real_preference_A > real_preference_B else 'B'
    
    agreement = (agent_winner == real_winner)
    return agreement
```

**評估指標**:
- Agreement Rate: 代理人與真實用戶得出相同結論的比例
- 目標: > 80% agreement rate

**方法 3: Causal Discovery Validation**

基於 Agent4Rec 的因果發現方法,驗證代理人是否能揭示真實的因果關係:

```python
# 使用代理人數據進行因果發現
agent_causal_graph = discover_causal_graph(agent_data)

# 使用真實用戶數據進行因果發現
real_causal_graph = discover_causal_graph(real_user_data)

# 比較兩個因果圖的相似度
graph_similarity = compute_graph_similarity(agent_causal_graph, real_causal_graph)
```

**評估指標**:
- Edge Precision: 代理人發現的因果關係中有多少是真實存在的
- Edge Recall: 真實存在的因果關係中有多少被代理人發現
- 目標: Precision > 0.7, Recall > 0.6

## 4. 實驗設計建議

### 4.1 實驗一: Persona 特質提取驗證

**目的**: 驗證從文本描述中提取 persona 特質的準確性

**方法**:
1. 使用 PersonaChat 數據集的 persona 描述
2. 使用 LLM 提取 Big Five 特質分數
3. 與人工標註的 Big5-Chat 數據集比較

**評估指標**:
- MAE < 10 (在 0-100 量表上)
- Pearson r > 0.6

**數據集**: PersonaChat (8,000+ personas), Big5-Chat (100,000 dialogues)

### 4.2 實驗二: Persona 到權重映射驗證

**目的**: 驗證 persona 特質到影片特徵權重的映射準確性

**方法**:
1. 從 MicroLens-100K 學習用戶權重
2. 為每個用戶標註 persona 特質 (基於觀看歷史)
3. 訓練映射函數: persona_traits → feature_weights
4. 在測試集上評估映射準確性

**評估指標**:
- Weight MAE < 0.15
- Weight Correlation > 0.5

**數據集**: MicroLens-100K (100,000 users, 719,405 interactions)

### 4.3 實驗三: 代理人行為一致性驗證

**目的**: 驗證代理人行為與 persona 描述的一致性

**方法**:
1. 創建 1,000 個代理人,每個有不同的 persona
2. 讓代理人與推薦系統互動 (觀看、評分、退出)
3. 評估行為一致性 (taste alignment, rating distribution, social traits)

**評估指標**:
- Taste Alignment Accuracy > 65%
- Rating Distribution KL Divergence < 0.1
- Social Trait Error < 20%

**數據集**: MicroLens-100K (用於驗證)

### 4.4 實驗四: 與真實世界差距評估

**目的**: 評估代理人反饋與真實用戶反饋的差距

**方法**:
1. Holdout Validation: 在測試集上比較代理人預測與真實行為
2. A/B Testing Simulation: 比較代理人和真實用戶對推薦算法的偏好
3. Causal Discovery: 比較代理人和真實用戶揭示的因果關係

**評估指標**:
- Retention Prediction MAE < 0.2
- A/B Test Agreement Rate > 80%
- Causal Graph Similarity > 0.7

**數據集**: MicroLens-100K (測試集)

## 5. 研究依據與引用

### 5.1 Persona 量化相關論文

1. **PersonaChat Dataset** (Zhang et al., 2018)
   - 8,000+ persona 描述
   - 用於訓練對話系統的 persona 一致性

2. **Big5-Chat Dataset** (arXiv:2410.16491v1)
   - 100,000 個基於 Big Five 的對話
   - 用於訓練 LLM 的人格表達

3. **PersonalityChat** (arXiv:2401.07363v1)
   - 基於 PersonaChat 和 Big Five 的合成數據集
   - 用於個性化對話建模

### 5.2 Persona 評估相關論文

1. **PersonaGym** (arXiv:2407.18416v4)
   - 第一個動態評估 persona 代理人的框架
   - PersonaScore: 基於決策理論的自動化評估指標

2. **Agent4Rec** (arXiv:2310.10108v3)
   - LLM 驅動的推薦系統用戶模擬器
   - 定義了 activity, conformity, diversity 三個社交特質
   - 評估了代理人與真實用戶的對齊程度

3. **SimTube** (arXiv:2411.09577v2)
   - 使用 persona 生成影片評論的系統
   - 從 PersonaChat 數據集查詢相關 persona

### 5.3 Big Five 與影片偏好相關論文

1. **Personality Prediction from Video** (arXiv:1805.00705)
   - 從影片預測 Big Five 人格特質
   - 證明了人格特質與影片內容的關聯

2. **Personality Analysis from Short Videos** (arXiv:2411.00813v1)
   - 從短影片平台分析用戶人格
   - 使用多模態數據推斷 Big Five 特質

## 6. 實現建議

### 6.1 Persona 特質提取模組

```python
class PersonaExtractor:
    def __init__(self, llm_model):
        self.llm = llm_model
        
    def extract_traits(self, persona_description):
        """
        從文本描述中提取 persona 特質
        """
        prompt = f"""
        Given the following persona description:
        "{persona_description}"
        
        Extract the following traits (0-100 scale):
        - Big Five: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism
        - MBTI: E/I, S/N, T/F, J/P (0-100 for each dimension)
        - Demographics: Age, Occupation, Interests
        - Social Traits: Activity, Conformity, Diversity
        
        Return as JSON.
        """
        
        traits = self.llm.generate(prompt)
        return json.loads(traits)
```

### 6.2 Persona 到權重映射模組

```python
class PersonaToWeightMapper:
    def __init__(self, microlens_data):
        # 從 MicroLens-100K 學習映射函數
        self.mapping_model = self.train_mapping(microlens_data)
        
    def train_mapping(self, data):
        """
        訓練 persona 特質到影片特徵權重的映射
        """
        X = []  # persona 特質向量
        y = []  # 影片特徵權重向量
        
        for user in data:
            traits = self.extract_user_traits(user)
            weights = self.learn_user_weights(user)
            X.append(traits)
            y.append(weights)
        
        # 使用神經網絡或其他回歸模型
        model = train_regression_model(X, y)
        return model
    
    def map(self, persona_traits):
        """
        將 persona 特質映射到影片特徵權重
        """
        weights = self.mapping_model.predict(persona_traits)
        return weights
```

### 6.3 Persona 代理人評估模組

```python
class PersonaEvaluator:
    def evaluate_alignment(self, agent, test_data):
        """
        評估代理人與 persona 的對齊程度
        """
        results = {
            'taste_alignment': self.evaluate_taste(agent, test_data),
            'rating_distribution': self.evaluate_ratings(agent, test_data),
            'social_traits': self.evaluate_traits(agent, test_data),
            'behavioral_consistency': self.evaluate_behavior(agent, test_data)
        }
        return results
    
    def evaluate_real_world_gap(self, agents, real_users):
        """
        評估代理人與真實用戶的差距
        """
        gap_metrics = {
            'retention_mae': self.compute_retention_error(agents, real_users),
            'ab_test_agreement': self.simulate_ab_test(agents, real_users),
            'causal_graph_similarity': self.compare_causal_graphs(agents, real_users)
        }
        return gap_metrics
```

## 7. 總結

本文檔提供了完整的 persona 量化和評估框架,包括:

1. **量化方法**: Big Five, MBTI, 社交特質, 人口統計學
2. **映射方法**: 從 persona 特質到影片特徵權重的數據驅動映射
3. **評估框架**: PersonaGym, Agent4Rec 的評估方法
4. **差距評估**: Holdout Validation, A/B Testing, Causal Discovery
5. **實驗設計**: 四個可行的實驗,使用現有數據集
6. **研究依據**: 所有方法都有學術論文支持

這個框架確保了:
- ✅ 所有 persona 特質都是可量化的
- ✅ 映射方法基於真實數據,不是 LLM 幻覺
- ✅ 評估方法客觀、可重現
- ✅ 可以測量與真實世界的差距
- ✅ 所有實驗都是可行的,使用現有數據集

## 參考文獻

1. Zhang et al. (2018). "Personalizing Dialogue Agents: I have a dog, do you have pets too?" ACL 2018.
2. arXiv:2410.16491v1. "Shaping LLM Personalities Through Training on Human-Grounded Data"
3. arXiv:2401.07363v1. "PersonalityChat: Conversation Distillation for Personalized Dialog Modeling"
4. arXiv:2407.18416v4. "PersonaGym: Evaluating Persona Agents and LLMs"
5. arXiv:2310.10108v3. "Agent4Rec: On Generative Agents in Recommendation"
6. arXiv:2411.09577v2. "SimTube: Generating Simulated Video Comments through Multimodal AI and User Personas"
7. arXiv:1805.00705. "Investigating Audio, Video, and Text Fusion Methods for End-to-End Automatic Personality Prediction"
8. arXiv:2411.00813v1. "Personality Analysis from Online Short Video Platforms with Multi-domain Adaptation"
