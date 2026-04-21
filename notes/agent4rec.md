Agent4Rec

link: https://arxiv.org/pdf/2310.10108
released on 2024 Nov
陳彥良給的
github
code:https://github.com/LehengTHU/Agent4Rec

introduction

傳統的監督式推薦方法仍顯不足，這體現在離線指標
(offline metrics) 與在線表現 (online performance)
之間存在顯著差距。這種脫節阻礙了學術研究與現實
推薦部署的整合，成為該領域未來發展的瓶頸

experiments

使用者角度

系統角度

重點在於評估智能體對齊程度（agent
alignment），即智能體在多大程度上能確保真
實用戶社交特徵、個性和偏好的連貫性。

評估配置不同算法的推薦器，指標涵蓋平均推薦/觀看
數、用戶評分、參與時間及整體滿意度

Agent Alignment 我們可以參考部分 他對於代理人的組成很有一套
然後他 filter bubble effect 部分感覺可以參考嗎?
 Dataset 部分也可以再看一下 有哪些欄位 可以做什麼
實驗部分其實內容也很多 像是 recommendation stategy evaluation 那邊
的我覺得可以用再熱力圖評分 就很多可以參考

潛在問題

在模擬器中模擬了過濾氣泡效應 (Filter Bubble
Effect) 觀察用戶是否因為持續接觸到相似或強化的內
容，導致項目屬性多樣性減少

利用模擬器作為數據收集工具，開創了面向數據的因果發現
(Data-oriented Causal Discovery)，產出了一個穩健的因
果圖，使我們能夠揭示複雜的潛在因果關係

模擬資訊同溫層

method

代理人架構

Recommendation
Environment
推薦環境

profile module

memory module

action module

benchmark dataset (MovieLens-1M, Steam,
AmazonBook) is used for initialization

social traits 

用於捕捉推薦場景中個人的個性與特性元
素，即：活躍度（activity）、從眾性
（conformity）與多樣性（diversity）

unique traits

為了以自然語言記錄使用者偏好，我們從每位用戶的
觀看紀錄中隨機選取了25個項目。評分為3分或以上
的項目被用戶歸類為「喜歡」，而評分低於 3 分的
項目則被視為「不喜歡」。接著，利用 ChatGPT
提取總結該用戶所表現出的獨特口味與評分模式。

活躍度量化了用戶與推薦項目交互的頻率與廣度，
區分了那些廣泛觀看並為大量項目評分的用戶，以
及那些將自己局限於極少數項目的用戶

從眾度探究了用戶的評分與項目平均評分的一
致程度，區分了具有獨特見解的用戶，與那些
觀點緊貼大眾情緒

多樣性反映了用戶對於多樣化項目類
別（genres）的傾向，或是他們對特
定類別的偏好

Factual memory

事實記憶封裝了推薦系統內的交互行為

Emotional memory

情感記憶則捕捉了源於這些交互的心理感受。記錄
了用戶在系統交互過程中的感受，例如疲勞程度和
整體滿意度。我們的目標是確保生成式智能體不僅
僅是基於過去的事實交互做出反應，還會考慮到感
受，從而更緊密地鏡像真實的人類行為。

以兩種格式存儲記憶：自然語言描述和向量表
示。前者旨在方便人類理解，而向量表示則為高
效的記憶檢索和提取做好了準備

記憶檢索: 此操作協助智能體從其記憶模組中提取最
相關的信息

記憶寫入：此操作能夠將智能體模擬的交互和情緒記
錄到記憶流（memory stream）中

記憶反思：意識到情緒對推薦中用戶行為的影響，我
們納入了一種情緒驅動的自我反思機制。憑藉 LLM
的強大能力，智能體會內省其對推薦的滿意度並評估
其疲勞程度，從而提供對其認知狀態的更深層理解

taste-driven action

emotion-driven action

在口味的引導下，智能體會評估頁面上的每個項目，判斷其是
否與自己的偏好一致。他們可能會選擇觀看某些激發其興趣的
項目，同時繞過（bypassing）其他項目，隨後為其觀看的每個
項目提供評分與感受

代理人對先前推薦項目的滿意度及其當前的疲勞程度，會影
響其決定繼續探索更多推薦頁面，還是退出推薦系統。透過
思維鏈（Chain-of-Thought, CoT）增強了代理人的情緒推
理能力，自主地表達其對當前推薦頁面的滿意度與疲勞程
度。結合這些洞察以及其個性化的活躍度特徵（activity
trait），代理人決定是否退出系統。

Item Profile Generation

用來捕捉關鍵物品特徵 (品質從歷史評分提取，熱門程度是依照觀看次數)。
我們的目標不僅是在輪廓中封裝項目的獨特性以模擬真實用戶的推薦場景，
還旨在測試 LLM 對該項目是否存在潛在的幻覺（hallucinations）。使用
few-shot learning，讓 LLM 憑項目標題分成18累病生成摘要。若 LLM 分類
錯誤，該項目會被剪枝 (pruned 剔除) 以降低幻覺風險，避免 LLM 不懂裝懂
的情況。

page-by-page recommendation scenario

模擬器鏡像現實世界推薦平台運作方式，已分頁的形式進
行。基於交互、偏好與反饋，後續頁面會進一步調整推薦
內容，旨在提供更精緻的用戶體驗

Algorithm Design

推薦的算法被建構為獨立的模組，目的在於可擴展
性。目前用了 colaborative filitering strategies
including Matrix Factorization, Light GCN,
MultVAE。

仁祥的想法

MovieLens

(Gemini 整理)

1. ratings.csv (評分數據)：最核心的檔案。包含 userId、movieId、rating (0.5 到 5.0 星) 以及時間戳記。
2. tags.csv (標籤數據)：用戶幫電影打的標籤（例如：深奧、驚悚、反烏托邦）
3. movies.csv (電影資訊)：包含電影標題（含年份）和類型 (Genres)（如：動作、喜劇、科幻等 19 種標籤）。
4. links.csv (外部連結)：提供對應到 IMDb 和 TMDB 的 ID，方便你爬取海報或更詳細的劇情摘要。
5. genome-scores.csv & genome-tags.csv (標籤基因組)：這比較進階。它不是原始標籤，而是透過演算法計算
出每部電影與特定標籤的相關性得分（例如：某部片「燒腦」的程度是多少）。

主要是在了解每一個使用者對於個別電影的評分

https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset 
有不同數據量的版本 這是 20m version

Steam

Amazon-book

Gemini 整理過

1. steam_games.json (遊戲元數據)：包含遊戲名稱、開發商 (developer)、類別 (genres)、用戶自定義標籤
(tags)、價格、發行日期以及 Metacritic 評分。這對於做「內容過濾 (Content-based)」或「多模態分析」非
常有幫助。
2. user_reviews.json (評論數據)：它包含 user_id、item_id 以及真實評論文字 (review)。最重要的欄位是
recommend (是否推薦)，這是二元化的評分訊號（喜歡/不喜歡），還有這則評論獲得的「讚數」。
3. users_items.json (用戶擁有項目與行為數據)：它列出每個用戶擁有的所有遊戲，以及最核心的指標：
playtime_forever (總遊玩分鐘數)。透過這個數值，你可以推斷出用戶的「沉浸程度」，而不僅僅是他們有沒有
買這款遊戲。
4. bundle_data.json (綑綁包資訊)：Steam 經常將多款遊戲綑綁銷售（例如「Valve 經典合集」）。這個檔案
包含綑綁包的名稱、內容物（包含哪些遊戲 ID）以及價格。這可以用來研究「綑綁銷售策略」或「交叉推
薦」。


https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data