link: https://arxiv.org/pdf/2407.06567
the paper is release on 2024 nov
陳彥良給的
有 github 但還沒有程式碼
https://github.com/The-FinAI/FinCon

FinCon

abstract

基於大型語言模型的多代理人框架
具備針對多樣化金融任務的概念性口頭強化
(conceptual verbal reinforcement)

採用 manager-analyst hieracrchy 透過自然語言互動
使跨功能的代理人能為統一目標進行同步協作

introduction

現有專門設計給金融決策的系統像是 FinGPT, FinMEM, FinAgent 都有些限制

1. 他們依賴基於個別代理人對於短期市場波動的風險偏好，未能控制長期風險
2. 局限於單一資產交易任務，對多資產金融應用的適應性差
3. 他們給單個代理人在受限的上下文窗口理解和處理資訊帶來壓力，這會降低決策質量

儘管有向 stockagent 這樣的方法使用多代理人系統進行股票交易，
但他們仰賴多個 LLM 代理人之間討論，導致高昂通訊成本和緩慢決策

1. 提出 manager-analyst 階層通訊框架，並配備風控組件。
2. 不只處理股票交易，也能處裡投資組合管理
3. 雙風險控制組建，在回合內 (within episodes) 和 回合間 (across episodes) 更新風險評估
    - within episodes: 用 Conditional Value at Risk 做監督
    - across episodes: 引入口頭強化機制 (verbal reinforcement mechanism) 
                           根據推理軌跡和損益趨勢更新投資信念，並將其蒸餾成概念性的觀點

Method

Manager-Analyst Agent Group

每個代理人先以單模態的方式處理單一來源的資訊
1. 三個文本數據處理代理人從每日新聞和財務報告中提取見解和情緒
2. 一個音訊代理人用 whisper API 從財報會議錄音解讀投資訊號
3. 一個數據分析代理人利用表格化時間序列計算金融指標
4. 一個選股特工透過量化金融的風險分散方式監督投資組合的選擇

唯一的決策者 負責連續的金融任務生成交易行動
整合多位分析師代理人提取的見解
透過先前交易推理結果作自我反思

manager agent

analyst agents

回合內風險預警是 CVaR 突然下降觸發的。CVaR 代表每
日交易損益中表現最差 1% 的平均值
下降通常表示最近的交易決策導致損益落入了這個最低百分
位數，發出了潛在的高風險市場信號，

透過 Actor-Critic 機制，FinCon 反思一系列
成功與失敗的行動。
反思由讀的概念口頭強化 (Conceptual Verbal
Reinforcement, CVRF) 提供動力。

什麼是概念口頭強化? 簡單比喻

傳統強化學習: 球員進球+1分，球員沒進-1分，球員得練習上千萬次才能靠直覺發現「喔，原來我要跳高一點才會進」
CVPR 口頭強化: 球員投丟後，教練 (風險控制組件) 過來跟她講道理。教練跟他講: 「以後記住：面對包夾時，傳球給
外線空檔才是勝率更高的選擇。」
球員把這句話記在腦海裡，下次遇到同樣情況就會知道怎麼做

概念化? 將失敗抽象化城概念
具體事實：2026年4月10日，台積電因為法說會展望不佳跌了 3%。
概念化觀點：當權值股的法說會展望低於預期時，即使技術面處於多頭，也應優先考慮防禦性操作。

具體上:
第一步: 收集失敗與成功的軌跡: 系統會將每天的盈虧結果、經理思考過程和數據分析報告打包成軌跡
第二步: 拿出這次訓練和上一次訓練的報告做對比 
第三步: 利用文本梯度下降 (text-based gradient desent) 為經理代理人提供概念性投資指導

回合內

回合間

風控組件

開發代理人認知結構以實現類人行為
1. general configuration and profiling module: 定義任務類型、指定交易目標
2. perception module: 定義每個代理人如何和市場互動
3. memory module: 分成工作記憶、程序記憶和情節記憶。程序記憶與情節記憶對於記
錄連續決策過程中的歷史行動、結果與反思至關重要。程序記憶 (Procedural
memory) 在回合（episode）內的每個決策步驟後生成，將數據儲存為記憶事件。針對
交易查詢，會從程序記憶中檢索排名靠前的事件，並根據近時性 (recency)、相關性
(relevance) 與重要性 (importance) 進行排序，

Modular Design of
FINCON Agents

仁祥的個人想法

他設計的代理人架構是我們可以參考的 
下面的 概念化口頭強化學習 也是我們如果要做
影片裁切或更改 這是很能夠參考的一環
很少看到在更新 prompt 的嗎? 所以也蠻特別的