# Related Work

## 1. Agent-Based Recommendation System Simulation

Recent breakthroughs in Large Language Models (LLMs) have introduced a new research paradigm for recommendation systems. Agent4Rec (Zhang et al., 2024) pioneered the use of LLM-driven generative agents to simulate user behavior in recommendation systems. The system successfully simulates real user interactions by equipping each agent with profile, memory, and action modules. Specifically, Agent4Rec's agent architecture comprises three core features:

**Social Traits**: The system quantifies three dimensions of user characteristics: activity, conformity, and diversity. Activity measures the frequency and breadth of user interactions with recommended items; conformity examines how closely user ratings align with average item ratings; diversity reflects users' propensity toward diverse item categories.

**Memory Mechanism**: Agent4Rec distinguishes between factual memory and emotional memory. Factual memory encapsulates interactive behaviors within the recommendation system, while emotional memory captures psychological feelings stemming from these interactions, such as fatigue levels and overall satisfaction. This dual memory mechanism enables agents to react not only based on past factual interactions but also to consider feelings, thereby more closely mirroring real human behavior.

**Action Module**: The system designs two types of actions: taste-driven actions and emotion-driven actions. The former includes watching, rating, and generating post-viewing feelings; the latter involves decisions such as exiting the system and evaluating recommendation lists, which are influenced by the agent's satisfaction with previously recommended items and current fatigue level.

However, Agent4Rec primarily focuses on traditional recommendation scenarios such as movies, games, and books. Its agent preference modeling is based on historical rating data and lacks deep understanding of video content itself. This approach faces significant challenges when dealing with short video recommendations, as the appeal of short videos highly depends on multimodal features such as visual content, pacing, and emotional tension, rather than merely genre tags or historical ratings.

## 2. Video Feature Extraction and Watch Time Prediction

### 2.1 Research Foundation for Objective Video Features

In recent years, video recommendation system research has gradually shifted from subjective evaluation to objective feature quantification. **Watch Time** has been proven to be the most important user engagement metric in video recommendation systems (arXiv:2508.11086, arXiv:2412.20211). Compared to traditional click-through rates or ratings, watch time more directly reflects users' genuine interest in content. **Retention Rate**, the completion rate of video viewing, has become a core metric for measuring content quality.

In terms of video features, **Shot Duration** and **Editing Pace** are measurable objective features with solid research foundations. Research on the evolution of movie pacing shows that **Average Shot Duration (ASD)** has gradually decreased from approximately 12 seconds in the 1930s to 2-4 seconds in modern films, reflecting changes in audience attention patterns and the evolution of visual storytelling techniques. This trend is even more pronounced on short video platforms, where fast-paced editing has become a key factor in attracting and maintaining audience attention.

**Shot Change Rate (SCR)**, the number of shot transitions per minute, is closely related to visual stimulation and attention maintenance. Research shows that appropriate shot change frequency can effectively enhance audience engagement, but excessively high change rates may lead to visual fatigue.

**Motion Intensity**, quantified through Optical Flow analysis, measures visual changes between frames and reflects the dynamic nature of videos. Videos with high motion intensity typically better capture audience attention, particularly in action, sports, and gaming content.

**Speech Density** and **Content Complexity** are two other important objective features. Speech density measures the proportion of video containing speech, reflecting the intensity of information delivery. Content complexity is quantified through visual feature diversity, with high complexity indicating richer visual elements and scene variations.

The advantages of these objective features are: **fully automated extraction**, independent of manual annotation or subjective judgment; **clear computational methods**, implementable using open-source tools (such as PySceneDetect, OpenCV, Whisper, CLIP); **verifiable correlation with watch time**, providing a solid theoretical foundation for prediction models.

### 2.2 Multimodal Understanding of Video Content

VRAgent-R1 (Chen et al., 2025) proposes an innovative multimodal understanding framework specifically for video recommendation scenarios. The system contains two core agents:

**Item Perception (IP) Agent**: Establishes comprehensive multimodal content understanding through multi-round, in-depth semantic interaction. Its workflow includes: (1) Key Frame Retrieval (KFR): Uses CLIP to compute visual-text similarity scores between sampled frames and video titles, retaining the top three highest-scoring frames as key visual information; (2) Collaborative Multimodal Perception (CMP): Inputs both retrieved frames and titles into MLLM to understand the semantic context implied by titles, providing relevant explanations and supplementary information; (3) Recommendation Relevant Analysis (RRA): Analyzes detailed video content, focusing on key information users are most likely to find interesting, reformulating video content into concise and precise descriptions.

**User Simulation (US) Agent**: Focuses on deep user behavior simulation, analyzing historical user behavior and comprehensively summarizing user status. The system designs two specific tasks: (1) User Preference Judgment: Given an item from the candidate list, determine whether the user would like the recommended video; (2) Next Video Selection: Review all candidate items and select the video the user is most likely to watch next.

VRAgent-R1's innovation lies in using Reinforcement Learning (RL) to fine-tune LLMs through the Group Relative Policy Optimization (GRPO) framework. This method directly compares groups of candidate responses without requiring a critic model to evaluate policy performance, enabling user simulation to better align with real decision-making processes.

### 2.3 Video Summarization and Key Content Extraction

360-VSumm (Kontostathis et al., 2024) provides important technical foundations for 360-degree video summarization. The dataset contains 40 2D videos, each with 15 human-generated summaries, providing benchmarks for training and evaluating video summarization methods. Its core technologies include:

**Salient Event Detection**: The system identifies salient regions in each frame by clustering salient points according to their intensity and distance. It then defines spatial-temporally-correlated 2D sub-volumes by grouping together spatially related regions across a sequence of frames.

**Multimodal Integration**: The system integrates multimodal data including visuals, audio, and metadata. Audio transcription uses the Whisper model to extract dialogue or narration, and frame captioning uses the LLaVA-NeXT 13B vision-language model to convert visual data into textual descriptions.

**Summary Generation**: Uses Claude 1.6 (with an extended context window of 200K tokens) to summarize all video captions and audio transcripts, generating video summaries and extracting keywords.

These technologies provide important inspiration for our system, particularly in how to effectively integrate multimodal information and how to extract key content from long videos.

## 3. Reinforcement Learning-Based Agent Optimization

### 3.1 Conceptual Verbal Reinforcement Learning

FinCon (Yu et al., 2024) proposes an innovative Conceptual Verbal Reinforcement (CVRF) mechanism for multi-agent system optimization in financial decision-making tasks. The system adopts a manager-analyst hierarchical structure with two key risk control components:

**Within-Episode Risk Control**: Monitors daily market risk through Conditional Value at Risk (CVaR). CVaR represents the average of the worst-performing 1% of daily trading Profits and Losses (PnLs). A sudden drop in CVaR triggers a risk alert, causing the manager agent to adopt a risk-averse stance.

**Over-Episode Risk Control**: Through an Actor-Critic mechanism, FinCon reflects on a series of successful and failed actions. The system evaluates the performance of consecutive training episodes, analyzes the information perspectives provided by analysts and reflected in manager decisions, then conceptualizes and attributes evaluation outcomes to these specific aspects.

CVRF's core innovation lies in using text-based gradient descent to provide optimal conceptual investment guidance. Unlike traditional optimizers based on model value gradient momentum, CVRF derives investment belief updates by measuring the overlapping percentage of trading actions between two consecutive training trajectories. This approach has proven effective in improving the performance of synthesized agent systems where each worker has a clearly defined and specialized role.

### 3.2 Application of Reinforcement Fine-Tuning in User Simulation

VRAgent-R1 pioneered the application of Reinforcement Fine-Tuning (RFT) to user simulation tasks. Compared to simple Supervised Fine-Tuning (SFT), the RFT method has the following advantages:

**Data Efficiency**: The RFT method requires less data to learn a reasoning strategy with good generalization, making it suitable for cold-start scenarios and user simulation. Experimental results show that training with information from 2000 users, the US Agent achieved 64.1% accuracy in user simulation, representing a 45.0% improvement over the SFT baseline.

**Deep Thinking Capability**: Unlike SFT which only memorizes answers, RFT enables the model to conduct in-depth analysis of user behavior. The system comprehensively summarizes user status (such as preferences and emotions), then analyzes candidate videos and executes appropriate actions.

**Reward Mechanism Design**: The system designs two types of rewards: (1) Format Reward (R_format): Ensures the model adheres to the required response format; (2) Judgment/Selection Reward (R_jud/R_sel): Allocates rewards based on whether the simulation is correct. This design enables the agent's user profile modeling to be dynamically updated based on task rewards, achieving learnable and more accurate simulation.

These methods provide important theoretical foundations for our system, particularly in how to optimize agent decision-making processes through reinforcement learning and how to design effective reward mechanisms to guide agent learning.

## Limitations of Existing Methods and Innovation of This Research

Despite significant progress in the above research, key challenges remain in agent simulation for short video recommendations:

1. **Lack of Objective Video Feature Quantification**: Existing methods primarily rely on subjective judgments from LLMs or simple metadata (such as genre tags), lacking standardized quantification methods for objective video features. Although VRAgent-R1 uses MLLM for multimodal understanding, its generated descriptions remain at the semantic level and are difficult to quantify into computable feature vectors.

2. **Insufficient Traceability of Agent Preferences**: In most systems, agent preferences and decision-making processes lack a clear mathematical foundation, making it difficult to trace and explain why agents make specific choices. Neural network-based recommendation methods, while performing well, suffer from black-box characteristics that make decision-making processes opaque.

3. **Missing Personalized Trait Mapping**: Existing datasets are insufficient to support user-created agents with different personalities, interests, and characteristics, lacking mechanisms to map user traits to video feature weights. Although Agent4Rec defines social traits, the relationship between these traits and specific video content features remains unclear.

4. **Unclear Prediction Path from Content Features to Watch Time**: While research has proven that watch time is an important engagement metric, there is still a lack of systematic research framework for how to predict watch time from measurable video features (such as shot duration, motion intensity, speech density, etc.) and how these features vary across users.

Our research aims to address these challenges by proposing a three-layer architecture: **Layer 1 uses automated tools to extract objective video features** (ASD, SCR, OFM, SD, CC), completely based on measurable video analysis methods without relying on subjective judgment; **Layer 2 learns personalized weight matrices from real user behavior data**, using watch time information from the MicroLens-100K dataset to establish the mapping relationship between user traits and feature weights; **Layer 3 predicts retention rate through linear weighted calculation**, achieving a 100% traceable decision-making process where every prediction result can be decomposed into the contribution of each feature.

The core advantages of this approach are: **fully automated**, requiring no large-scale manual annotation; **research-based**, with each feature grounded in published academic literature; **highly traceable**, with transparent and explainable decision-making processes; **experimentally feasible**, verifiable using existing public datasets.
