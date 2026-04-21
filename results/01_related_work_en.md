# Related Work

## 1. Audience Behavior Simulation Techniques

Recent breakthroughs in Large Language Models (LLMs) have introduced new paradigms for simulating user behavior. Agent4Rec (Zhang et al., 2024) pioneered the use of LLM-driven generative agents to simulate user behavior. Although Agent4Rec was originally developed for recommendation system simulation, its user behavior simulation techniques can be applied to audience reaction prediction. The system successfully simulates real user interactions by equipping each agent with profile, memory, and action modules. Specifically, Agent4Rec's agent architecture comprises three core features:

**Social Traits**: The system quantifies three dimensions of user characteristics: activity, conformity, and diversity. Activity measures the frequency and breadth of user interactions with recommended items; conformity examines how closely user ratings align with average item ratings; diversity reflects users' propensity toward diverse item categories.

**Memory Mechanism**: Agent4Rec distinguishes between factual memory and emotional memory. Factual memory encapsulates interactive behaviors within the recommendation system, while emotional memory captures psychological feelings stemming from these interactions, such as fatigue levels and overall satisfaction. This dual memory mechanism enables agents to react not only based on past factual interactions but also to consider feelings, thereby more closely mirroring real human behavior.

**Action Module**: The system designs two types of actions: taste-driven actions and emotion-driven actions. The former includes watching, rating, and generating post-viewing feelings; the latter involves decisions such as exiting the system and evaluating content lists, which are influenced by the agent's satisfaction with previously viewed items and current fatigue level.

While Agent4Rec was designed for recommendation scenarios, its core user behavior simulation techniques—particularly the social trait quantification (activity, conformity, diversity) and dual memory mechanism—provide valuable foundations for audience behavior prediction. However, when applying these techniques to short video audience prediction, we need to address a key limitation: Agent4Rec's preference modeling is based on historical rating data and lacks deep understanding of video content itself. For audience retention prediction, the appeal of short videos highly depends on multimodal features such as visual content, pacing, and emotional tension, rather than merely genre tags or historical ratings. Our system addresses this by combining Agent4Rec's behavior simulation framework with objective video feature extraction.

## 2. Video Feature Extraction for Retention Prediction

### 2.1 Research Foundation for Objective Video Features

In recent years, video analysis research has gradually shifted from subjective evaluation to objective feature quantification. **Watch Time** has been proven to be the most important user engagement metric for video content (arXiv:2508.11086, arXiv:2412.20211). Compared to traditional click-through rates or ratings, watch time more directly reflects users' genuine interest in content. **Retention Rate**, the completion rate of video viewing, has become a core metric for measuring content quality and predicting audience engagement.

In terms of video features, **Shot Duration** and **Editing Pace** are measurable objective features with solid research foundations. Research on the evolution of movie pacing shows that **Average Shot Duration (ASD)** has gradually decreased from approximately 12 seconds in the 1930s to 2-4 seconds in modern films, reflecting changes in audience attention patterns and the evolution of visual storytelling techniques. This trend is even more pronounced on short video platforms, where fast-paced editing has become a key factor in attracting and maintaining audience attention.

**Shot Change Rate (SCR)**, the number of shot transitions per minute, is closely related to visual stimulation and attention maintenance. Research shows that appropriate shot change frequency can effectively enhance audience engagement, but excessively high change rates may lead to visual fatigue.

**Motion Intensity**, quantified through Optical Flow analysis, measures visual changes between frames and reflects the dynamic nature of videos. Videos with high motion intensity typically better capture audience attention, particularly in action, sports, and gaming content.

**Speech Density** and **Content Complexity** are two other important objective features. Speech density measures the proportion of video containing speech, reflecting the intensity of information delivery. Content complexity is quantified through visual feature diversity, with high complexity indicating richer visual elements and scene variations.

The advantages of these objective features are: **fully automated extraction**, independent of manual annotation or subjective judgment; **clear computational methods**, implementable using open-source tools (such as PySceneDetect, OpenCV, Whisper, CLIP); **verifiable correlation with watch time**, providing a solid theoretical foundation for audience retention prediction models.

### 2.2 Multimodal Video Understanding

VRAgent-R1 (Chen et al., 2025) proposes an innovative multimodal understanding framework for video content analysis. Although originally designed for recommendation scenarios, its multimodal content understanding techniques are applicable to audience behavior prediction. The system contains two core agents:

**Item Perception (IP) Agent**: Establishes comprehensive multimodal content understanding through multi-round, in-depth semantic interaction. Its workflow includes: (1) Key Frame Retrieval (KFR): Uses CLIP to compute visual-text similarity scores between sampled frames and video titles, retaining the top three highest-scoring frames as key visual information; (2) Collaborative Multimodal Perception (CMP): Inputs both retrieved frames and titles into MLLM to understand the semantic context implied by titles, providing relevant explanations and supplementary information; (3) Recommendation Relevant Analysis (RRA): Analyzes detailed video content, focusing on key information users are most likely to find interesting, reformulating video content into concise and precise descriptions.

**User Simulation (US) Agent**: Focuses on deep user behavior simulation, analyzing historical user behavior and comprehensively summarizing user status. The system designs two specific tasks: (1) User Preference Judgment: Given an item from the candidate list, determine whether the user would like the video; (2) Next Video Selection: Review all candidate items and select the video the user is most likely to watch next.

VRAgent-R1's innovation lies in using Reinforcement Learning (RL) to fine-tune LLMs through the Group Relative Policy Optimization (GRPO) framework. This method directly compares groups of candidate responses without requiring a critic model to evaluate policy performance, enabling user simulation to better align with real decision-making processes. While VRAgent-R1 was developed for recommendation systems, its multimodal understanding and user simulation techniques provide valuable insights for audience behavior prediction in creator tools.

### 2.3 Video Summarization and Key Content Extraction

360-VSumm (Kontostathis et al., 2024) provides important technical foundations for 360-degree video summarization. The dataset contains 40 2D videos, each with 15 human-generated summaries, providing benchmarks for training and evaluating video summarization methods. Its core technologies include:

**Salient Event Detection**: The system identifies salient regions in each frame by clustering salient points according to their intensity and distance. It then defines spatial-temporally-correlated 2D sub-volumes by grouping together spatially related regions across a sequence of frames.

**Multimodal Integration**: The system integrates multimodal data including visuals, audio, and metadata. Audio transcription uses the Whisper model to extract dialogue or narration, and frame captioning uses the LLaVA-NeXT 13B vision-language model to convert visual data into textual descriptions.

**Summary Generation**: Uses Claude 1.6 (with an extended context window of 200K tokens) to summarize all video captions and audio transcripts, generating video summaries and extracting keywords.

These technologies provide important inspiration for our system, particularly in how to effectively integrate multimodal information and how to extract key content from long videos.

## 3. Personality Traits and Video Preference Research

### 3.1 Big Five Personality Model and Video Preferences

In recent years, psychological research has revealed deep connections between personality traits and media consumption behavior. The **Big Five personality model** (Five-Factor Model) is the most widely accepted personality framework in psychology, comprising five core dimensions: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. These traits have been proven to have significant correlations with users' video preferences.

Research shows that **users with high Openness** tend to explore diverse and innovative content, showing higher acceptance of visual complexity and novel narrative techniques. **Users with high Extraversion** prefer dynamic, socially-oriented content, demonstrating stronger engagement with videos featuring high motion intensity and dialogue-dense content. **Users with high Conscientiousness** tend toward structured, information-dense educational content, showing higher tolerance for videos with stable pacing. These findings provide a theoretical foundation for quantifying personality traits and mapping them to video feature weights.

### 3.2 Persona Datasets and Dialogue Modeling

The **PersonaChat dataset** (Zhang et al., 2018) is the first large-scale persona dialogue dataset, containing 8,000+ persona descriptions and corresponding dialogues. Each persona consists of 4-5 natural language sentences covering multiple dimensions including age, occupation, interests, and personality. This dataset provides an important foundation for training dialogue systems that maintain persona consistency, and also provides training data for extracting quantifiable traits from textual descriptions.

The **Big5-Chat dataset** (arXiv:2410.16491v1) further extends this direction, containing 100,000 dialogue samples based on Big Five personality traits. This dataset annotates each dialogue with the speaker's Big Five scores (0-100 scale), enabling us to train models to accurately extract personality traits from natural language descriptions. This data-driven trait extraction approach avoids the problem of relying purely on LLM subjective judgment, ensuring the reliability of persona quantification.

### 3.3 Persona Agent Evaluation Framework

**PersonaGym** (arXiv:2407.18416v4) proposes the first framework for dynamically evaluating persona agent consistency. The framework is based on decision theory and defines the **PersonaScore** metric to quantify the alignment between agent behavior and persona descriptions. PersonaGym evaluates four core dimensions:

**Preference Consistency**: Measures whether the agent's content choices align with the preferences described in its persona. Evaluation metrics include accuracy, precision, recall, and F1 score.

**Rating Consistency**: Compares the agent's rating distribution with real user rating patterns, using KL divergence and Jensen-Shannon divergence to quantify differences.

**Behavioral Consistency**: Measures whether the agent's watch time, exit behavior, etc., align with persona traits, using Mean Absolute Error (MAE) and Pearson correlation coefficient for evaluation.

**Trait Coherence**: Detects whether the agent maintains stable persona traits during long-term interactions, avoiding trait drift.

This evaluation framework provides us with objective, reproducible methods to validate persona agent accuracy and measure the gap between agent behavior and real user behavior.

### 3.4 Social Trait Quantification

As previously mentioned, Agent4Rec defines three quantifiable social traits that are closely related to video viewing behavior:

**Activity**: Quantified as the total number of videos watched by a user, reflecting the frequency of user interaction with video content. High-activity users watch more videos and have higher tolerance for content, while low-activity users are more selective and make exit decisions more quickly.

**Conformity**: Measured through the consistency between user ratings and average video ratings, with the formula `T_conformity(u) = (1/N) Σ |r_ui - R_i|²`. Users with low conformity have higher acceptance of niche content, while users with high conformity tend to follow mainstream preferences.

**Diversity**: Quantified as the total number of video genres watched by a user, reflecting the breadth of user interests. Users with high diversity have higher acceptance of different styles of content, while users with low diversity focus on specific genres.

These social traits complement Big Five personality traits, together forming a complete persona quantification framework. For example, high Extraversion is typically associated with high Activity, and high Openness is associated with high Diversity.

### 3.5 How Persona Quantification Extends the Objective Feature Approach

Our system integrates persona quantification into the existing objective feature extraction architecture rather than replacing it. Specifically:

**For users with viewing history** (Path A): The system continues to use MicroLens-100K data to directly learn personalized weights from watch time, which is the most accurate method because it is based on real behavioral data.

**For new users in cold-start scenarios** (Path B): The system uses the persona quantification approach: (1) Extract Big Five traits, social traits, and demographic features from the user's natural language description; (2) Convert these traits into video feature weights through a mapping function learned from MicroLens-100K; (3) Use these weights to predict retention rates.

The key advantage of this design is that **the mapping function is data-driven**, not LLM hallucinations. We cluster MicroLens-100K users by their learned weights, annotate trait patterns for each cluster, then train a regression model to learn the trait-to-weight mapping. This ensures that persona agent behavior is based on real user patterns, not imagined from thin air.

Furthermore, the persona quantification approach maintains the system's **100% traceability**. Each predicted retention rate can still be decomposed into the weighted contributions of five objective features (ASD, SCR, OFM, SD, CC), allowing users to clearly see why a particular persona would like or dislike a particular video. For example, a persona of "a 20-year-old college student who likes excitement and has no patience" would assign high weights to SCR (Shot Change Rate) and OFM (Optical Flow Magnitude), so fast-paced, dynamic videos would receive high retention rate predictions.

## 4. Reinforcement Learning-Based Agent Optimization

### 4.1 Conceptual Verbal Reinforcement Learning

FinCon (Yu et al., 2024) proposes an innovative Conceptual Verbal Reinforcement (CVRF) mechanism for multi-agent system optimization in financial decision-making tasks. The system adopts a manager-analyst hierarchical structure with two key risk control components:

**Within-Episode Risk Control**: Monitors daily market risk through Conditional Value at Risk (CVaR). CVaR represents the average of the worst-performing 1% of daily trading Profits and Losses (PnLs). A sudden drop in CVaR triggers a risk alert, causing the manager agent to adopt a risk-averse stance.

**Over-Episode Risk Control**: Through an Actor-Critic mechanism, FinCon reflects on a series of successful and failed actions. The system evaluates the performance of consecutive training episodes, analyzes the information perspectives provided by analysts and reflected in manager decisions, then conceptualizes and attributes evaluation outcomes to these specific aspects.

CVRF's core innovation lies in using text-based gradient descent to provide optimal conceptual investment guidance. Unlike traditional optimizers based on model value gradient momentum, CVRF derives investment belief updates by measuring the overlapping percentage of trading actions between two consecutive training trajectories. This approach has proven effective in improving the performance of synthesized agent systems where each worker has a clearly defined and specialized role.

### 4.2 Application of Reinforcement Fine-Tuning in User Simulation

VRAgent-R1 pioneered the application of Reinforcement Fine-Tuning (RFT) to user simulation tasks. Compared to simple Supervised Fine-Tuning (SFT), the RFT method has the following advantages:

**Data Efficiency**: The RFT method requires less data to learn a reasoning strategy with good generalization, making it suitable for cold-start scenarios and user simulation. Experimental results show that training with information from 2000 users, the US Agent achieved 64.1% accuracy in user simulation, representing a 45.0% improvement over the SFT baseline.

**Deep Thinking Capability**: Unlike SFT which only memorizes answers, RFT enables the model to conduct in-depth analysis of user behavior. The system comprehensively summarizes user status (such as preferences and emotions), then analyzes candidate videos and executes appropriate actions.

**Reward Mechanism Design**: The system designs two types of rewards: (1) Format Reward (R_format): Ensures the model adheres to the required response format; (2) Judgment/Selection Reward (R_jud/R_sel): Allocates rewards based on whether the simulation is correct. This design enables the agent's user profile modeling to be dynamically updated based on task rewards, achieving learnable and more accurate simulation.

These methods provide important theoretical foundations for our system, particularly in how to optimize agent decision-making processes through reinforcement learning and how to design effective reward mechanisms to guide agent learning.

## 5. Limitations of Existing Methods and Innovation of This Research

Despite significant progress in the above research, key challenges remain when applying these techniques to audience behavior prediction for creator tools:

1. **Lack of Objective Video Feature Quantification**: Existing behavior simulation methods primarily rely on subjective judgments from LLMs or simple metadata (such as genre tags), lacking standardized quantification methods for objective video features. Although VRAgent-R1 uses MLLM for multimodal understanding, its generated descriptions remain at the semantic level and are difficult to quantify into computable feature vectors.

2. **Insufficient Traceability of Agent Preferences**: In most systems, agent preferences and decision-making processes lack a clear mathematical foundation, making it difficult to trace and explain why agents make specific choices. Neural network-based methods, while performing well, suffer from black-box characteristics that make decision-making processes opaque.

3. **Missing Personalized Trait Mapping**: Existing datasets are insufficient to support creator-defined audience personas with different personalities, interests, and characteristics, lacking mechanisms to map audience traits to video feature weights. Although Agent4Rec defines social traits, the relationship between these traits and specific video content features remains unclear.

4. **Unclear Prediction Path from Content Features to Watch Time**: While research has proven that watch time is an important engagement metric, there is still a lack of systematic research framework for how to predict watch time from measurable video features (such as shot duration, motion intensity, speech density, etc.) and how these features vary across different audience segments.

Our research aims to address these challenges by proposing a three-layer architecture: **Layer 1 uses automated tools to extract objective video features** (ASD, SCR, OFM, SD, CC), completely based on measurable video analysis methods without relying on subjective judgment; **Layer 2 learns personalized weight matrices from real user behavior data**, using watch time information from the MicroLens-100K dataset to establish the mapping relationship between audience traits and feature weights; **Layer 3 predicts retention rate through linear weighted calculation**, achieving a 100% traceable decision-making process where every prediction result can be decomposed into the contribution of each feature.

The core advantages of this approach are: **fully automated**, requiring no large-scale manual annotation; **research-based**, with each feature grounded in published academic literature; **highly traceable**, with transparent and explainable decision-making processes; **experimentally feasible**, verifiable using existing public datasets. By adapting user behavior simulation techniques from Agent4Rec and VRAgent-R1 to the context of audience retention prediction, we create a system that serves content creators rather than recommendation platforms.
