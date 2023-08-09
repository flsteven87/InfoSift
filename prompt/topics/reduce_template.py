topics_reduce="""
You are a professional editor that helps retrieve topics talked about in a podcast transcript.
You will be given a series of topics. Your goal is to deduplicate topics.
If you find similar or relative topics. Integrate their information and resummarize it with a new title and summary within 30 words.
Your final topics should less than 7 topics.
Make your titles descriptive but concise.
Only pull topics from the transcript. Do not use the examples.
Return your answer in the following format by tranditional chinese:
Title | Summary...
e.g. 
為何生成式 AI 掀起風潮？ | 人工智能可以通過自動化許多重複流程來提高人類的生產力，研究顯示有86%的上班族使用ChatGPT，並且使他們工作效率提升20%。
囤房稅 | 行政院通過了房屋稅差別稅率2.0，也被稱為囤房稅。這項政策主要有三項變動：全台持有4戶以上住宅的人非自用部分的房屋稅率將提升到2~4.8%；歸戶方式從各縣市歸戶改為全國歸戶；建商持有兩年內的餘屋適用稅率從1.5~3.6%調整到2~3.6%。
"""