system='''
You are a professional editor who is also proficient in Mandarin.
'''

user='''
你的目標是以我所提供的 transcript 產出一個有條理架構的 Summary
請一步一步來，
第一步，理解全部內容有哪些。
第二步，決定要分成幾群。
第三部，為每個群組下15個字以內標題，不要出現群組和編號，並且提供至多3個 bullet points並符合下方的規範:
- Ensure that each bullet point is a self-contained sentence and does not show the linking words to the previous sentence. 
- A period at the end of a sentence.
- Relative times such as "Tuesday" or "yesterday" should be changed to exact dates. 
- Names of individuals should be presented in English, and clearly identify who is speaking in every sentence if you mention the third person, and standing for what position and company. 
- Fiscal quarters, such as "2023 fiscal year Q1," should be changed to "2023 Q1". 
- All numbers or English letters are adjacent to Chinese characters must be separated by a space, for example, 「 Nike 於 3 月 20 日 發布…」\n\n---\n\nexample:\ntoday is 2023-04-19\n"Tuesday" ->  "4 月 17 日"\n"上週三" ->  "4 月 12 日"\n\n---\n\noutput format：\n\n- \n-\n\n---
第四部，檢視你所有的群組及其 bullet points，給一段不超過30個字的標題。
第五部，針對整份內容，撰寫不超過 70 個字的描述。
最後，下方是你輸出內容應該參考的格式，請使用相同的格式，但不要使用裡頭的內容:
===Start of example===
標題：探索旅行箱與科技產業：Logel、ADVENTEC與投資建議
描述：在這一集中，我們將深入探討Logel旅行箱的特色與優勢，並帶你認識工業電腦龍頭廠ADVENTEC研華科技。我們也會分析工業電腦（IPC）產業與鴻海集團的現況，並提供實用的投資建議。最後，我們會分享一些個人觀點與建議，包括對於「啃老」這個議題的看法。希望這一集的內容能讓你有所收穫，一起來聆聽吧！

## Logel 旅行箱介紹
- Logel 旅行箱具有前開擴充式設計，平頂式的箱蓋方便開啟，並配有雙層防盜拉鍊和 TSA 海關鎖提供安全保障。
- 旅行箱附有拉鍊擴充圈提供更多儲物空間，並配有靜音雙輪具輪胎，推動更安靜、滑順。
- Logel 旅行箱提供多種顏色和尺寸選擇，專櫃價為 8980 起，國外聽眾的優惠價為 6645 起，優惠至 8 月 28 日。

## ADVENTEC 研華科技介紹
- ADVENTEC 研華科技是工業電腦的龍頭廠，品牌在全球的市佔大概是有四成以上左右。
- ADVENTEC 研華科技的新園區設計給人大學城的感覺，有大量的書籍供員工閱讀，並有大的廣場和草皮開放給附近的住家作為公園使用。
- ADVENTEC 研華科技的產品線繁多，包括公控小電腦、學校研究機構單位醫院的設備等，其中許多產品都有 Parallel Computing GPU 運算的晶片。
===End of example===
請你輸出最終結果即可，不要輸出每一步的局部結果
'''

