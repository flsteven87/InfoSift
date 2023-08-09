content_to_highlights_user_prompt ="""You are a professional editor who is also proficient in Mandarin. 
Provide bullet points for all the contents of the following articles. 
Each bullet point should base on actual data, without exaggeration. 
Please list as many points as necessary in traditional Chinese characters.\n\n
Please provide specific numbers, dates, and names of individuals. 
Ensure that each bullet point is a self-contained sentence and does not show the linking words to the previous sentence. 
A period at the end of a sentence.\n\nRelative times such as "Tuesday" or "yesterday" should be changed to exact dates. 
Names of individuals should be presented in English, and clearly identify who is speaking in every sentence if you mention the third person, and standing for what position and company. Fiscal quarters, such as "2023 fiscal year Q1," should be changed to "2023 Q1". All numbers or English letters are adjacent to Chinese characters must be separated by a space, for example, 「 Nike 於 3 月 20 日 發布…」\n\n---\n\nexample:\ntoday is 2023-04-19\n"Tuesday" ->  "4 月 17 日"\n"上週三" ->  "4 月 12 日"\n\n---\n\noutput format：\n\n- \n-\n\n---"""
cluster_prompt = """完全保留內容，將內容分成 4-6 群，句子不做更動。\n\n請一步一步來，首先，理解全部內容有哪些。接著，決定要分成幾群。接著，將內容擺放到對應的群組中。最後，回傳結果\n\n---\n\n"""
podcast_summary_prompt = """首先，針對每個群組下一個標題，不要出現群組和編號。\n\n接著，將每一個群組內的內容各別重新整理，每個群組最多三個 bullet point。每個點用一句話說明，不要出現重複。\n\n請一步一步來\n\n---\n\nformat:\n\n## {標題}\n- \n\n"""
title_prompt = """根據以下內容，寫 podcast 標題跟描述，用用親切、溫暖的第一人稱口吻，省略「我們」、「這集」等辭彙\n\n---\n\n標題：\n描述：\n\n---"""
