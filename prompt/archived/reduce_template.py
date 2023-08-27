bullet_points_summary='''
Maintain the content entirely, without altering the sentences.
Please proceed step by step. First, understand all the contents. Then, decide how many groups you want to divide it into. 
Next, place the content in the corresponding groups.
For each group, give a title, and provide a description for the title. Do not include the group name or number in the description.
Please do it by traditional Chinese characters.

Finally, reorganize the content within each group individually, with a maximum of 6 bullet points per group. Describe each point with a sentence, and avoid repetition.
% START OF EXAMPLES
'Topic title: Brief Description within 3 sentence'
 - I am bullet point 1
 - I am bullet point 2
% END OF EXAMPLES
'''

youtube_topic_bullet_points_summary = '''
- You are a professional finance editor who is also proficient in Mandarin that helps summarize an investment podcast transcript
- You will be given a series of topics and their bullet points.
- Your goal is to give a final summary of each topic and also select 3-7 bullet points.
- First, understand all the contents. Then, decide how many topics you want to divide it into. 
- Deduplicate any topics or bullet points with same meaning and information.
- Maintain the bullet points entirely, without altering the sentences.
- Only pull topics from the transcript.
- Please use traditional Chinese characters.
- The format should including a topic title, topic summary, and topic bullet points.

% START OF EXAMPLES
'Topic title: Brief summary within 3 sentence'
 - I am bullet point 1
 - I am bullet point 2
% END OF EXAMPLES
'''


cluster_prompt='''
完全保留內容，將內容分成 4-6 群，句子不做更動。\n\n請一步一步來，首先，理解全部內容有哪些。接著，決定要分成幾群。接著，將內容擺放到對應的群組中。最後，回傳結果\n\n---\n\n
'''

summary_prompt='''
針對每個群組下一個標題，不要出現群組和編號。\n\n接著，將每一個群組內的內容各別重新整理，每個群組最多四個 bullet point。每個點用一句話說明，不要出現重複。
'''