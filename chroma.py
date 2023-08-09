#%%
# Supporting libraries
import os
from dotenv import load_dotenv

load_dotenv()

# LangChain Plus
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_SESSION"] = "InfoSift"

# LangChain basics
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain
from langchain.callbacks import get_openai_callback
from langchain.chains import RetrievalQAWithSourcesChain

# Vector Store and retrievals
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.vectorstores import Chroma
# import pinecone

from prompt import map_template 
from prompt import reduce_template

# Chat Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from prompt import map_template 
from prompt import reduce_template

# Creating two versions of the model so I can swap between gpt3.5 and gpt4
llm3 = ChatOpenAI(temperature=0,
                  openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'),
                  model_name="gpt-3.5-turbo-16k-0613",
                  request_timeout = 180
                )

llm4 = ChatOpenAI(temperature=0,
                  openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'),
                  model_name="gpt-4-0613",
                #   model_name="gpt-4-32k-0613",
                  request_timeout = 180,
                  max_tokens=3000
                 )

#%%
transcript_path = f'./transcript/財報狗_240.wav.txt'

with open(transcript_path) as file:
    transcript = file.read()

print(transcript[:30])

# %%
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# %%
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,
    chunk_overlap=500
)
# %%
embedding = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'))
persist_directory = f'./chroma/'
vectordb = Chroma(
    embedding_function = embedding,
    persist_directory = persist_directory
)        
print(vectordb._collection.count())
# %%
# The system instructions. Notice the 'context' placeholder down below. This is where our relevant docs will go.
# The 'question' in the human message below won't be a question per se, but rather a topic we want to get relevant information on
system_template = """
You will be given text from a podcast transcript which contains many topics.
You goal is to write a summary (5 sentences or less) about a topic the user chooses
Do not respond with information that isn't relevant to the topic that the user gives you
----------------
{context}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]

# This will pull the two messages together and get them ready to be sent to the LLM through the retriever
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

# I'm using gpt4 for the increased reasoning power.
# I'm also setting k=4 so the number of relevant docs we get back is 4. This parameter should be tuned to your use case
qa = RetrievalQA.from_chain_type(llm=llm4,
                                chain_type="stuff",
                                retriever=vectordb.as_retriever(k=1),
                                chain_type_kwargs = {
#                                      'verbose': True,
                                    'prompt': CHAT_PROMPT
                                })
# %%
query = "美光最新一度營運報告分析: 美光在6月29號公佈了最新一季的營運報告，營收成長了1.6%，高於財測中位數。"
expanded_topic = qa.run(query)

#     print(f"{topic['topic_name']}: {topic['description']}")
print(expanded_topic)
with open(f"./topics/{self.file_name}/summary.txt", "w") as file:
    file.write(expanded_topic)
print ("\n\n")

#%%
query = "美光最新一度營運報告分析: 美光在6月29號公佈了最新一季的營運報告，營收成長了1.6%，高於財測中位數。"
docs = vectordb.similarity_search(query,k=2)
for doc in docs:    
    print(doc.page_content[:30])
# %%
docs = vectordb.max_marginal_relevance_search(query,k=2,fetch_k=3)
# %%
for doc in docs:    
    print(doc.page_content[:30])

# %%
print(docs[0].page_content)
# %%
import openai
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

user_prompt = '''
You will be given text from a podcast transcript which contains many topics.
Step1: You goal is to write a summary (5 sentences or less) about a topic the user chooses
Step2: List all key information as bullet point under the summary
- Each bullet point should base on actual data, without exaggeration. 
- Please provide specific numbers, dates, and names of individuals. 
- Ensure that each bullet point is a self-contained sentence and does not show the linking words to the previous sentence. 
- A period at the end of a sentence. Relative times such as "Tuesday" or "yesterday" should be changed to exact dates. 
- Names of individuals should be presented in English, and clearly identify who is speaking in every sentence if you mention the third person, and standing for what position and company. Fiscal quarters, such as "2023 fiscal year Q1," should be changed to "2023 Q1". 
- All numbers or English letters are adjacent to Chinese characters must be separated by a space, for example, 「 Nike 於 3 月 20 日 發布…」\n\n---\n\nexample:\ntoday is 2023-04-19\n"Tuesday" ->  "4 月 17 日"\n"上週三" ->  "4 月 12 日"\n\n---\n\noutput format：\n\n- \n-\n\n---
- Please use traditional Chinese characters.
- The format should including a topic title, topic summary, and topic bullet points.

% START OF EXAMPLES
'Topic title: Brief summary within 10 sentence'
 - I am bullet point 1
 - I am bullet point 2
% END OF EXAMPLES

Do not respond with information that isn't relevant to the topic that the user gives you
'''

messages = [
    {"role": "system", "content": 'You are a professional financial editor who is also proficient in Mandarin.'},
    # {"role": "assistant", "content": doc.page_content},
    {"role": "user", "content": f'{user_prompt}' + '\n\n' + f'transcript: {docs[0].page_content}'},
]

# print(messages)

response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=messages,
    temperature=0
)

print(response['choices'][0]['message']['content'])
print(f"Prompt Token: {response['usage']['prompt_tokens']}.  Completion Tokens: {response['usage']['completion_tokens']}. Total tokens: {response['usage']['total_tokens']}")

# %%
schema = {
    "properties": {
        # The title of the topic
        "topic_name": {
            "type": "string",
            "description" : "The title of the topic listed"
        },
        # The description
        "description": {
            "type": "string",
            "description" : "The description of the topic listed"
        }
    },
    "required": ["topic", "description"],
}

# Using gpt3.5 here because this is an easy extraction task and no need to jump to gpt4
chain = create_extraction_chain(schema, llm3)

with open(f'./topics/財報狗_240.txt') as file:
    topics = file.read()

print(topics)
topics_structured = chain.run(topics)
print(topics_structured)
# %%
summary = ''
total_tokens = 0
for topic in topics_structured:
    query = f"""
        {topic['topic_name']}: {topic['description']}
    """

    docs = vectordb.max_marginal_relevance_search(query,k=2,fetch_k=3)
    print(f"{topic['topic_name']}: {topic['description']}")

    user_prompt = '''
    You will be given text from a podcast transcript which contains many topics.
    Step1: Understand all the transcript. and your goal is to write a summary (10 sentences or less) about a topic the user chooses
    Step2: List all key information as bullet point under the summary
    - Each bullet point should base on actual data, without exaggeration. 
    - Please provide specific numbers, dates, and names of individuals. 
    - Ensure that each bullet point is a self-contained sentence and does not show the linking words to the previous sentence. 
    - A period at the end of a sentence. Relative times such as "Tuesday" or "yesterday" should be changed to exact dates. 
    - Names of individuals should be presented in English, and clearly identify who is speaking in every sentence if you mention the third person, and standing for what position and company. Fiscal quarters, such as "2023 fiscal year Q1," should be changed to "2023 Q1". 
    - All numbers or English letters are adjacent to Chinese characters must be separated by a space, for example, 「 Nike 於 3 月 20 日 發布…」\n\n---\n\nexample:\ntoday is 2023-04-19\n"Tuesday" ->  "4 月 17 日"\n"上週三" ->  "4 月 12 日"\n\n---\n\noutput format：\n\n- \n-\n\n---
    - Please use traditional Chinese characters.
    - The format should including a topic title, topic summary, and topic bullet points.

    % START OF EXAMPLES
    'Topic title: Brief summary within 10 sentences'
    - I am bullet point 1
    - I am bullet point 2
    % END OF EXAMPLES

    Do not respond with information that isn't relevant to the topic that the user gives you
    '''

    messages = [
        {"role": "system", "content": 'You are a professional financial editor who is also proficient in Mandarin.'},
        # {"role": "assistant", "content": doc.page_content},
        {"role": "user", "content": f'{user_prompt}' + '\n\n' + f'transcript: {docs[0].page_content}'},
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        temperature=0
    )

    topic_summary = response['choices'][0]['message']['content']
    print(topic_summary)
    summary = summary + '\n\n------------------\n\n' + topic_summary 
    print(f"Prompt Token: {response['usage']['prompt_tokens']}.  Completion Tokens: {response['usage']['completion_tokens']}. Total tokens: {response['usage']['total_tokens']}")
    total_tokens = total_tokens + response['usage']['total_tokens']

summary = summary + '/n/n------------------/n/n' + f'total_tokens: {total_tokens}'
with open(f"./topics/財報狗_240_summary.txt", "w") as file:
    file.write(summary)

# %%
