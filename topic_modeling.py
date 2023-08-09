#%%
# Make the display a bit wider
# from IPython.display import display, HTML
# display(HTML("<style>.container { width:90% !important; }</style>"))

# LangChain basics
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import create_extraction_chain

# Vector Store and retrievals
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone

# Chat Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# Supporting libraries
import os
from dotenv import load_dotenv

load_dotenv()
# %%
# LangChain Plus
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_SESSION"] = "InfoSift"

# %%
# Creating two versions of the model so I can swap between gpt3.5 and gpt4
llm3 = ChatOpenAI(temperature=0,
                  openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'),
                  model_name="gpt-3.5-turbo-16k-0613",
                  request_timeout = 180
                )

llm4 = ChatOpenAI(temperature=0,
                  openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'),
                  model_name="gpt-4-0613",
                  request_timeout = 180
                 )

with open('./transcript/財報狗_podcast__232.wav.txt') as file:
    transcript = file.read()
# %%
print(transcript[:100])
# %%
# Load up your text splitter
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=4000, chunk_overlap=500)

docs = text_splitter.create_documents([transcript])
print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")
# %%
from prompt import map_template 
from prompt import reduce_template

# %%
# Map with several chunks
system_message_prompt_map = SystemMessagePromptTemplate.from_template(map_template.youtube_topic_bullet_points)

human_template="Transcript: {text}" # Simply just pass the text as a human message
human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])
# %%
# Reduce to final topics
system_message_prompt_reduce = SystemMessagePromptTemplate.from_template(reduce_template.youtube_topic_summary)

human_template="Transcript: {text}" # Simply just pass the text as a human message
human_message_prompt_reduce = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_reduce, human_message_prompt_map])
# %%
chain = load_summarize_chain(llm4,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt_map,
                             combine_prompt=chat_prompt_combine,
                             verbose=True,
                             return_intermediate_steps=True
                            )
# %%
chain({"input_documents": docs}, return_only_outputs=True)

# %%
with open('./topics/232.txt') as file:
    topic_found = file.read()
# %%
print(topic_found)


# %%
# Structured Data
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
        },
    },
    "required": ["topic", "description"],
}

# Using gpt3.5 here because this is an easy extraction task and no need to jump to gpt4
chain = create_extraction_chain(schema, llm3)
topics_structured = chain.run(topic_found)
topics_structured
# %%
# Embedding & Expand on the topics you found
text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
docs = text_splitter.create_documents([transcript])
print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")
# %%
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'))

#%%
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

#%%
# initialize pinecone
pinecone.init(os.environ["PINECONE_API_KEY"], environment="asia-southeast1-gcp-free")
index_name = "statementdog-podcast-232"
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
#%%
# I'm using gpt4 for the increased reasoning power.
# I'm also setting k=4 so the number of relevant docs we get back is 4. This parameter should be tuned to your use case
qa = RetrievalQA.from_chain_type(llm=llm4,
                                 chain_type="stuff",
                                 retriever=docsearch.as_retriever(k=4),
                                 chain_type_kwargs = {
#                                      'verbose': True,
                                     'prompt': CHAT_PROMPT
                                 })

# %%
# Only doing the first 3 for conciseness 
for topic in topics_structured[1:3]:
    query = f"""
        {topic['topic_name']}: {topic['description']}
    """

    print(query)

    expanded_topic = qa.run(query)

    print(f"{topic['topic_name']}: {topic['description']}")
    print(expanded_topic)
    print ("\n\n")
