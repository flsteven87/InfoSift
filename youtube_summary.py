# Make the display a bit wider
# from IPython.display import display, HTML
# display(HTML("<style>.container { width:90% !important; }</style>"))

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

# Vector Store and retrievals
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
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
                  request_timeout = 180
                 )

class YoutubeSummary:
    def __init__(self):
        self.file_name = '財報狗_240'
        self.transcript_path = f'./transcript/{self.file_name}.wav.txt'
        self.highligt_path = f'./highlight/{self.file_name}.txt'

    def run_bullet_points_summary(self):
        with open(self.transcript_path) as file:
            transcript = file.read()
        print(transcript[:30])

        # Load up your text splitter
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=4000, chunk_overlap=800)
        docs = text_splitter.create_documents([transcript])
        print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

        # Map with several chunks
        system_message_prompt_map = SystemMessagePromptTemplate.from_template(map_template.bullet_points)

        human_template="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])
        # Reduce to final topics
        system_message_prompt_reduce = SystemMessagePromptTemplate.from_template(reduce_template.bullet_points_summary)

        human_template="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_reduce = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_reduce, human_message_prompt_reduce])


        chain = load_summarize_chain(llm4,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt_map,
                             combine_prompt=chat_prompt_combine,
                             verbose=True
                            )
        final_summary = chain.run({"input_documents": docs})
        print (final_summary)

        with open(f"./summary/{self.file_name}_bullet_points.txt", "w") as file:
            file.write(final_summary)


    def run_topics_summary(self):

        with open(self.transcript_path) as file:
            transcript = file.read()
        print(transcript[:30])

        # Load up your text splitter
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=4000, chunk_overlap=500)
        docs = text_splitter.create_documents([transcript])
        print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

        # Map with several chunks
        system_message_prompt_map = SystemMessagePromptTemplate.from_template(map_template.youtube_topic_bullet_points)

        human_template="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])
        # Reduce to final topics
        system_message_prompt_reduce = SystemMessagePromptTemplate.from_template(reduce_template.youtube_topic_bullet_points_summary)

        human_template="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_reduce = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_reduce, human_message_prompt_reduce])


        chain = load_summarize_chain(llm4,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt_map,
                             combine_prompt=chat_prompt_combine,
                             verbose=True
                            )
        final_summary = chain.run({"input_documents": docs})
        print (final_summary)

        with open(f"./summary/{self.file_name}_topics.txt", "w") as file:
            file.write(final_summary)

    def embedding(self):
        print('TBC')
        # embeddings = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'))
        # print(embeddings)
        # print(type(embeddings))
        # index_name = "statementdog-podcast-232"

        # pinecone.init(os.environ["PINECONE_API_KEY"], environment="asia-southeast1-gcp-free")
        # # pinecone.create_index("statementdog-podcast-232", dimension=1536, metric="euclidean")
        

        # with open(self.transcript_path) as file:
        #     transcript = file.read()

        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=800)
        # docs = text_splitter.create_documents([transcript])
        # print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

        # # index = pinecone.Index("statementdog_podcast_232")
        # # index.upsert(embeddings)
        # docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    def query(self):
        print('TBC')
        # index_name = 'statementdog-podcast-232'
        # index = pinecone.Index(index_name)
        # print(index.query("主題：美光科技的股票狀況", top_k=1, include_metadata=True))

    def evaluation(self):
        print('TBC')

if __name__ == '__main__':
    
    # YoutubeSummary().run_topics_summary()
    YoutubeSummary().run_bullet_points_summary()
    # YoutubeSummary().embedding()
    # YoutubeSummary().query()
    # YoutubeSummary().evaluation()