# Supporting libraries
import os
from dotenv import load_dotenv
load_dotenv()

# OpenAI
import openai
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
from prompt.bullet_points import version2

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
# from langchain.vectorstores import Pinecone
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

from prompt.topics import map_template
from prompt.topics import reduce_template


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
                #   max_tokens=3000
                 )

class YoutubeSummary:
    def __init__(self, video_id):
        self.video_id = video_id
        self.transcript_file_path = f'./transcript/txt/{self.video_id}.wav.txt'
        self.transcript = self.load_transcript()
        # self.topics_structured = []

    def load_transcript(self):
        with open(self.transcript_file_path, 'r') as f:
            transcript = f.read()
        return transcript

    def evaluate_transcript(self):

        transcript = self.load_transcript()
        print(transcript[:30])
        print(f"Transcript total tokens: {llm3.get_num_tokens(transcript)}")

        # Load up your text splitter
        # text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=4000, chunk_overlap=500)
        # docs = text_splitter.create_documents([transcript])
        # print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

    def gen_bullet_points(self):

        bullet_points = []

        with open(self.transcript_path) as file:
            transcript = file.read()
        print(transcript[:30])

        # Load up your text splitter
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=4000, chunk_overlap=400)
        docs = text_splitter.create_documents([transcript])
        print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

        for doc in docs:
            # print(len(doc.page_content))
            # print(doc.page_content[:30])

            messages = [
                {"role": "system", "content": version2.system},
                # {"role": "assistant", "content": doc.page_content},
                {"role": "user", "content": version2.user + '\n\n' + f'transcript: {doc.page_content}'},
            ]

            # print(messages)

            response = openai.ChatCompletion.create(
                model="gpt-4-0613",
                messages=messages,
                temperature=0
            )

            bullet_points.append(response['choices'][0]['message']['content'])
            print(f"Prompt Token: {response['usage']['prompt_tokens']}.  Completion Tokens: {response['usage']['completion_tokens']}. Total tokens: {response['usage']['total_tokens']}")

        bullet_points = "\n".join(bullet_points)
        paragraphs = bullet_points.split("\n")
        bullet_points = [p for p in paragraphs if p.startswith("-")]

        # Export to txt
        with open(f'./bullet_points/{self.file_name}_version2.txt', 'w') as f:
            for item in bullet_points:
                f.write("%s\n" % item)

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

    def gen_topics(self):

        print(self.transcript[0:30])

        # Load up your text splitter
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.create_documents([self.transcript])
        print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

        # Map with several chunks
        system_message_prompt_map = SystemMessagePromptTemplate.from_template(map_template.topics_map)

        human_template="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_map = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_map = ChatPromptTemplate.from_messages(messages=[system_message_prompt_map, human_message_prompt_map])
        # Reduce to final topics
        system_message_prompt_reduce = SystemMessagePromptTemplate.from_template(reduce_template.topics_reduce)

        human_template="Transcript: {text}" # Simply just pass the text as a human message
        human_message_prompt_reduce = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt_combine = ChatPromptTemplate.from_messages(messages=[system_message_prompt_reduce, human_message_prompt_reduce])

        chain = load_summarize_chain(llm3,
                             chain_type="map_reduce",
                             map_prompt=chat_prompt_map,
                             combine_prompt=chat_prompt_combine,
                             verbose=True
                            )
        topics = chain.run({"input_documents": docs})
        print (topics)

        with open(f"./topics/{self.video_id}.txt", "w") as file:
            file.write(topics)

    def functioning_api(self):

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

        with open(f'./topics/{self.video_id}.txt') as file:
            self.topics = file.read()

        self.topics_structured = chain.run(self.topics)
        topics_structured = ' '.join(str(item) for item in self.topics_structured)

        with open(f"./topics/{self.video_id}_structured.txt", "w") as file:
            file.write(topics_structured)

    def embedding(self):
                
        transcript = self.transcript

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
        docs = text_splitter.create_documents([transcript])
        print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

        persist_directory = f'./chroma/'
        embedding = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'))

        vectordb = Chroma.from_documents(
            documents = docs,
            embedding = embedding,
            persist_directory = persist_directory
        )

        vectordb.persist()
        print(vectordb._collection.count())

        return vectordb

    def retriever(self):

        persist_directory = f'./chroma/'
        embedding = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY', 'YourAPIKeyIfNotSet'))

        vectordb = Chroma(
            embedding_function = embedding,
            persist_directory = persist_directory
        )        
        print(vectordb._collection.count())

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

        # Only doing the first 3 for conciseness 
        # for topic in self.topics_structured[:3]:
        #     query = f"""
        #         {topic['topic_name']}: {topic['description']}
        #     """

        #     expanded_topic = qa.run(query)

        #     print(f"{topic['topic_name']}: {topic['description']}")
        #     print(expanded_topic)
        #     with open(f"./topics/{self.file_name}/{topic['topic_name']}_summary.txt", "w") as file:
        #         file.write(expanded_topic)
        #     print ("\n\n")

        query = "美光最新一度營運報告分析: 美光在6月29號公佈了最新一季的營運報告，營收成長了1.6%，高於財測中位數。"
        expanded_topic = qa.run(query)

        #     print(f"{topic['topic_name']}: {topic['description']}")
        print(expanded_topic)
        with open(f"./topics/{self.file_name}/summary.txt", "w") as file:
            file.write(expanded_topic)
        print ("\n\n")


if __name__ == '__main__':

    # YoutubeSummary().evaluate_transcript()    
    # YoutubeSummary().gen_bullet_points()
    YoutubeSummary(video_id).gen_topics()
    # YoutubeSummary().functioning_api()
    # YoutubeSummary().run_bullet_points_summary()
    # vectordb = YoutubeSummary().embedding()
    # YoutubeSummary().retriever()
    # YoutubeSummary().query()
