# Supporting libraries
import os
from dotenv import load_dotenv
load_dotenv()

# OpenAI
import openai
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')
# from prompt.bullet_points import version2

# Get token counter
import tiktoken
# Get the encoding for a specific model
enc = tiktoken.get_encoding("cl100k_base")

# LangChain Plus
import os
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
# os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGCHAIN_API_KEY')
# os.environ["LANGCHAIN_SESSION"] = "InfoSift"

# LangChain basics
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
# from langchain.chains import create_extraction_chain
# from langchain.callbacks import get_openai_callback

# Vector Store and retrievals
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
# from langchain.vectorstores import Pinecone
from langchain.vectorstores import Chroma
# import pinecone

# from prompt import map_template 
# from prompt import reduce_template

# Summary evluation
# import gensim
# from gensim import corpora
# from gensim.models import LdaModel
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords

# BERTopics evaluate summary quality
from bertopic import BERTopic
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# LDA evaluate summary quality
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Chat Prompt templates for dynamic values
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from prompt.topics import version1
from prompt.bullet_points import version2
from prompt.single_summary import version1

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
        self.summary_file_path = f'./bullet_points/{self.video_id}_version2_summary.txt'
        self.transcript = self.load_transcript()
        # self.topics_structured = []

    def load_transcript(self):
        with open(self.transcript_file_path, 'r') as f:
            transcript = f.read()
        return transcript

    def evaluate_transcript(self):

        transcript = self.load_transcript()
        # print(transcript[:30])
        print(f"Transcript total tokens: {len(enc.encode(transcript))}")

        return len(enc.encode(transcript))

        # Load up your text splitter
        # text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=4000, chunk_overlap=500)
        # docs = text_splitter.create_documents([transcript])
        # print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

    def evaluate_bulle_points(self):
        with open(f'./bullet_points/{self.video_id}_mp_v2_bullet_points.txt', 'r') as f:
            bullet_points = f.read()
        print(f"Bullet Points total tokens: {len(enc.encode(bullet_points))}")

        return len(enc.encode(bullet_points))
        
    def gen_topics(self):

        print(self.transcript[0:30])

        # Load up your text splitter
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=2000, chunk_overlap=200)
        docs = text_splitter.create_documents([self.transcript])
        print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

        # Map with several chunks
        system_message_prompt_map = SystemMessagePromptTemplate.from_template(version1.topics_map)

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
            # metadata = self.video_id
        )

        vectordb.persist()
        print(vectordb._collection.count())

        # return vectordb

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


    def bullet_points(self):

        bullet_points = []

        transcript = self.load_transcript()

        # Load up your text splitter
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=4000, chunk_overlap=200)
        docs = text_splitter.create_documents([transcript])
        print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

        print("Generating Bullet Points...")
        for doc in docs:
            # print(len(doc.page_content))
            # print(doc.page_content[:30])

            messages = [
                {"role": "system", "content": version2.map_system},
                # {"role": "assistant", "content": doc.page_content},
                {"role": "user", "content": version2.map_user + '\n\n' + f'transcript: {doc.page_content}'},
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
        with open(f'./bullet_points/{self.video_id}_mp_v2_bullet_points.txt', 'w') as f:
            for item in bullet_points:
                f.write("%s\n" % item)       


    def cluster(self):

        cluster = ''
        with open(f'./bullet_points/{self.video_id}_mp_v2_bullet_points.txt', 'r') as f:
            bullet_points = f.read()

        print("Clustering...")
        messages = [
            {"role": "system", "content": version2.map_system},
            # {"role": "assistant", "content": doc.page_content},
            {"role": "user", "content": version2.cluster_prompt + '\n\n' + f'bullet points: {bullet_points}'},
        ]

        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=messages,
            temperature=0
        )

        cluster = response['choices'][0]['message']['content']
        print(f"Prompt Token: {response['usage']['prompt_tokens']}.  Completion Tokens: {response['usage']['completion_tokens']}. Total tokens: {response['usage']['total_tokens']}")

        # Export to txt
        with open(f'./bullet_points/{self.video_id}_mp_v2_cluster.txt', 'w') as f:
            f.write(cluster)       


    def title_and_summary(self):
        
        with open(f'./bullet_points/{self.video_id}_mp_v2_cluster.txt', 'r') as f:
            cluster = f.read()

        print("Summary & Title")

        messages = [
            {"role": "system", "content": version2.map_system},
            # {"role": "assistant", "content": doc.page_content},
            {"role": "user", "content": version2.podcast_summary_prompt + '\n\n' + f'cluster: {cluster}'},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=messages,
            temperature=0
        )

        summary = response['choices'][0]['message']['content']
        print(f"Prompt Token: {response['usage']['prompt_tokens']}.  Completion Tokens: {response['usage']['completion_tokens']}. Total tokens: {response['usage']['total_tokens']}")

        messages = [
            {"role": "system", "content": version2.map_system},
            # {"role": "assistant", "content": doc.page_content},
            {"role": "user", "content": version2.title_prompt + '\n\n' + f'summary: {summary}' +'\n\n'+ f'cluster: {cluster}'},
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=messages,
            temperature=0
        )

        title = response['choices'][0]['message']['content']
        print(f"Prompt Token: {response['usage']['prompt_tokens']}.  Completion Tokens: {response['usage']['completion_tokens']}. Total tokens: {response['usage']['total_tokens']}")

        # Export to txt
        with open(f'./summary/{self.video_id}_mp_v2_summary.txt', 'w') as f:
            f.write(title + '\n\n' + summary)       


    def bertopic(self):

        with open(self.summary_file_path, 'r') as f:
            self.summary = f.read()

        # print(self.transcript[0:20])
        # print(self.summary[0:20])

        if not self.transcript.strip() or not self.summary.strip():
            raise ValueError("Transcript or Summary is empty or only contains whitespace characters.")
        
        # transcript_sentences = sent_tokenize(self.transcript)
        transcript_sentences = self.transcript.split('\n')

        # summary_sentences = sent_tokenize(self.summary)
        summary_sentences = self.summary.split('\n')

        # # 使用 chunks 切    
        # text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "], chunk_size=4000, chunk_overlap=500)
        # docs = text_splitter.create_documents([self.transcript])
        # print (f"You have {len(docs)} docs. First doc is {llm3.get_num_tokens(docs[0].page_content)} tokens")

        # transcript_sentences = []
        # for doc in docs:
        #     transcript_sentences.append(doc.page_content)

        # print(len(transcript_sentences))
        # print(len(summary_sentences))

        print(f"BERTopic Evaluating Summary of {self.video_id}...")
        # 使用BERTopic進行主題建模
        topic_model = BERTopic()
        
        similarity = []
        for i in range(0,10):

            # 對transcript的句子進行主題建模
            transcript_topics, _ = topic_model.fit_transform(transcript_sentences)
            # print(topic_model.get_topic_info())
            transcript_embedding = [word[1] for word in topic_model.get_topic(transcript_topics[0])]
            
            # 對summary的句子進行主題建模
            summary_topics, _ = topic_model.fit_transform(summary_sentences)
            # print(topic_model.get_topic_info())
            summary_embedding = [word[1] for word in topic_model.get_topic(summary_topics[0])]
            
            # 計算transcript和summary主題之間的餘弦相似性
            cosine_sim = cosine_similarity([transcript_embedding], [summary_embedding])[0][0]
            
            similarity.append(round(cosine_sim,2))

        # return similarity
        print(f"BERTopics Cosine Similarity: {similarity}")
        

    def single_summary(self):

        print(self.transcript[0:30])

        messages = [
            {"role": "system", "content": version1.system},
            # {"role": "assistant", "content": doc.page_content},
            {"role": "user", "content": version1.user + '\n\n' + f'transcript: {self.transcript}'},
        ]

        # print(messages)

        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=messages,
            temperature=0
        )

        summary = response['choices'][0]['message']['content']
        print(f"Prompt Token: {response['usage']['prompt_tokens']}.  Completion Tokens: {response['usage']['completion_tokens']}. Total tokens: {response['usage']['total_tokens']}")

        # Export to txt
        with open(f'./summary/{self.video_id}_single_v1_summary.txt', 'w') as f:
            f.write(summary)       


if __name__ == '__main__':

    # YoutubeSummary(video_id).gen_topics()
    # YoutubeSummary().functioning_api()
    # YoutubeSummary('Bazoq4mGWdU').embedding()
    # YoutubeSummary().retriever()
    # YoutubeSummary().query()
    video_id = 'Vwh2QwU9NhA'
    print(f"Video ID: {video_id}")

    tokens = YoutubeSummary(video_id).evaluate_transcript()
    
    if tokens <= 450:
        print("do it by 1 API call")
        YoutubeSummary(video_id).single_summary()
    elif tokens >= 32000:
        print("do it by topic modeling")
    else:
        print(f"do it by Map-Reduce")
        YoutubeSummary(video_id).bullet_points()
        btokens = YoutubeSummary(video_id).evaluate_bulle_points()
        if btokens <= 5000:
            YoutubeSummary(video_id).cluster()
            YoutubeSummary(video_id).title_and_summary()
        elif btokens <= 10000:
            print("do it by 32K API to get cluster & title & summary")
        else:
            print("do it by claude")

    # YoutubeSummary(video_id).bertopic()

