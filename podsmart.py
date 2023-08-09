import os
from datetime import datetime
import pandas as pd
import numpy as np
import json
from dotenv import load_dotenv
from scipy.spatial.distance import cosine
import networkx as nx
from networkx.algorithms import community
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import matplotlib.pyplot as plt

load_dotenv()

# OpenAI
import openai
os.environ["OPENAI_API_KEY"] = os.environ.get('OPENAI_API_KEY')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_API_KEY"] = os.environ.get('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_SESSION"] = "InfoSift"

class PodSmart:
    def __init__(self):
        self.video_id = 'Bazoq4mGWdU'
        self.transcript_file_path = f'./transcript/txt/{self.video_id}.wav.txt'
        self.openai_api_key = os.environ["OPENAI_API_KEY"]
        self.langchain_api_key = os.environ["LANGCHAIN_API_KEY"]
        self.model_name = "gpt-3.5-turbo-16k-0613"
        self.min_words = 20
        self.max_words = 80
        self.chunk_length = 5
        self.stride = 1
        self.num_topics = 5
        self.bonus_constant = 0.25
        self.min_size = 3
        self.summary_num_words = 250

    def load_transcript(self):
        with open(self.transcript_file_path, 'r') as f:
            transcript = f.read()

        # Get segments from transcript by splitting on .
        segments =  transcript.split('\n')
        # Put the . back in
        segments = [segment + '.' for segment in segments]
        # Further split by comma
        segments = [segment.split(',') for segment in segments]
        # Flatten
        self.segments = [item for sublist in segments for item in sublist]
    
    def create_sentences(self):
        # Combine the non-sentences together
        sentences = []
        is_new_sentence = True
        sentence_length = 0
        sentence_num = 0
        sentence_segments = []

        for i in range(len(self.segments)):
            if is_new_sentence:
                is_new_sentence = False
            # Append the segment
            sentence_segments.append(self.segments[i])
            segment_words = self.segments[i].split(' ')
            sentence_length += len(segment_words)
            
            # If exceed MAX_WORDS, then stop at the end of the segment
            # Only consider it a sentence if the length is at least MIN_WORDS
            if (sentence_length >= self.min_words and self.segments[i][-1] == '.') or sentence_length >= self.max_words:
                sentence = ' '.join(sentence_segments)
                sentences.append({
                    'sentence_num': sentence_num,
                    'text': sentence,
                    'sentence_length': sentence_length
                })
                # Reset
                is_new_sentence = True
                sentence_length = 0
                sentence_segments = []
                sentence_num += 1
                
        self.sentences = sentences

    def create_chunks(self):
        sentences_df = pd.DataFrame(self.sentences)
        
        chunks = []
        for i in range(0, len(sentences_df), (self.chunk_length - self.stride)):
            chunk = sentences_df.iloc[i:i+self.chunk_length]
            chunk_text = ' '.join(chunk['text'].tolist())
            
            chunks.append({
                'start_sentence_num': chunk['sentence_num'].iloc[0],
                'end_sentence_num': chunk['sentence_num'].iloc[-1],
                'text': chunk_text,
                'num_words': len(chunk_text.split(' '))
            })
            
        self.chunks = chunks
    
    def parse_title_summary_results(self, results):
        out = []
        for e in results:
            e = e.replace('\n', '')
            if '|' in e:
                processed = {'title': e.split('|')[0],
                            'summary': e.split('|')[1][1:]
                            }
            elif ':' in e:
                processed = {'title': e.split(':')[0],
                            'summary': e.split(':')[1][1:]
                            }
            elif '-' in e:
                processed = {'title': e.split('-')[0],
                            'summary': e.split('-')[1][1:]
                            }
            else:
                processed = {'title': '',
                            'summary': e
                            }
            out.append(processed)
        return out
    
    def summarize_stage_1(self):
        print(f'Start time: {datetime.now()}')

        topics = []
        for chunk in self.chunks:

            # Prompt to get title and summary for each chunk
            map_prompt_template = f"""Firstly, give the following text an informative title. 
            Then, on a new line, write a 75-100 word summary of the following text:
            {chunk}

            Return your answer in the following format by tranditional chinese:
            Title | Summary...
            e.g. 
            為何生成式 AI 掀起風潮？ | 人工智能可以通過自動化許多重複流程來提高人類的生產力。

            TITLE AND CONCISE SUMMARY:"""

            messages = [
                {"role": "system", "content": "You are a helpful assistant that helps retrieve topics talked about in a podcast transcript"},
                # {"role": "assistant", "content": doc.page_content},
                {"role": "user", "content": map_prompt_template,}
            ]

            # Get the response from OpenAI
            response = openai.ChatCompletion.create(
                model = self.model_name,
                messages=messages, 
                temperature=0
            )

            topics.append(response['choices'][0]['message']['content'])
            print(f"Prompt Token: {response['usage']['prompt_tokens']}.  Completion Tokens: {response['usage']['completion_tokens']}. Total tokens: {response['usage']['total_tokens']}")

        topics = "\n".join(topics)
        paragraphs = topics.split("\n")
        topics = [p for p in paragraphs]

        with open(f'./topics/stage1_{self.video_id}.txt', 'w') as f:
            for item in topics:
                f.write("%s\n" % item)

        # Parse the results from OpenAI
        stage_1_outputs = self.parse_title_summary_results(topics)

        print(f'Stage 1 done time {datetime.now()}')
        
        return stage_1_outputs
    
    def cluster_topics(self, summary_embeds, num_topics, bonus_constant=0.2):
        """
        Get similarity matrix between the embeddings of the chunk summaries
        """
        num_1_chunks = len(summary_embeds)
        summary_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))
        summary_similarity_matrix[:] = np.nan

        for row in range(num_1_chunks):
            for col in range(row, num_1_chunks):
                # Calculate cosine similarity between the two vectors
                similarity = 1 - cosine(summary_embeds[row], summary_embeds[col])
                summary_similarity_matrix[row, col] = similarity
                summary_similarity_matrix[col, row] = similarity

        # Get topics
        num_topics = min(int(num_1_chunks / 4), num_topics)
        topics_out = self.get_topics(summary_similarity_matrix, num_topics=num_topics, bonus_constant=bonus_constant)
        chunk_topics = topics_out['chunk_topics']
        topics = topics_out['topics']

        return chunk_topics, topics

    def get_topics(self, title_similarity, num_topics=8, bonus_constant=0.25, min_size=3):
        """
        Get topics from the title similarity
        """
        import networkx as nx
        from networkx.algorithms import community

        proximity_bonus_arr = np.zeros_like(title_similarity)
        for row in range(proximity_bonus_arr.shape[0]):
            for col in range(proximity_bonus_arr.shape[1]):
                if row == col:
                    proximity_bonus_arr[row, col] = 0
                else:
                    proximity_bonus_arr[row, col] = 1 / (abs(row - col)) * bonus_constant

        title_similarity += proximity_bonus_arr

        title_nx_graph = nx.from_numpy_array(title_similarity)

        desired_num_topics = num_topics
        topics_title_accepted = []

        resolution = 0.85
        resolution_step = 0.01
        iterations = 40

        topics_title = []
        while len(topics_title) not in [desired_num_topics, desired_num_topics + 1, desired_num_topics + 2]:
            topics_title = community.louvain_communities(title_nx_graph, weight='weight', resolution=resolution)
            resolution += resolution_step

        lowest_sd_iteration = 0
        lowest_sd = float('inf')

        for i in range(iterations):
            topics_title = community.louvain_communities(title_nx_graph, weight='weight', resolution=resolution)

            topic_sizes = [len(c) for c in topics_title]
            sizes_sd = np.std(topic_sizes)

            topics_title_accepted.append(topics_title)

            if sizes_sd < lowest_sd and min(topic_sizes) >= min_size:
                lowest_sd_iteration = i
                lowest_sd = sizes_sd

        topics_title = topics_title_accepted[lowest_sd_iteration]
        topic_id_means = [sum(e) / len(e) for e in topics_title]
        topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title), key=lambda pair: pair[0])]

        chunk_topics = [None] * title_similarity.shape[0]
        for i, c in enumerate(topics_title):
            for j in c:
                chunk_topics[j] = i

        return {
            'chunk_topics': chunk_topics,
            'topics': topics_title
        }

    def summarize_stage_2(self, stage_1_outputs, topics, summary_num_words=250):
        """
        Summarize each topic into an overall summary
        """
        print(f'Stage 2 start time {datetime.now()}')

        # Prompt that passes in all the titles of a topic, and asks for an overall title of the topic
        title_prompt_template = """Write an informative title that summarizes each of the following groups of titles. 
            Make sure that the titles capture as much information as possible, 
            and are different from each other:
            {text}
            
            Return your answer in a numbered list and using traditional chinese, with new line separating each title: 
            1. Title 1
            2. Title 2
            3. Title 3

            TITLES:
            """

        map_prompt_template = """Write a 75-100 word summary of the following text in traditional chinese:
            {text}

            CONCISE SUMMARY:"""

        combine_prompt_template = 'Write a ' + str(summary_num_words) + """-word summary of the following, removing irrelevant information. Finish your answer in traditional chinese:
            {text}
            """ + str(summary_num_words) + """-WORD SUMMARY:"""

        title_prompt = PromptTemplate(template=title_prompt_template, input_variables=["text"])
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

        topics_data = []
        for c in topics:
            topic_data = {
                'summaries': [stage_1_outputs[chunk_id]['summary'] for chunk_id in c],
                'titles': [stage_1_outputs[chunk_id]['title'] for chunk_id in c]
            }
            topic_data['summaries_concat'] = ' '.join(topic_data['summaries'])
            topic_data['titles_concat'] = ', '.join(topic_data['titles'])
            topics_data.append(topic_data)

        # Get a list of each community's summaries (concatenated)
        topics_summary_concat = [c['summaries_concat'] for c in topics_data]
        topics_titles_concat = [c['titles_concat'] for c in topics_data]

        # Concat into one long string to do the topic title creation
        topics_titles_concat_all = ''''''
        for i, c in enumerate(topics_titles_concat):
            topics_titles_concat_all += f'''{i+1}. {c}
            '''

        title_llm = OpenAI(temperature=0, model_name='text-davinci-003')
        title_llm_chain = LLMChain(llm=title_llm, prompt=title_prompt)
        title_llm_chain_input = [{'text': topics_titles_concat_all}]
        title_llm_chain_results = title_llm_chain.apply(title_llm_chain_input)

        titles = title_llm_chain_results[0]['text'].split('\n')
        titles = [t for t in titles if t != '']
        titles = [t.strip() for t in titles]

        map_llm = OpenAI(temperature=0, model_name='text-davinci-003')
        reduce_llm = OpenAI(temperature=0, model_name='text-davinci-003', max_tokens=-1)

        docs = [Document(page_content=t) for t in topics_summary_concat]
        chain = load_summarize_chain(chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt, return_intermediate_steps=True,
                                     llm=map_llm, reduce_llm=reduce_llm)

        output = chain({"input_documents": docs}, return_only_outputs=True)
        summaries = output['intermediate_steps']
        stage_2_outputs = [{'title': t, 'summary': s} for t, s in zip(titles, summaries)]
        final_summary = output['output_text']

        print(f'Stage 2 done time {datetime.now()}')

        return stage_2_outputs, final_summary
    
    def summarize(self):
        self.load_transcript()
        # print(self.segments)
        self.create_sentences()
        # print(self.sentences)
        self.create_chunks()
        # print(self.chunks)

        stage_1_output = self.summarize_stage_1()
        stage_1_summaries = [e['summary'] for e in stage_1_output]
        stage_1_titles = [e['title'] for e in stage_1_output]
        # num_1_chunks = len(stage_1_summaries)

        # Use OpenAI to embed the summaries and titles. Size of _embeds: (num_chunks x 1536)
        openai_embed = OpenAIEmbeddings()

        summary_embeds = np.array(openai_embed.embed_documents(stage_1_summaries))
        title_embeds = np.array(openai_embed.embed_documents(stage_1_titles))
        chunk_topics, topics = self.cluster_topics(summary_embeds, self.num_topics, bonus_constant=0.2)
        print(chunk_topics)
        print(topics)

        stage_2_outputs, final_summary = self.summarize_stage_2(stage_1_output, topics, summary_num_words=250)
        print(stage_2_outputs)
        print(final_summary)


        with open(f'./topics/stage2_{self.video_id}.txt', 'w') as f:
            for item in stage_2_outputs:
                f.write("%s\n" % item)

if __name__ == '__main__':
    PodSmart().summarize()
