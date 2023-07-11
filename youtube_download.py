from dotenv import load_dotenv
import os
import openai
import argparse
from urllib.parse import parse_qs, urlparse
from pytube import YouTube
from pydub import AudioSegment
import json
# import whisper

# Load environment variables from .env file
load_dotenv()
# Access API key from environment variable
openai.api_key = os.environ.get('GPT_API_KEY')
# print(openai.api_key)

class YoutubeSummary:
    def __init__(self):
        self.youtube_url = 'https://www.youtube.com/watch?v=v8jnUQAXHaw&ab_channel=%E8%B2%A1%E5%A0%B1%E7%8B%97'
        self.file_name = '財報狗_242'
    
    def extract_video_id(self, url):
        query = urlparse(url)
        if query.hostname == 'youtu.be': # Shortened URL
            return query.path[1:]
        elif query.hostname in {'www.youtube.com', 'youtube.com'}:
            if query.path == '/watch': # Full URL
                return parse_qs(query.query)['v'][0]
            elif query.path[:7] == '/embed/': # Embedded URL
                return query.path.split('/')[2]
            elif query.path[:3] == '/v/': # Old-style URL
                return query.path.split('/')[2]
        # None of the above cases match, so raise an exception
        raise ValueError(f"Invalid YouTube URL: {url}")                   

    def download_youtube_video(self, video_id):
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        yt = YouTube(youtube_url, use_oauth=True, allow_oauth_cache=True)
        
        # Get the first available audio or video stream
        stream = yt.streams.filter(progressive=True, file_extension='wav').first()
        if not stream:
            stream = yt.streams.filter(only_audio=True).first()
        if not stream:
            raise Exception("No available streams found.")
        
        # Download the stream
        stream.download(output_path='./video/')
        print(yt.title)
        os.rename(f'./video/{yt.title}.mp4',f'./video/{self.file_name}.mp4')
        
        return yt.title

    # A function to transcript the audio file to text
    def speech_to_text(self, file_name):

        try:
            with open(f"./transcript/{file_name}.json") as f:
                data = json.load(f)
                transcript = data['text']
        except FileNotFoundError:
            print("File does not exist")
            # print("Now calling API...")
            audio_file= open(os.getcwd()+f'./video/{file_name}.mp4', "rb")

            # get the audio file path
            path = os.path.join(".", "video", f"{file_name}.mp4")
            print(path)

            # read the audio file
            # audio_file = AudioSegment.from_file(path)

            # 檢查音檔是否可以正常解碼
            # print(audio_file)

            # 將音檔轉換為 MP3 格式
            # mp3_file = audio_file.export(os.getcwd()+f'./video/{file_name}.mp3', format="mp3")

            # 使用 OPENAI API
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
            
            with open(f"./transcript/{file_name}.json", 'w') as f:
                json.dump(transcript, f)

            with open(f"./transcript/{file_name}.json") as f:
                data = json.load(f)
                transcript = data['text']

        return transcript

    # wait for retired
    def generate_summary(self, file_name, transcript, max_token):
        llm = OpenAI(openai_api_key=openai.api_key, max_tokens=max_token)
        char_text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=30)
        transcripts = char_text_splitter.split_text(transcript)
        print(len(transcripts))
        docs = [Document(page_content=t) for t in transcripts]

        # Refine
        # prompt = '''
        # Your job is to produce a final summary into newsletter under the structure with a article headline, backgroud, and few important bullet points by Tranditional Chinese.

        # {text}
        
        # And do it with Tranditional Chinese.
        # '''
        # # prompt_template = """Write a concise summary of the following: {text} CONCISE SUMMARY IN ITALIAN:"""
        # PROMPT = PromptTemplate(template=prompt, input_variables=["text"])
        # refine_template = (
        #     "Your job is to produce a final summary into newsletter under the structure with a article headline, backgroud, and few important bullet points by Tranditional Chinese\n"
        #     "We have provided an existing summary newsletter up to a certain point: {existing_answer}\n"
        #     "We have the opportunity to refine the existing summary"
        #     "(only if needed) with some more context below.\n"
        #     "------------\n"
        #     "{text}\n"
        #     "------------\n"
        #     "Given the new context, refine the original newsletter."
        # )
        # refine_prompt = PromptTemplate(
        #     input_variables=["existing_answer", "text"],
        #     template=refine_template,
        # )
        # chain = load_summarize_chain(llm=llm, chain_type="refine", question_prompt=PROMPT, refine_prompt=refine_prompt)
        
        # Map-Reduce
        map_prompt = """
        Your job is to produce a map part summary of a map-reduce structure youtube transcript summarization.
        Write a map part summary of the following: \n
        {text}
        """
        combine_prompt_template = """
        Your job is to produce a final summary into newsletter under the structure with a article headline, backgroud, and few important bullet points
        "{text}\n"
        """
        map_prompt_template = """Your job is to produce a map part summary of a map-reduce structure youtube video summarization.
        Write a concise summary of the following:
        {text}
        CONCISE SUMMARY:"""
        map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
        combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm=llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=map_prompt, combine_prompt=combine_prompt)
        summary = chain({"input_documents": docs}, return_only_outputs=True)

        with open(f"./summary/{file_name}.json", 'w') as f:
            json.dump(summary, f)

        with open(f"./summary/{file_name}.json") as f:
            data = json.load(f)
            summary = data['output_text']

        # response = openai.Completion.create(
        #     engine="text-davinci-003",
        #     prompt=prompt + transcript,
        #     max_tokens=max_token,
        #     n=1,
        #     stop=None,
        #     temperature=0.7,
        # )
        # with open(f"./transcript/{file_name}_summary.txt", "w") as text_file:
        #     text_file.write( response.choices[0].text)

        # return response.choices[0].text
        return summary



    def convert_mp4_to_wav(self, mp4_path, wav_path):
        audio = AudioSegment.from_file(mp4_path, "mp4")
        audio.export(wav_path, format="wav")

    def run(self):
        video_id = self.extract_video_id(self.youtube_url)
        file_name = self.download_youtube_video(video_id)
        print(f"Download Completed: {file_name}")
        self.convert_mp4_to_wav(f'./video/{self.file_name}.mp4',f'./video/{self.file_name}.wav')

def parse_args():
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    return args


#usage 轉為 wav 檔案 ffmpeg -i /Users/po-chi/Desktop/InfoSift/video/財報狗_podcast_239.mp4 -ar 16000 -ac 1 -c:a pcm_s16le video/財報狗_podcast__239.wav
#usage 使用 whisper 轉換 wav > transcript ./main -m models/ggml-large.bin -l zh -otxt -f '/Users/po-chi/Desktop/InfoSift/video/財報狗_podcast__239.wav'
if __name__ == '__main__':

    args = parse_args()
    YoutubeSummary().run()