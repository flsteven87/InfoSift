from dotenv import load_dotenv
import os
import openai
import argparse
from urllib.parse import parse_qs, urlparse
from pytube import YouTube
from pydub import AudioSegment
from webvtt import WebVTT

# import whisper
# import subprocess
# import threading

# Load environment variables from .env file
load_dotenv()
# Access API key from environment variable
openai.api_key = os.environ.get('GPT_API_KEY')
# print(openai.api_key)

class YoutubeDownload:
    def __init__(self, yt_url):
        self.youtube_url = yt_url
        self.file_name = None  # Initialize file_name as None
        self.current_folder = os.getcwd()
        # self.youtube_url = 'https://www.youtube.com/watch?v=v8jnUQAXHaw&ab_channel=%E8%B2%A1%E5%A0%B1%E7%8B%97'
        # self.file_name = '財報狗_242'
    
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

    def extract_video_info(self, youtube_url):
        try:
            video = YouTube(youtube_url)

            video_length = str(video.length) + ' seconds'
            video_size = str(round(video.streams.filter(file_extension='mp4').first().filesize*0.000001,2)) + ' MB'
            video_title = video.title
            channel_name = video.author
            # channel_subscribers = video.author.subscriber_count
            views_count = str(int(round(video.views/1000,0))) + "K views"

            return {
                "Video Title": video_title,
                "Channel Name": channel_name,
                "Video Length": video_length,
                "Video Size": video_size,
                # "Channel Subscribers": channel_subscribers,
                "Views": views_count
            }

        except Exception as e:
            return {"Error": str(e)}

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
        self.file_name = yt.title
        stream.download(output_path='./video/', filename=f'{self.file_name}.mp4')
        print(yt.title)
        # os.rename(f'./video/{yt.title}.mp4',f'./video/{self.file_name}.mp4')
        
        return yt.title

    def convert_mp4_to_wav(self, mp4_path, wav_path):
        audio = AudioSegment.from_file(mp4_path, "mp4")
        audio.export(wav_path, format="wav")

    def resample_wav(self, wav_path, output_path, new_rate=16000):
        audio = AudioSegment.from_wav(wav_path)
        audio = audio.set_frame_rate(new_rate)
        audio.export(output_path, format="wav")

    # Too Slow
    def video_to_transcript(self, format='wav'):

        model = whisper.load_model("large")
        result = model.transcribe(f"./video/{self.file_name}.{format}", language='zh')
        # print(result["text"])
        # Export to txt
        with open(f'./transcript/{self.file_name}.wav.txt', 'w') as f:
            f.write(result["text"])

    # Money
    def whisper_api(self,format='wav'):
        audio_file= open("/path/to/file/audio.mp3", "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file, language='zh')
        # Export to txt
        with open(f'./transcript/{self.file_name}.{format}.txt', 'w') as f:
            f.write(transcript['text'])


    def whisper_cpp(self):
        command = f"./whisper/whisper.cpp/main -m ./whisper/whisper.cpp/models/ggml-large.bin -l zh -ovtt -f '{self.current_folder}/video/{self.file_name}.wav'"
        print(command)
        os.system(command)

        source_file = f'./video/{self.file_name}.wav.vtt'
        destination_folder = './transcript/'
        filename = os.path.basename(source_file)
        destination_file = os.path.join(destination_folder, filename)
        os.replace(source_file, destination_file)

        subtitles = WebVTT().read(f'./transcript/{self.file_name}.wav.vtt')
        with open(f'./transcript/{self.file_name}.wav.txt', 'w', encoding='utf-8') as txt_file:
            for subtitle in subtitles:
                txt_file.write(subtitle.text + '\n')
        print(f"Conversion successful. TXT file saved as {self.file_name}.wav.txt.")

    def run(self):
        video_id = self.extract_video_id(self.youtube_url)
        file_name = self.download_youtube_video(video_id)
        print(f"Download Completed: {file_name}")
        self.convert_mp4_to_wav(f'./video/{self.file_name}.mp4',f'./video/{self.file_name}.wav')
        print(f"mp4 to wav Completed: {file_name}")
        rate = 16000
        self.resample_wav(f'./video/{self.file_name}.wav',f'./video/{self.file_name}.wav', new_rate=rate)
        print(f'Resample to {rate} Hz Completed: {file_name}')
        # self.video_to_transcript(format='mp4')
        # self.whisper_api(format='mp4')
        self.whisper_cpp()
        # self.whisper_cpp_streamlit()
        print(f'Video to transcript Completed: {self.file_name}')

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-u','--url', type=str, required=True, help='url of Youtube video')
    parser.add_argument('-n','--file_name', type=str, required=True, help='file name')
    args = parser.parse_args()
    return args

# usage: python -m youtube_download -u 'https://www.youtube.com/watch?v=ZQyMZKQessg&t=31s&ab_channel=%E8%B2%A1%E5%A0%B1%E7%8B%97' -n '財報狗_241'

if __name__ == '__main__':

    args = parse_args()
    YoutubeDownload(
        yt_url = args.url,
        # file_name = args.file_name
    ).run()