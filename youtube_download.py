from dotenv import load_dotenv
import os
import openai
import argparse
from urllib.parse import parse_qs, urlparse
from pytube import YouTube
from pydub import AudioSegment
from webvtt import WebVTT
import sqlite3
from datetime import datetime

# Load environment variables from .env file
load_dotenv()
# Access API key from environment variable
openai.api_key = os.environ.get('GPT_API_KEY')

class YoutubeDownload:
    def __init__(self, yt_url):
        self.yt_url = yt_url
        self.video_id = self.extract_video_id(yt_url)
        self.yt = YouTube(f'https://www.youtube.com/watch?v={self.video_id}')
        self.current_folder = os.getcwd()
        self.conn = sqlite3.connect('./sqlite/infosift.db')
        self.cursor = self.conn.cursor()

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
    
    def check_in_db(self):

        sql = f'''
            SELECT *
            FROM video
            WHERE video_id = '{self.video_id}'
        '''
        self.cursor.execute(sql)
        video_info = self.cursor.fetchone()
        return video_info

    def download(self):
        video = self.yt.streams.get_highest_resolution()
        video.download(output_path='./video/', filename=f'{self.video_id}.mp4')

        video_info = {
            'video_id': self.video_id,
            'url': self.yt_url,
            'title': self.yt.title,
            'channel': self.yt.author,
            'video_length': self.yt.length,
            'video_size': video.filesize,
            'views_count': self.yt.views,
            'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'updated_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        print(video_info)
        placeholders = ', '.join(['?'] * len(video_info))
        columns = ', '.join(video_info.keys())
        sql = f'''
            INSERT INTO video ({columns})
            VALUES ({placeholders})
            ON CONFLICT(video_id) DO UPDATE SET
            title = excluded.title,
            channel = excluded.channel,
            video_length = excluded.video_length,
            video_size = excluded.video_size,
            views_count = excluded.views_count,
            updated_time = excluded.updated_time
        '''
        self.cursor.execute(sql, list(video_info.values()))
        self.conn.commit()

    def convert_mp4_to_wav(self, mp4_path, wav_path):
        audio = AudioSegment.from_file(mp4_path, "mp4")
        audio.export(wav_path, format="wav")

    def resample_wav(self, wav_path, output_path, new_rate=16000):
        audio = AudioSegment.from_wav(wav_path)
        audio = audio.set_frame_rate(new_rate)
        audio.export(output_path, format="wav")

    def whisper_cpp(self):
        command = f"./whisper/whisper.cpp/main -m ./whisper/whisper.cpp/models/ggml-large.bin -l zh -ovtt -f '{self.current_folder}/video/{self.video_id}.wav'"
        print(command)
        os.system(command)

        source_file = f'./video/{self.video_id}.wav.vtt'
        destination_folder = './transcript/vtt/'
        filename = os.path.basename(source_file)
        destination_file = os.path.join(destination_folder, filename)
        os.replace(source_file, destination_file)

        subtitles = WebVTT().read(f'./transcript/vtt/{self.video_id}.wav.vtt')
        with open(f'./transcript/txt/{self.video_id}.wav.txt', 'w', encoding='utf-8') as txt_file:
            for subtitle in subtitles:
                txt_file.write(subtitle.text + '\n')
        print(f"Conversion successful. TXT file saved as {self.video_id}.wav.txt.")

    def run(self):
        video_info = self.check_in_db()
        if video_info is None:
            # Run the full download and transcription process if the video is not in the database
            self.download()
            print(f"Download Completed: {self.yt.title}")
            self.convert_mp4_to_wav(f'./video/{self.video_id}.mp4',f'./video/{self.video_id}.wav')
            print(f"mp4 to wav Completed: {self.yt.title}")
            rate = 16000
            self.resample_wav(f'./video/{self.video_id}.wav',f'./video/{self.video_id}.wav', new_rate=rate)
            print(f'Resample to {rate} Hz Completed: {self.yt.title}')
            self.whisper_cpp()
            print(f'Video to transcript Completed: {self.yt.title}')
        else:
            print(f'Video already in database: {self.yt.title}')

    def transcript_only(self):
        self.extract_video_id(self.yt_url)
        self.whisper_cpp()

    def __del__(self):
        self.conn.close()

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-u','--url', type=str, required=True, help='url of Youtube video')
    args = parser.parse_args()
    return args

# usage: python -m youtube_download -u 'https://www.youtube.com/watch?v=ZQyMZKQessg&t=31s&ab_channel=%E8%B2%A1%E5%A0%B1%E7%8B%97'

if __name__ == '__main__':

    args = parse_args()
    YoutubeDownload(yt_url = args.url,).run()
    # YoutubeDownload(yt_url = args.url,).transcript_only()