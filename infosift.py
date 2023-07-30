import streamlit as st
from youtube_download import YoutubeDownload

# Function to read transcript file
def read_transcript_file(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

# Streamlit app main function
def main():
    st.title("YouTube Summarizer")
    
    yt_url = st.text_input("Enter YouTube Video URL:", value="")
    
    if yt_url:
        yt_download = YoutubeDownload(yt_url=yt_url)
        
        video_id = yt_download.extract_video_id(yt_url)
        st.write(f"Extracted Video ID: {video_id}")
        
        with st.spinner("Checking if video is in the database..."):
            video_info = yt_download.check_in_db()
            
        if video_info:
            st.success(f"Video is in the database. Displaying video info...")
            st.write(video_info)
            transcript_path = f"./transcript/txt/{video_id}.wav.txt"
            transcript = read_transcript_file(transcript_path)
            st.write("Transcript:")
            st.text_area("", value=transcript, height=200, max_chars=None)
            st.download_button("Download Transcript", transcript, file_name="transcript.txt")

        else:
            with st.spinner("Downloading video..."):
                yt_download.download()
            # Display a message once the video has been downloaded
            st.success('Video downloaded successfully')
    
            with st.spinner("Converting mp4 to wav..."):
                yt_download.convert_mp4_to_wav(f'./video/{video_id}.mp4', f'./video/{video_id}.wav')

            # Display a message once the conversion to .wav has been completed
            st.success('Video converted to .wav successfully')

            with st.spinner("Resampling wav..."):
                rate = 16000
                yt_download.resample_wav(f'./video/{video_id}.wav', f'./video/{video_id}.wav', new_rate=rate)

            # Display a message once the resampling has been completed
            st.success('Audio resampled to 16000Hz successfully')
        
            with st.spinner("Generating transcript..."):
                yt_download.whisper_cpp()
                
            st.success("Transcription process completed.")
            transcript_path = f"./transcript/txt/{video_id}.wav.txt"
            transcript = read_transcript_file(transcript_path)
            st.write("Transcript:")
            st.text_area("", value=transcript, height=200, max_chars=None)
            st.download_button("Download Transcript", transcript, file_name="transcript.txt")

if __name__ == "__main__":
    main()
