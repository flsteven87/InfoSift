import streamlit as st
from youtube_download import YoutubeDownload  # Correct class name

st.header("Youtube - Transcript & Summary")
# Get the YouTube URL from the user
url = st.text_input('Enter a YouTube URL')

# col1, col2 = st.columns([3,4])

if url:
    # Create an instance of the YoutubeDownload class
    video = YoutubeDownload(url)

    # Extract video id from url
    video_id = video.extract_video_id(url)

    # Display video information
    with st.spinner('Extracting video information...'):
        video_info = video.extract_video_info(url)

    st.success('Video information extracted successfully')
    st.write("### Video Information")
    for key, value in video_info.items():
        st.write(f"**{key}**: {value}")
    
    # Show a loading indicator while we're downloading the video
    with st.spinner('Downloading video...'):
        video.download_youtube_video(video_id)

    # Display a message once the video has been downloaded
    st.success('Video downloaded successfully')

    # Show a loading indicator while we're converting the video to .wav format
    with st.spinner('Converting video to .wav...'):
        video.convert_mp4_to_wav(f'./video/{video.file_name}.mp4',f'./video/{video.file_name}.wav')

    # Display a message once the conversion to .wav has been completed
    st.success('Video converted to .wav successfully')

    # Show a loading indicator while we're resampling the .wav to 16000Hz
    with st.spinner('Resampling .wav to 16000Hz...'):
        video.resample_wav(f'./video/{video.file_name}.wav',f'./video/{video.file_name}.wav', new_rate=16000)

    # Display a message once the resampling has been completed
    st.success('Audio resampled to 16000Hz successfully')

    # Show a loading indicator while we're generating the transcript using whisper.cpp
    with st.spinner('Generating transcript...'):
        video.whisper_cpp()

    # Display a message once the transcript completed
    st.success('Audeio transcript gererated successfully')

    # Read and display the transcript once it's ready
    with open(f'./transcript/{video.file_name}.wav.txt', 'r') as f:
        transcript = f.read()
        
    st.text_area('Transcript', transcript)
