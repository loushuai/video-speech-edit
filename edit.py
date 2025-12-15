import os
import tempfile
import subprocess
import streamlit as st
from utils import run_subprocess


os.environ["PYTHONPATH"] = "MuseTalk"


@st.fragment
def dub_function(output_path=None):
    txt = st.text_area("Dub Text", "", height=200)
    if "reference_voice_path_dict" in st.session_state:
        reference_voice = st.selectbox(
            "Select Reference Voice",
            st.session_state["reference_voice_path_dict"].keys(),
        )
        if st.button("Dub Video"):
            reference_audio_path = st.session_state["reference_voice_path_dict"][reference_voice]
            dub_audio_path = "tmp/dub_audio.wav"
            with st.spinner("Cloning voice and dubbing video..."):
                run_subprocess([
                    "python", "voice_clone.py",
                    "--text", txt,
                    "--reference_audio_path", reference_audio_path,
                    "--output_path", dub_audio_path,
                ], env="chatterbox")
                st.success("Dubbing completed!")

            with st.spinner("Generating video..."):
                run_subprocess([
                    "python", "dub.py",
                    "--video_path", st.session_state["video_path"],
                    "--audio_path", f"../{dub_audio_path}",
                    "--output_path", f"../{output_path}",
                ], env="MuseTalk")
                st.success("Dubbing completed!")

            st.video(output_path)


uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Create a temporary file to save the uploaded video
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        print(f"Video saved to {video_path}")
        st.video(video_path)

        apath = f"{video_path}.wav"
        vpath = f"{video_path}.mp4"
        subprocess.run(['ffmpeg', '-i', video_path, "-vn", apath , "-an", vpath], check=True)
        
        st.session_state["video_path"] = video_path
        st.session_state["output_ready"] = False

        output_path = "tmp/output.mp4"
        dub_function(output_path)

        # clean up temporary files
        # os.remove(video_path)
        # os.remove(apath)
        # os.remove(vpath)