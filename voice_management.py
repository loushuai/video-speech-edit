import tempfile
import streamlit as st


st.title("Voice Management")

uploaded_audio = st.file_uploader("Upload a reference voice file", type=["mp3", "wav"])

if "reference_voice_path_dict" not in st.session_state:
    st.session_state["reference_voice_path_dict"] = {}

if uploaded_audio is not None:
    # Create a temporary file to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        tfile.write(uploaded_audio.read())
        audio_path = tfile.name
        print(f"audio saved to {audio_path}")
        # st.audio(audio_path)
        st.session_state["reference_voice_path_dict"][uploaded_audio.name] = audio_path
        st.success("Reference voice uploaded")
        print(st.session_state["reference_voice_path_dict"])

for name, path in st.session_state["reference_voice_path_dict"].items():
    col1, col2 = st.columns(2)
    with col1:
        st.write(name)
    with col2:
        st.audio(path)
