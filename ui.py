import os
import tempfile
import subprocess
import whisper
import streamlit as st


edit_page = st.Page("edit.py", title="Video Speech Edit")
voice_management_page = st.Page("voice_management.py", title="Voice Management")
pg = st.navigation([edit_page, voice_management_page])
pg.run()




