@echo off
set TRANSFORMERS_NO_TF=1
set TF_ENABLE_ONEDNN_OPTS=0
streamlit run app.py
