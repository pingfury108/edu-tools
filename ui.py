import streamlit as st

st.title("edu-tools")

topic = st.container(key="topic", border=True)
topic.text_area(label="题目整理")
if topic.button(label="开始整理"):
    topic.text("helllo")
