import streamlit as st
from src.orchestrator import run_multi_agent

st.set_page_config(
    page_title="Multi-Agent System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agent AI System")
st.subheader("Three AI Agents working together!")

st.markdown("""
### How It Works:
- 🔍 **Agent 1** — Researches your topic
- 📊 **Agent 2** — Analyzes the research  
- 📝 **Agent 3** — Summarizes everything
""")

topic = st.text_input("Enter any topic:")

if st.button("Run Multi-Agent System"):
    if topic:
        with st.spinner("Agents working..."):
            results = run_multi_agent(topic)

        st.subheader("🔍 Agent 1 — Research:")
        st.write(results["research"])

        st.subheader("📊 Agent 2 — Analysis:")
        st.write(results["analysis"])

        st.subheader("📝 Agent 3 — Summary:")
        st.write(results["summary"])