import streamlit as st
from pypdf import PdfReader
from src.analyzer import analyze_document, summarize_document
from src.summarizer import extract_key_points

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Intelligence")
st.subheader("Upload any PDF — Ask questions, get summaries!")

# File upload
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    # Extract text
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    st.success(f"✅ Document loaded: {len(reader.pages)} pages")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "❓ Ask Questions",
        "📝 Summary",
        "🔑 Key Points"
    ])

    with tab1:
        question = st.text_input("Ask anything about the document:")
        if question:
            with st.spinner("Analyzing..."):
                answer = analyze_document(text, question)
            st.write(answer)

    with tab2:
        if st.button("Generate Summary"):
            with st.spinner("Summarizing..."):
                summary = summarize_document(text)
            st.write(summary)

    with tab3:
        if st.button("Extract Key Points"):
            with st.spinner("Extracting..."):
                points = extract_key_points(text)
            st.write(points)