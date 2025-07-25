import streamlit as st
import pandas as pd
from summarizer_module import TextSummarizer
import time
import io
import PyPDF2
import traceback

# Page settings
st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load summarizer model
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None

def load_summarizer():
    if st.session_state.summarizer is None:
        with st.spinner("Loading AI model... This may take a moment on first use."):
            st.session_state.summarizer = TextSummarizer()
    return st.session_state.summarizer

def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()]).strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    try:
        return txt_file.read().decode('utf-8').strip()
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

def main():
    st.title("🤖 AI Content Summarizer")
    st.markdown("### Transform long content into concise, meaningful summaries using Facebook's BART model.")

    with st.sidebar:
        st.header("⚙️ Settings")
        summary_length = st.selectbox("Summary Length:", ["short", "medium", "long"], index=1)
        input_method = st.radio("Input Method:", ["Text Input", "File Upload"])
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Best for articles, papers, transcripts
        - Min 50 characters
        - PDF/TXT supported
        """)
        st.markdown("---")
        st.info("Model: `facebook/bart-large-cnn`\nStreamlit + Hugging Face")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📝 Input")
        text_to_summarize = ""

        if input_method == "Text Input":
            text_to_summarize = st.text_area("Paste your content here:", height=300)
        else:
            uploaded_file = st.file_uploader("Upload a file:", type=['pdf', 'txt'])
            if uploaded_file:
                with st.spinner("Reading file..."):
                    if uploaded_file.type == "application/pdf":
                        text_to_summarize = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "text/plain":
                        text_to_summarize = extract_text_from_txt(uploaded_file)

                    if text_to_summarize:
                        st.success(f"✅ File loaded! ({len(text_to_summarize)} characters)")
                        with st.expander("📄 Preview"):
                            preview = text_to_summarize[:1000] + "..." if len(text_to_summarize) > 1000 else text_to_summarize
                            st.text_area("Content Preview", preview, height=200, disabled=True)

        summarize_btn = st.button("🚀 Generate Summary")

    with col2:
        st.subheader("📋 Summary Output")
        summary_container = st.container()

        if summarize_btn:
            if not text_to_summarize or len(text_to_summarize.strip()) < 50:
                st.error("⚠️ Please provide at least 50 characters.")
            else:
                try:
                    summarizer = load_summarizer()
                    with summary_container:
                        st.info("🔄 Generating summary...")
                        progress = st.progress(0)
                        for i in range(100):
                            time.sleep(0.01)
                            progress.progress(i + 1)

                        start_time = time.time()
                        summary = summarizer.summarize(text_to_summarize, length=summary_length)
                        end_time = time.time()
                        progress.empty()

                        # ✅ Display summary clearly
                        st.success("✅ Summary Generated!")
                        st.markdown("### 📄 Summary:")
                        st.markdown(f"""
                        <div style="background-color:#f8f9fa;padding:20px;border-left:5px solid #4CAF50;border-radius:5px;">
                        {summary}
                        </div>
                        """, unsafe_allow_html=True)

                        # 📊 Stats
                        original_len = len(text_to_summarize)
                        summary_len = len(summary)
                        compression = round((1 - summary_len / original_len) * 100, 1)
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Original", f"{original_len} chars")
                        col2.metric("Summary", f"{summary_len} chars")
                        col3.metric("Compression", f"{compression}%")
                        st.caption(f"⏱️ Processed in {end_time - start_time:.2f} sec")

                        # ✅ Optional Download Checkbox
                        st.markdown("---")
                        st.markdown("### 📥 Want to keep this?")
                        if st.checkbox("✅ I'm happy with the summary. Show download button."):
                            st.download_button("📥 Download Summary", summary, file_name="summary.txt", mime="text/plain")

                except Exception as e:
                    st.error("❌ Error during summarization.")
                    with st.expander("Debug Info"):
                        st.code(traceback.format_exc())

        else:
            with summary_container:
                st.info("👆 Paste/upload your content and click 'Generate Summary'.")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #888;'>
        Built with ❤️ using Streamlit & Hugging Face<br>
        <a href='https://streamlit.io/cloud' target='_blank'>Deploy it yourself</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
