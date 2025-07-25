import streamlit as st
import pandas as pd
from summarizer_module import TextSummarizer  # ✅ FIXED import
import time
import io
import PyPDF2
import traceback

st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None

def load_summarizer():
    if st.session_state.summarizer is None:
        with st.spinner("Loading AI model... This may take a moment on first use."):
            st.session_state.summarizer = TextSummarizer()
    return st.session_state.summarizer

def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

def extract_text_from_txt(txt_file):
    try:
        text = txt_file.read().decode('utf-8')
        return text.strip()
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

def main():
    st.title("🤖 AI Content Summarizer")
    st.markdown("### Transform long content into concise, meaningful summaries")
    st.markdown("Powered by Facebook's BART model via Hugging Face Transformers")

    with st.sidebar:
        st.header("⚙️ Settings")

        summary_length = st.selectbox(
            "Summary Length:",
            options=["short", "medium", "long"],
            index=1
        )

        input_method = st.radio(
            "Input Method:",
            options=["Text Input", "File Upload"]
        )

        st.markdown("---")
        st.info("Using facebook/bart-large-cnn\nOptimized for news summarization")

        st.markdown("### 💡 Tips")
        st.markdown("""
        - Works best with articles, papers, and transcripts
        - Minimum 50 characters required
        - Large texts are automatically chunked
        - PDF and TXT files supported
        """)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📝 Input Content")
        text_to_summarize = ""

        if input_method == "Text Input":
            text_to_summarize = st.text_area(
                "Paste your content here:",
                height=300,
                placeholder="Enter text here..."
            )
        else:
            uploaded_file = st.file_uploader("Upload a file:", type=['txt', 'pdf'])
            if uploaded_file is not None:
                with st.spinner("Reading file..."):
                    if uploaded_file.type == "application/pdf":
                        text_to_summarize = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "text/plain":
                        text_to_summarize = extract_text_from_txt(uploaded_file)

                    if text_to_summarize:
                        st.success(f"✅ File loaded successfully! ({len(text_to_summarize)} characters)")
                        with st.expander("📄 Preview uploaded content"):
                            preview = text_to_summarize[:1000] + "..." if len(text_to_summarize) > 1000 else text_to_summarize
                            st.text_area("File content:", preview, height=200, disabled=True)

        summarize_btn = st.button("🚀 Generate Summary", type="primary")

    with col2:
        st.subheader("📋 Summary Output")
        summary_container = st.container()

        if summarize_btn:
            if not text_to_summarize or len(text_to_summarize.strip()) < 50:
                st.error("⚠️ Please provide at least 50 characters of text.")
            else:
                try:
                    summarizer = load_summarizer()

                    with summary_container:
                        progress_placeholder = st.empty()
                        with progress_placeholder:
                            st.info("🔄 Generating summary... Please wait.")
                            progress_bar = st.progress(0)
                            for i in range(100):
                                time.sleep(0.01)
                                progress_bar.progress(i + 1)

                        start_time = time.time()
                        summary = summarizer.summarize(text_to_summarize, length=summary_length)
                        end_time = time.time()

                        progress_placeholder.empty()

                        st.success("✅ Summary Generated!")
                        st.markdown("### 📄 Summary:")
                        st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;"><p style="margin: 0; font-size: 16px; line-height: 1.6;">{summary}</p></div>', unsafe_allow_html=True)

                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("Original Length", f"{len(text_to_summarize)} chars")
                        with col_stats2:
                            st.metric("Summary Length", f"{len(summary)} chars")
                        with col_stats3:
                            compression = round((1 - len(summary) / len(text_to_summarize)) * 100, 1)
                            st.metric("Compression", f"{compression}%")

                        st.caption(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")

                        st.download_button(
                            label="📥 Download Summary",
                            data=summary,
                            file_name="summary.txt",
                            mime="text/plain"
                        )
                except Exception as e:
                    st.error(f"❌ Error during summarization: {str(e)}")
                    with st.expander("Debug Info"):
                        st.code(traceback.format_exc())
        else:
            with summary_container:
                st.info("👆 Enter text and click 'Generate Summary' to see results here.")

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit and Hugging Face Transformers<br>
            Deploy for free at <a href='https://streamlit.io/cloud' target='_blank'>Streamlit Community Cloud</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
