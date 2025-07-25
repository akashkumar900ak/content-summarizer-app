import streamlit as st
import pandas as pd
from summarizer import TextSummarizer
import time
import io
import PyPDF2
import traceback

# Page configuration
st.set_page_config(
    page_title="AI Content Summarizer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = None

def load_summarizer():
    """Load the summarizer model with caching"""
    if st.session_state.summarizer is None:
        with st.spinner("Loading AI model... This may take a moment on first use."):
            st.session_state.summarizer = TextSummarizer()
    return st.session_state.summarizer

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
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
    """Extract text from uploaded TXT file"""
    try:
        # Convert bytes to string
        text = txt_file.read().decode('utf-8')
        return text.strip()
    except Exception as e:
        st.error(f"Error reading text file: {str(e)}")
        return None

def main():
    # Header
    st.title("🤖 AI Content Summarizer")
    st.markdown("### Transform long content into concise, meaningful summaries")
    st.markdown("Powered by Facebook's BART model via Hugging Face Transformers")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Summary length selection
        summary_length = st.selectbox(
            "Summary Length:",
            options=["short", "medium", "long"],
            index=1,
            help="Choose how detailed you want the summary to be"
        )
        
        # Input method selection
        input_method = st.radio(
            "Input Method:",
            options=["Text Input", "File Upload"],
            help="Choose how to provide your content"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.info("Using facebook/bart-large-cnn\nOptimized for news summarization")
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Works best with articles, papers, and transcripts
        - Minimum 50 characters required
        - Large texts are automatically chunked
        - PDF and TXT files supported
        """)
    
    # Main content area
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📝 Input Content")
        
        text_to_summarize = ""
        
        if input_method == "Text Input":
            # Text input area
            text_to_summarize = st.text_area(
                "Paste your content here:",
                height=300,
                placeholder="Enter news articles, research papers, meeting transcripts, or any text you'd like to summarize...",
                help="Paste or type the content you want to summarize"
            )
        
        else:  # File Upload
            uploaded_file = st.file_uploader(
                "Upload a file:",
                type=['txt', 'pdf'],
                help="Upload a .txt or .pdf file to summarize its contents"
            )
            
            if uploaded_file is not None:
                with st.spinner("Reading file..."):
                    if uploaded_file.type == "application/pdf":
                        text_to_summarize = extract_text_from_pdf(uploaded_file)
                    elif uploaded_file.type == "text/plain":
                        text_to_summarize = extract_text_from_txt(uploaded_file)
                    
                    if text_to_summarize:
                        st.success(f"✅ File loaded successfully! ({len(text_to_summarize)} characters)")
                        # Show preview of uploaded content
                        with st.expander("📄 Preview uploaded content"):
                            st.text_area("File content:", text_to_summarize[:1000] + "..." if len(text_to_summarize) > 1000 else text_to_summarize, height=200, disabled=True)
        
        # Summarize button
        summarize_btn = st.button(
            "🚀 Generate Summary",
            type="primary",
            use_container_width=True,
            help="Click to generate an AI-powered summary"
        )
    
    with col2:
        st.subheader("📋 Summary Output")
        
        # Summary container
        summary_container = st.container()
        
        if summarize_btn:
            if not text_to_summarize or len(text_to_summarize.strip()) < 50:
                st.error("⚠️ Please provide at least 50 characters of text to summarize.")
            else:
                try:
                    # Load model
                    summarizer = load_summarizer()
                    
                    # Show processing message
                    with summary_container:
                        progress_placeholder = st.empty()
                        with progress_placeholder:
                            st.info("🔄 Generating summary... Please wait.")
                            progress_bar = st.progress(0)
                            for i in range(100):
                                time.sleep(0.01)
                                progress_bar.progress(i + 1)
                        
                        # Generate summary
                        start_time = time.time()
                        summary = summarizer.summarize(
                            text_to_summarize, 
                            length=summary_length
                        )
                        end_time = time.time()
                        
                        # Clear progress and show results
                        progress_placeholder.empty()
                        
                        # Display summary
                        st.success("✅ Summary Generated!")
                        st.markdown("### 📄 Summary:")
                        st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;"><p style="margin: 0; font-size: 16px; line-height: 1.6;">{summary}</p></div>', unsafe_allow_html=True)
                        
                        # Statistics
                        st.markdown("### 📊 Statistics:")
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        
                        with col_stats1:
                            st.metric("Original Length", f"{len(text_to_summarize)} chars")
                        
                        with col_stats2:
                            st.metric("Summary Length", f"{len(summary)} chars")
                        
                        with col_stats3:
                            compression_ratio = round((1 - len(summary) / len(text_to_summarize)) * 100, 1)
                            st.metric("Compression", f"{compression_ratio}%")
                        
                        st.caption(f"⏱️ Processing time: {end_time - start_time:.2f} seconds")
                        
                        # Download button for summary
                        st.download_button(
                            label="📥 Download Summary",
                            data=summary,
                            file_name="summary.txt",
                            mime="text/plain",
                            help="Download the generated summary as a text file"
                        )
                
                except Exception as e:
                    st.error(f"❌ An error occurred while generating the summary: {str(e)}")
                    st.error("Please try again or contact support if the issue persists.")
                    # For debugging (remove in production)
                    with st.expander("Debug Information"):
                        st.code(traceback.format_exc())
        
        else:
            with summary_container:
                st.info("👆 Enter text and click 'Generate Summary' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit and Hugging Face Transformers<br>
            Deploy this app for free on <a href='https://streamlit.io/cloud' target='_blank'>Streamlit Community Cloud</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Example texts for testing
def show_examples():
    st.sidebar.markdown("### 📚 Example Texts")
    
    examples = {
        "News Article": """
        The global economy is facing unprecedented challenges as inflation rates continue to soar across major economies. 
        Central banks worldwide are grappling with the delicate balance between controlling inflation and maintaining economic growth. 
        The Federal Reserve has raised interest rates multiple times this year, while the European Central Bank follows a similar trajectory. 
        Supply chain disruptions, energy costs, and geopolitical tensions are contributing factors to the current economic climate. 
        Consumers are feeling the impact through higher prices for everyday goods and services, leading to changes in spending behavior. 
        Economists predict that the situation may persist for several more quarters before stabilization occurs.
        """,
        
        "Research Abstract": """
        This study investigates the impact of artificial intelligence on modern healthcare delivery systems. 
        Through a comprehensive analysis of 500 healthcare institutions across North America and Europe, 
        we examined the implementation of AI-driven diagnostic tools and their effect on patient outcomes. 
        Our findings indicate a 23% improvement in diagnostic accuracy and a 15% reduction in treatment time 
        when AI systems are integrated with traditional medical practices. The research methodology included 
        both quantitative analysis of patient data and qualitative interviews with healthcare professionals. 
        Statistical significance was achieved across all measured parameters (p < 0.05). 
        These results suggest that AI integration in healthcare can significantly enhance both efficiency and accuracy of medical services.
        """
    }
    
    for title, text in examples.items():
        if st.sidebar.button(f"Load {title}", key=title):
            st.session_state.example_text = text

if __name__ == "__main__":
    main()
