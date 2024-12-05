import streamlit as st
import sys
from pathlib import Path
import PyPDF2
from typing import List, Dict
import json
from datetime import datetime

# Basic setup
st.set_page_config(page_title="TEMP Rebuttal Assistant", page_icon="📝", layout="wide")

# Setup paths and imports
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent.absolute()
function_calls_dir = parent_dir / "function_calls"
sys.path.append(str(function_calls_dir))

from retriever import parse_subsections, retrieve
from generation import generate_response

class SessionState:
    def __init__(self):
        self.messages: List[Dict] = []
        self.office_action_text: str = None
        self.current_sections: List[Dict] = []
        self.selected_sections: List[Dict] = []

def init_session_state():
    if 'state' not in st.session_state:
        st.session_state.state = SessionState()
    return st.session_state.state

def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        return " ".join(page.extract_text() or "" for page in pdf_reader.pages)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def display_chat_message(message: Dict, state: SessionState):
    role = message["role"]
    content = message["content"]
    
    # Display message with appropriate styling
    st.markdown(f"""
        <div class="streamlit-chat-message" data-role="{role}">
            {content}
        </div>
    """, unsafe_allow_html=True)
    
    # Display sections if present
    if "sections" in message:
        display_sections(message["sections"], state)

def display_sections(sections: List[Dict], state: SessionState):
    """Display retrieved sections with improved UI"""
    st.markdown("### Retrieved Sections")
    
    if isinstance(sections[0], str):
        # Extract individual references
        # Remove the outer list brackets and split by Reference
        content = sections[0].strip('[]"').replace('\\n', '\n')
        references = content.split("- Reference[")
        
        for ref in references:
            if not ref.strip():  # Skip empty entries
                continue
            
            try:
                # Split into subsection and summary
                # First, get the reference number
                ref_num = ref.split(']:')[0] if ']' in ref else ""
                
                # Split the remaining content by the known markers
                subsection = ""
                summary = ""
                
                if "- Subsection:" in ref and "- Summary of content:" in ref:
                    parts = ref.split("- Subsection:", 1)[1].split("- Summary of content:")
                    subsection = parts[0].strip()
                    summary = parts[1].strip() if len(parts) > 1 else ""
                
                # Create a clean, organized display using a container
                with st.container():
                    # Use columns for better organization
                    col1, col2 = st.columns([0.1, 0.9])
                    
                    with col1:
                        # Checkbox for selection
                        selected = st.checkbox("Select", key=f"select_{ref_num}")
                    
                    with col2:
                        # Display reference content in a card-like format
                        st.markdown(f"""
                            <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 0.5rem; border: 1px solid #dee2e6;'>
                                <h4 style='color: #0d6efd; margin-bottom: 0.5rem;'>Reference {ref_num}:</h4>
                                <div style='margin-left: 1rem;'>
                                    <p style='font-weight: bold; margin-bottom: 0.5rem;'>Subsection:</p>
                                    <p style='margin-left: 1rem; margin-bottom: 1rem;'>{subsection}</p>
                                    <p style='font-weight: bold; margin-bottom: 0.5rem;'>Summary:</p>
                                    <p style='margin-left: 1rem;'>{summary}</p>
                                </div>
                            </div>
                            <div style='margin-bottom: 1rem;'></div>
                        """, unsafe_allow_html=True)
                    
                    # Handle selection
                    if selected:
                        # Find corresponding full section in current_sections
                        full_section = next(
                            (s for s in state.current_sections 
                             if subsection in s['header']), 
                            None
                        )
                        if full_section and full_section not in state.selected_sections:
                            state.selected_sections.append(full_section)
                    else:
                        # Remove from selected sections if unchecked
                        full_section = next(
                            (s for s in state.current_sections 
                             if subsection in s['header']), 
                            None
                        )
                        if full_section and full_section in state.selected_sections:
                            state.selected_sections.remove(full_section)
                            
            except Exception as e:
                st.error(f"Error parsing reference: {e}")
                
        # Show summary of selected sections
        if state.selected_sections:
            st.markdown(f"""
                <div style='background-color: #e7f3ff; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;'>
                    <p style='color: #0d6efd; margin: 0;'>
                        Selected {len(state.selected_sections)} sections for response generation
                    </p>
                </div>
            """, unsafe_allow_html=True)

def main():
    st.title("USPTO Office Action Response Assistant")
    state = init_session_state()

    # Initialize session state variables
    if 'retrieval_method' not in st.session_state:
        st.session_state.retrieval_method = "vector"
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Office Action Letter (PDF)",
            type="pdf",
            help="Upload the USPTO Office Action letter you want to analyze"
        )
        
        if uploaded_file and not state.office_action_text:
            state.office_action_text = extract_text_from_pdf(uploaded_file)
            if state.office_action_text:
                st.success("✅ Office Action letter processed successfully!")
        
        # Retrieval settings
        st.subheader("Retrieval Settings")
        
        retrieval_method = st.selectbox(
            "Retrieval Method",
            ["vector", "BM25", "RFF"],
            help="Select the method used to retrieve relevant sections"
        )
        
        top_k = st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of relevant sections to retrieve"
        )

    # Main chat interface
    with st.container():
        # Display existing messages
        for message in state.messages:
            display_chat_message(message, state)

        # Chat input
        if prompt := st.chat_input("Ask about specific issues or sections...", key="chat_input"):
            if not state.office_action_text:
                st.error("Please upload an Office Action letter first!")
            else:
                # Add user message
                user_message = {"role": "user", "content": prompt}
                state.messages.append(user_message)
                
                # Process query
                # Process query
                with st.spinner("Searching relevant sections..."):
                    try:
                        segments, results = retrieve(
                            query=prompt,
                            methods=retrieval_method,
                            topK=top_k
                        )
                        
                        state.current_sections = segments
                        
                        # Create assistant response
                        assistant_message = {
                            "role": "assistant",
                            "content": "I found these relevant sections from the manual:",
                            "sections": [results]  # Wrap the results string in a list
                        }
                        state.messages.append(assistant_message)
                        
                        # Force refresh to show new messages and sections
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error during search: {str(e)}")
                        st.write("Error details:", e)

        # Generate response if sections are selected
        if state.selected_sections:
            if st.button("Generate Response", key="generate_response"):
                with st.spinner("Generating response..."):
                    try:
                        response = generate_response(
                            segments=state.current_sections,  # Using stored segments here
                            user_input=prompt
                        )
                        
                        st.markdown("### Generated Response")
                        for section, content in response.items():
                            st.markdown(f"#### {section}")
                            st.markdown(content)
                        
                        # Save and offer download options
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"response_{timestamp}"
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "Download as JSON",
                                data=json.dumps(response, indent=4),
                                file_name=f"{filename}.json",
                                mime="application/json",
                                key="download_json"
                            )
                        with col2:
                            st.download_button(
                                "Download as Text",
                                data="\n\n".join([f"{section}:\n{content}" 
                                                for section, content in response.items()]),
                                file_name=f"{filename}.txt",
                                mime="text/plain",
                                key="download_text"
                            )
                            
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.write("Error details:", e)

    # Debug information (optional)
    if st.checkbox("Show Debug Info", key="debug_info"):
        st.write("Current Retrieval Method:", retrieval_method)
        st.write("Number of Results:", top_k)
        st.write("Selected Sections:", len(state.selected_sections))
        st.write("Current Sections:", len(state.current_sections))
        st.write("Message History:", len(state.messages))

if __name__ == "__main__":
    main()