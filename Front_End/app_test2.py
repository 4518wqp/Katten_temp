import streamlit as st
import sys
from pathlib import Path
import PyPDF2
from typing import List, Dict
import json
from datetime import datetime

# Basic setup
st.set_page_config(page_title="USPTO Rebuttal Assistant", page_icon="📝", layout="wide")

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
        self.current_query: str = None

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
    try:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
                <div style='background-color: #e3f2fd; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                    {content}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style='background-color: #f5f5f5; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;'>
                    {content}
                </div>
            """, unsafe_allow_html=True)
        
        if "sections" in message:
            display_sections(message["sections"], state)
        
        if "generated_response" in message:
            display_generated_response(message["generated_response"])
            
    except Exception as e:
        st.error(f"Error displaying message: {str(e)}")

def display_sections(sections: List[Dict], state: SessionState):
    try:
        st.markdown("### Retrieved Sections")
        if not sections:
            return
            
        content = sections[0] if isinstance(sections[0], str) else sections
        references = []
        
        # Handle string format from retriever
        if isinstance(content, str):
            references = content.split("- Reference[")
        # Handle list format
        elif isinstance(content, list):
            references = content
            
        for i, ref in enumerate(references, 1):
            if not ref.strip():
                continue
                
            try:
                if isinstance(ref, str):
                    # Parse reference content
                    ref_num = f"{i}"
                    subsection = ""
                    summary = ""
                    
                    if "- Subsection:" in ref and "- Summary of content:" in ref:
                        parts = ref.split("- Subsection:", 1)[1].split("- Summary of content:")
                        subsection = parts[0].strip()
                        summary = parts[1].strip() if len(parts) > 1 else ""
                    
                    with st.container():
                        col1, col2 = st.columns([0.1, 0.9])
                        
                        with col1:
                            key = f"select_{ref_num}_{subsection}"
                            selected = st.checkbox("Select", key=key)
                        
                        with col2:
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
                            """, unsafe_allow_html=True)
                        
                        if selected:
                            matching_section = next(
                                (s for s in state.current_sections if subsection in s['header']),
                                None
                            )
                            if matching_section and matching_section not in state.selected_sections:
                                state.selected_sections.append(matching_section)
                        else:
                            matching_section = next(
                                (s for s in state.current_sections if subsection in s['header']),
                                None
                            )
                            if matching_section and matching_section in state.selected_sections:
                                state.selected_sections.remove(matching_section)
            except Exception as e:
                st.error(f"Error processing reference {i}: {str(e)}")
                
    except Exception as e:
        st.error(f"Error displaying sections: {str(e)}")

def display_generated_response(response: Dict):
    st.markdown("### Generated Response")
    for section, content in response.items():
        with st.expander(f"Section: {section}", expanded=True):
            st.markdown(content)

def main():
    st.title("USPTO Office Action Response Assistant")
    state = init_session_state()

    # Initialize session state variables
    if 'retrieval_method' not in st.session_state:
        st.session_state.retrieval_method = "vector"
    if 'top_k' not in st.session_state:
        st.session_state.top_k = 5
    if 'initial_query' not in st.session_state:
        st.session_state.initial_query = None
    if 'awaiting_generation' not in st.session_state:
        st.session_state.awaiting_generation = False

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        
        uploaded_file = st.file_uploader(
            "Upload Office Action Letter (PDF)",
            type="pdf",
            help="Upload the USPTO Office Action letter you want to analyze"
        )
        
        if uploaded_file and not state.office_action_text:
            state.office_action_text = extract_text_from_pdf(uploaded_file)
            if state.office_action_text:
                st.success("✅ Office Action letter processed successfully!")
        
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

    # Chat interface
    chat_container = st.container()
    
    with chat_container:
        for message in state.messages:
            display_chat_message(message, state)

        # Chat input
        if prompt := st.chat_input("Ask about specific issues or sections...", key="chat_input"):
            if not state.office_action_text:
                st.error("Please upload an Office Action letter first!")
            else:
                user_message = {"role": "user", "content": prompt}
                state.messages.append(user_message)
                
                if not st.session_state.awaiting_generation:
                    # Initial retrieval
                    with st.spinner("Searching relevant sections..."):
                        try:
                            segments, results = retrieve(
                                query=prompt,
                                methods=retrieval_method,
                                topK=top_k,
                                filenumber="1200",
                                toc=35213
                            )
                            
                            state.current_sections = segments
                            st.session_state.initial_query = prompt
                            
                            if isinstance(results, str):
                                results = [results]
                                
                            assistant_message = {
                                "role": "assistant",
                                "content": "I found these relevant sections from the manual:",
                                "sections": results
                            }
                            state.messages.append(assistant_message)
                            st.session_state.awaiting_generation = True
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")
                else:
                    # Handle follow-up with selected sections
                    if state.selected_sections:
                        try:
                            # Convert selected sections to the correct format
                            print("Selected sections:", state.selected_sections)  # Debug
                            
                            response = generate_response(
                                segments=state.selected_sections,  # List of dicts with header/content
                                user_input=prompt
                            )
                            
                            assistant_message = {
                                "role": "assistant",
                                "content": "Based on the selected sections:",
                                "sections": results  # Keep the original sections
                            }
                            
                            # Add the generated response separately
                            state.messages.append({
                                "role": "assistant",
                                "content": "\n".join([f"### {section}\n{content}" 
                                                    for section, content in response.items()])
                            })
                            
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error generating response: {str(e)}")
                            st.write("Selected sections:", state.selected_sections)  # Debug
                    else:
                        st.warning("Please select relevant sections before asking follow-up questions.")

    # Generate final response button
    if state.selected_sections and st.session_state.awaiting_generation:
        col1, col2 = st.columns([0.7, 0.3])
        with col2:
            if st.button("Generate Final Response", key="generate_final"):
                with st.spinner("Generating final response..."):
                    try:
                        final_response = generate_response(
                            segments=state.selected_sections,
                            user_input=st.session_state.initial_query
                        )
                        
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"response_{timestamp}"
                        
                        st.markdown("### Final Response")
                        for section, content in final_response.items():
                            with st.expander(f"Section: {section}", expanded=True):
                                st.markdown(content)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.download_button(
                                "Download as JSON",
                                data=json.dumps(final_response, indent=4),
                                file_name=f"{filename}.json",
                                mime="application/json"
                            )
                        with col2:
                            st.download_button(
                                "Download as Text",
                                data="\n\n".join([f"{section}:\n{content}" 
                                                for section, content in final_response.items()]),
                                file_name=f"{filename}.txt",
                                mime="text/plain"
                            )
                        
                        st.session_state.awaiting_generation = False
                        
                    except Exception as e:
                        st.error(f"Error generating final response: {str(e)}")

    # Reset button
    if st.session_state.awaiting_generation:
        if st.button("Start New Search", key="reset"):
            st.session_state.awaiting_generation = False
            state.selected_sections = []
            state.current_sections = []
            st.session_state.initial_query = None
            st.rerun()

if __name__ == "__main__":
    main()