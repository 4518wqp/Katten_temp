import re
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage

# Initialize chat history
chat_history = []

def extract_sections_from_input(user_input):
    """Extract section references from user input."""
    # Updated regex to allow for more than three digits before the decimal point
    return re.findall(r'\d{1,4}\.\d{2}\([a-z]\)(?:\(\w+\))?', user_input)

def find_section_content(segments, section_refs):
    """Find content for specified section references in segments."""
    section_content = {}
    for ref in section_refs:
        for segment in segments:
            if ref in segment['header']:
                section_content[ref] = segment['content']
                break  # Stop searching once a match is found for this section
    return section_content

# Define the ChatPromptTemplate
template = """
Based on the following data, generate a response addressing the specified legal issue:
{content}

Response (format):
- Section: {section}
- Response:
"""

# Chat model
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def generate_response(segments, user_input):
    """
    Generate response based on segments and user input.
    
    Args:
        segments (list): List of dictionaries containing section information
        user_input (str): User's query or input
    
    Returns:
        dict: Generated responses for each section
    """
    global chat_history  # Use global variable to maintain chat history
    
    # Add user input to chat history
    chat_history.append({"role": "user", "content": user_input})
    
    sections = extract_sections_from_input(user_input)
    section_content = find_section_content(segments, sections)
    
    responses = {}
    
    if section_content:  # If specific sections are found
        for section, content in section_content.items():
            prompt_text = template.format(section=section, content=content)
            messages = chat_history + [SystemMessage(content=prompt_text)]  # Include chat history
            response = llm(messages=messages)
            responses[section] = response.content
            chat_history.append({"role": "assistant", "content": response.content})  # Add response to chat history
    else:
        # Handle follow-up questions with a focus on elaboration
        prompt_text = "You asked a follow-up question based on previous responses. Please provide an updated response with more details."
        messages = chat_history + [SystemMessage(content=prompt_text)]
        response = llm(messages=messages)
        responses["follow_up"] = response.content
        chat_history.append({"role": "assistant", "content": response.content})
    
    return responses
 