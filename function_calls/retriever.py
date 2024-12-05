from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
import os
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# Access the API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

def get_trademark_guide_path(filenumber="1200"):
    """Get the absolute path to the Trademark Guidebook PDF."""
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent.absolute()
    guide_path = project_root / "Trademark Guidebook" / f"tmep-{filenumber}.pdf"
    
    if not guide_path.exists():
        raise FileNotFoundError(f"Trademark Guide not found at: {guide_path}")
    
    return str(guide_path)

def parse_subsections(filenumber="1200", toc=35213):
    """Parse PDF into subsections with detailed error handling."""
    try:
        file_path = './parsed_segments/'+filenumber+'.txt'
        
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                segments = eval(file.read())
        else:
            filename = get_trademark_guide_path(filenumber)
            print(f"Opening PDF file: {filename}")  # Debug print
            
            # Load PDF
            with pdfplumber.open(filename) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""

            if not text:
                raise ValueError("No text extracted from PDF")
            
            print(f"Total text length: {len(text)}")  # Debug print

            # Process table of contents
            table_of_content = text[:toc]
            content_lines = table_of_content.split('\n')
            temp = str(int(filenumber.strip('0')))
            toc_sections = [line for line in content_lines if line.strip().startswith(temp)]
            
            print(f"Found {len(toc_sections)} TOC sections")  # Debug print

            if not toc_sections:
                raise ValueError("No sections found starting with "+temp)

            # Process sections
            segments = []
            main_text = text[toc:]

            # Process sections with overlap to avoid missing content
            for i in range(len(toc_sections)):
                current_section = toc_sections[i]
                print(f"Processing section {i + 1}: {current_section[:50]}...")  # Debug print

                try:
                    # For all sections except the last one
                    if i < len(toc_sections) - 1:
                        next_section = toc_sections[i + 1]
                        start_pattern = re.escape(current_section)
                        end_pattern = re.escape(next_section)
                        
                        # Find content between current and next section
                        match = re.search(f'({start_pattern})(.*?)({end_pattern})', main_text, re.DOTALL)
                        
                        if match:
                            content = match.group(1) + match.group(2)  # Include section header and content
                            segments.append({
                                'header': current_section.strip(),
                                'content': content.strip()
                            })
                            print(f"Successfully processed section {i + 1}")  # Debug print
                        else:
                            print(f"No match found for section {i + 1}")  # Debug print
                    
                    # For the last section
                    else:
                        start_pattern = re.escape(current_section)
                        match = re.search(f'({start_pattern})(.*?)$', main_text, re.DOTALL)
                        
                        if match:
                            content = match.group(1) + match.group(2)
                            segments.append({
                                'header': current_section.strip(),
                                'content': content.strip()
                            })
                            print("Successfully processed last section")  # Debug print
                        else:
                            print("No match found for last section")  # Debug print

                except Exception as section_error:
                    print(f"Error processing section {i + 1}: {str(section_error)}")  # Debug print
                    continue

            if not segments:
                raise ValueError("No segments were successfully parsed")

            print(f"Successfully parsed {len(segments)} segments")  # Debug print

            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as file:
                file.write(str(segments))
        return segments

    except Exception as e:
        print(f"Error in parse_subsections: {str(e)}")  # Debug print
        raise Exception(f"Error parsing subsections: {str(e)}")

import shutil

def create_vectordb(segments, filename):
    """Create vector database with error handling."""
    try:
        if not segments:
            raise ValueError("No segments provided for vector database creation")

        persist_directory = './chroma_local_db/'+filename
        collection_name = "Katten_Embed"+filename
        embeddings = OpenAIEmbeddings()
        
        if os.path.isdir(persist_directory) and bool(os.listdir(persist_directory)):
            existing_db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
            )

            return existing_db
        else:
            os.makedirs(persist_directory, exist_ok=True)
            os.chmod(persist_directory, 0o777)
            
            vector_db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory)
            vector_db.delete_collection()
            vector_db = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=persist_directory)

            for segment in segments:
                content = segment['content']
                metadata = {'header': segment['header']}
                vector_db.add_texts([content], metadatas=[metadata])

            vector_db.persist()

            return vector_db
    except Exception as e:
        print(f"Error in create_vectordb: {str(e)}")  # Debug print
        raise Exception(f"Error creating vector database: {str(e)}")

def retrieve(filenumber="1200", toc=35213, topK=5, query="", methods="vector"):
    """Retrieve relevant sections with enhanced error handling."""
    try:
        print(f"Starting retrieval with method: {methods}")  # Debug print
        
        # Get segments
        segments = parse_subsections(filenumber, toc)
        if not segments:
            raise ValueError("No segments found in document")

        print(f"Creating vector database for {len(segments)} segments")  # Debug print
        
        # Create vector database
        vector_db = create_vectordb(segments, filenumber)

        # Process based on method
        result = None
        if methods == "vector":
            result = retrieve_subsection_vector(topK, vector_db, query)
        elif methods == "BM25":
            result = retrieve_subsection_bm25(topK, vector_db, query)
        elif methods == "RFF":
            result = retrieve_subsection_rff(topK, vector_db, query)
        else:
            raise ValueError(f"Unknown retrieval method: {methods}")

        if not result:
            raise ValueError("No results returned from retrieval method")

        print("Retrieval completed successfully")  # Debug print
        return segments, result

    except Exception as e:
        print(f"Error in retrieve: {str(e)}")  # Debug print
        raise Exception(f"Retrieval error: {str(e)}")

def retrieve_subsection_vector(topK=5, vector_db=None, query=""):
    """Vector-based retrieval with error handling."""
    try:
        if not vector_db:
            raise ValueError("Vector database is not initialized")

        if not query:
            raise ValueError("Query is empty")

        template = """
        Based on the following data, for **EACH Document**:
        - Directly extract the 'header' noted as `title`, **without ANY modification**
        - Summarize the 'content', noted as `summary`

        You **MUST Exactly** follow the output format below:
        - Reference[1, 2, 3, 4, 5, ...]: 
            - Subsection: `title`
            - Summary of content: `summary`

        {context}
        """
        prompt = ChatPromptTemplate.from_template(template)
        
        # Set up retriever with bounds checking
        search_k = max(1, min(topK, 10))
        retriever = vector_db.as_retriever(search_kwargs={"k": search_k})
        
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        output = rag_chain.invoke(query)
        
        if not output:
            raise ValueError("No output generated from the language model")

        return output

    except Exception as e:
        print(f"Error in retrieve_subsection_vector: {str(e)}")  # Debug print
        raise Exception(f"Vector retrieval error: {str(e)}")

# Retrieve documents using BM25 scoring with vector database content
def retrieve_subsection_bm25(topK = 5, vector_db = None, query = ""):

    top_k = topK

    # Step 1: Retrieve all documents from vector_db (or a subset based on initial filter criteria)
    # Retrieve documents based on some criteria, such as all documents or initial vector similarity (if needed)
    all_docs = vector_db.similarity_search(query, k=top_k * 2)  # Retrieve more than top_k for better results

    # Step 2: Prepare corpus from the retrieved documents
    corpus = [doc.page_content for doc in all_docs]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)  # Initialize BM25 on the corpus

    # Tokenize the query
    tokenized_query = query.split()

    # Step 3: Calculate BM25 scores
    scores = bm25.get_scores(tokenized_query)
    top_indices = scores.argsort()[-top_k:][::-1]  # Sort and get top_k indices

    # Format results with metadata and scores
    results = [{'header': all_docs[i].metadata['header'], 'content': all_docs[i].page_content} for i in top_indices if scores[i] > 0]

    template = """
    Based on the following data, for **EACH Document**:
    - Directly extract the 'header' noted as `title`, **without ANY modification**
    - Summarize the 'content', noted as `summary`

    You **MUST Exactly** follow the output format below, or you will be punished: 
    - Reference[1, 2, 3, 4, 5, ...]: 
        - Subsection: `title`
        - Summary of content: `summary`

    {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Use LLMChain to generate output
    chain = LLMChain(prompt=prompt, llm=llm)

    # Provide the context for the chain
    output = chain.run(context=results)

    return output

def retrieve_subsection_rff(topK = 5, vector_db = None, query = ""):

    embeddings = OpenAIEmbeddings()

    rff = RBFSampler()  # Customize gamma as needed
    top_k = topK
    alpha = 0.7

    all_docs = vector_db.similarity_search(query, k=top_k * 2)
    corpus = [doc.page_content for doc in all_docs]
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # BM25 scores
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Step 2: Transform query and documents with RFF
    query_embedding = embeddings.embed_query(query)  # Use embed_query for the query
    query_embedding_normalized = normalize(np.array(query_embedding).reshape(1, -1))
    rff.fit(query_embedding_normalized)
    query_embedding_rff = rff.transform(query_embedding_normalized).reshape(1, -1)

    rff_scores = []
    for doc in all_docs:
        # Use embed_query to embed the document content if embed_document is not available
        doc_embedding = embeddings.embed_query(doc.page_content)
        doc_embedding_normalized = normalize(np.array(doc_embedding).reshape(1, -1))
        doc_embedding_rff = rff.transform(doc_embedding_normalized).reshape(1, -1)
        cosine_sim = cosine_similarity(query_embedding_rff, doc_embedding_rff)[0][0]
        rff_scores.append(cosine_sim)

    # Step 3: Combine BM25 and RFF scores
    combined_scores = [
        (doc.metadata['header'], doc.page_content, alpha * bm25_scores[i] + (1 - alpha) * rff_scores[i])
        for i, doc in enumerate(all_docs)
    ]

    # Sort by combined score
    sorted_results = sorted(combined_scores, key=lambda x: x[2], reverse=True)[:top_k]

    results = [{'header': i[0], 'content': i[1]} for i in sorted_results]

    template = """
    Based on the following data, for **EACH Document**:
    - Directly extract the 'header' noted as `title`, **without ANY modification**
    - Summarize the 'content', noted as `summary`

    You **MUST Exactly** follow the output format below, or you will be punished: 
    - Reference[1, 2, 3, 4, 5, ...]: 
        - Subsection: `title`
        - Summary of content: `summary`

    {context}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Use LLMChain to generate output
    chain = LLMChain(prompt=prompt, llm=llm)

    # Provide the context for the chain
    output = chain.run(context=results)

    return output