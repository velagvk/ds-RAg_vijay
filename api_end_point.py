# This is the code for your 'process-pdf-vlm' endpoint in Dataiku.

import dataiku
from dataiku import Folder
import os
import json
import tempfile
import logging

# --- Imports from our Git repository ---
# These work because we imported the repo into the project library
from dsrag.dsparse.main import parse_and_chunk
from dsrag.dsparse.file_parsing.file_system import LocalFileSystem
from dsrag.llm import AzureOpenAIChatAPI
from dsrag.auto_context import get_document_title, get_document_summary, get_section_summary, get_chunk_header
import requests # Make sure 'requests' is in your code env

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper function to upload chunks ---
# This is the same logic from our previous script
def upload_chunk_to_iliad(chunk_content, original_filename, chunk_index, section_title, source_name):
    ILIAD_URL = os.environ.get("ILIAD_URL")
    ILIAD_API_KEY = os.environ.get("ILIAD_API_KEY")
    USER_TOKEN = os.environ.get("USER_TOKEN")

    if not all([ILIAD_URL, ILIAD_API_KEY, USER_TOKEN]):
        logging.error("ILIAD environment variables are not set in the API node.")
        return None
        
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as tmp:
        tmp.write(chunk_content)
        file_path = tmp.name

    chunk_filename = f"{os.path.splitext(original_filename)[0]}_chunk_{chunk_index}.txt"
    custom_fields = {
        "original_document": original_filename,
        "chunk_index": chunk_index,
        "section_title": section_title
    }

    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                url=f"{ILIAD_URL}/api/v1/sources/{source_name}/documents",
                headers={"x-api-key": ILIAD_API_KEY, "x-user-token": USER_TOKEN},
                files={"file": (chunk_filename, f, 'text/plain')},
                params={"custom_fields": json.dumps(custom_fields)}
            )
            resp.raise_for_status()
            logging.info(f"Successfully uploaded chunk {chunk_index} for {original_filename}")
            return resp.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error uploading chunk {chunk_index}: {e}")
        if e.response:
            logging.error(f"Response content: {e.response.text}")
        return None
    finally:
        os.unlink(file_path)

# --- Main Function Called by Dataiku ---
# The name 'process_and_upload' must match what you entered in the "Settings" tab.

def process_and_upload(request, body, **kwargs):
    """
    Main endpoint function. It receives the uploaded file, processes it using the 
    VLM pipeline, and uploads the resulting chunks.
    """
    # 1. Get uploaded file and parameters
    try:
        uploaded_file = request.files['file']
        source_name = request.args.get('source_name')
        azure_deployment = request.args.get('azure_deployment')

        if not all([uploaded_file, source_name, azure_deployment]):
            return {"error": "Missing required parameters: 'file' (in form data), 'source_name', and 'azure_deployment' (in query string)."}
            
    except KeyError:
        return {"error": "No 'file' provided in the multipart/form-data request."}

    # 2. Set up Dataiku-specific file handling
    # The 'folders' variable is globally available in the endpoint, as per Dataiku docs
    # It contains the paths to the managed folders you linked in the UI.
    managed_folder_path = folders[0]
    dataiku_file_system = LocalFileSystem(base_path=managed_folder_path)
    
    # Save the uploaded file to a temporary path that dsrag can access
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.stream.read())
        temp_pdf_path = tmp.name

    # 3. Configure and run the VLM pipeline
    try:
        logging.info(f"Starting VLM processing for {uploaded_file.filename}...")
        
        file_parsing_config = {
            "use_vlm": True,
            "vlm_config": {
                "provider": "anthropic",
                "model": "claude-3-7-sonnet-20250219"
            }
        }

        # The core call to the dsrag library
        sections, chunks = parse_and_chunk(
            kb_id="dss_vlm_project", # An arbitrary ID for folder organization
            doc_id=uploaded_file.filename,
            file_path=temp_pdf_path,
            file_parsing_config=file_parsing_config,
            file_system=dataiku_file_system
        )
        
        logging.info(f"Generated {len(sections)} sections and {len(chunks)} chunks.")

        # 4. Run AutoContext Summarization
        logging.info("Generating AutoContext summaries with Azure OpenAI...")
        auto_context_model = AzureOpenAIChatAPI(azure_deployment=azure_deployment)
        
        # Reconstruct the full text for the document-level summary
        full_text = "\n".join(s['content'] for s in sections)
        
        document_title = get_document_title(auto_context_model, full_text)
        document_summary = get_document_summary(auto_context_model, full_text, document_title)

        section_summaries = [
            get_section_summary(auto_context_model, s['content'], document_title, s['title'])
            for s in sections
        ]

        # 5. Enrich and upload each chunk
        logging.info(f"Enriching and uploading {len(chunks)} chunks...")
        upload_count = 0
        for i, chunk in enumerate(chunks):
            header = get_chunk_header(
                document_title=document_title,
                document_summary=document_summary,
                section_title=sections[chunk.section_index]['title'],
                section_summary=section_summaries[chunk.section_index]
            )
            content_with_header = f"{header}\n\n{chunk.content}"
            
            result = upload_chunk_to_iliad(content_with_header, uploaded_file.filename, i, sections[chunk.section_index]['title'], source_name)
            if result:
                upload_count += 1

        # 6. Return the final result
        return {
            "status": "success",
            "original_filename": uploaded_file.filename,
            "sections_found": len(sections),
            "chunks_generated": len(chunks),
            "chunks_uploaded": upload_count
        }

    except Exception as e:
        logging.error(f"An error occurred during processing: {str(e)}")
        return {"error": str(e)}
        
    finally:
        # Clean up the temporary PDF file
        os.unlink(temp_pdf_path)
