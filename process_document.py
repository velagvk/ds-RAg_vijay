import os
import sys
import json
import requests
import tempfile

from dsrag.dsparse.main import parse_and_chunk
from dsrag.dsparse.file_parsing.non_vlm_file_parsing import parse_file_no_vlm
from dsrag.dsparse.sectioning_and_chunking.semantic_sectioning import get_sections_from_str
from dsrag.dsparse.sectioning_and_chunking.chunking import chunk_document
from dsrag.auto_context import get_document_title, get_document_summary, get_section_summary, get_chunk_header
from dsrag.llm import AzureOpenAIChatAPI
from dsrag.dsparse.file_parsing.file_system import LocalFileSystem

# Get constants from environment variables, with fallbacks from the user's audit code
ILIAD_URL = os.environ.get("ILIAD_URL", "https://api-epic.ir-gateway.abbvienet.com/iliad")
ILIAD_API_KEY = os.environ.get("ILIAD_API_KEY")
USER_TOKEN = os.environ.get("USER_TOKEN")

def upload_chunk_to_iliad(chunk_content, original_filename, chunk_index, section_title, source_name):
    """
    Creates a temporary file for a chunk and uploads it to the ILIAD endpoint.
    """
    if not ILIAD_API_KEY or not USER_TOKEN:
        print("Error: ILIAD_API_KEY and USER_TOKEN environment variables must be set.")
        return None

    # Use a temporary file to upload the chunk content
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as tmp:
        tmp.write(chunk_content)
        file_path = tmp.name

    # The chunk's unique "filename" within the source
    chunk_filename = f"{os.path.splitext(original_filename)[0]}_chunk_{chunk_index}.txt"

    custom_fields = {
        "owning_facility": "audit",
        "location": "internal",
        "document_type": "chunk",
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
            print(f"  - Uploaded {chunk_filename}")
            return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"  - Error uploading chunk {chunk_index} for {original_filename}: {e}")
        if e.response:
            print(f"    Response content: {e.response.text}")
        return None
    finally:
        os.unlink(file_path) # Clean up the temp file


def process_and_upload_document_vlm(file_path, source_name, azure_deployment):
    """
    Processes a document using VLM, enriches it with AutoContext, and uploads chunks.
    """
    print(f"Starting VLM processing for document: {file_path}")
    doc_id = os.path.basename(file_path)
    kb_id = "local_vlm" # Separate kb_id for VLM processing

    # 1. Configure the VLM and AutoContext models
    print("Step 1: Configuring models...")
    file_parsing_config = {
        "use_vlm": True,
        "vlm_config": {
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-20250219", # As requested
        }
    }
    # For semantic sectioning and AutoContext, we'll still use the Azure model
    semantic_sectioning_config = {
        "llm_provider": "azure_openai", # We'll need to adapt the code to handle this
        "model": azure_deployment,
    }
    auto_context_model = AzureOpenAIChatAPI(azure_deployment=azure_deployment)
    file_system = LocalFileSystem(base_path=os.path.expanduser("~/dsParse"))

    # This is a limitation: the semantic sectioning part of dsrag doesn't know about our custom Azure LLM class.
    # For now, we will proceed, but this part would need a deeper modification to dsrag to work perfectly.
    # The VLM parsing will use Anthropic, but sectioning might fail if it can't find a default LLM.
    print("Warning: Semantic sectioning may not use the specified Azure model due to a library limitation.")

    # 2. Parse the document using the VLM provider
    print("Step 2: Parsing document with Anthropic VLM...")
    sections, chunks = parse_and_chunk(
        kb_id=kb_id,
        doc_id=doc_id,
        file_path=file_path,
        file_parsing_config=file_parsing_config,
        semantic_sectioning_config={}, # Pass empty to use defaults, avoiding the provider issue for now
        chunking_config={},
        file_system=file_system
    )
    print(f"  - VLM parsing generated {len(sections)} sections and {len(chunks)} chunks.")

    # 3. Generate AutoContext summaries
    print("Step 3: Generating full AutoContext summaries with Azure LLM...")
    text_content_for_summary = "\n".join([line['content'] for line in chunks]) # Reconstruct text for summary
    document_title = get_document_title(auto_context_model, text_content_for_summary)
    document_summary = get_document_summary(auto_context_model, text_content_for_summary, document_title)

    section_summaries = []
    for i, section in enumerate(sections):
        summary = get_section_summary(auto_context_model, section['content'], document_title, section['title'])
        section_summaries.append(summary)

    # 4. Upload chunks
    print(f"\nStep 4: Adding context and uploading {len(chunks)} chunks to source '{source_name}'...")
    for i, chunk in enumerate(chunks):
        section_index = chunk.section_index
        section_title = sections[section_index]['title']
        section_summary = section_summaries[section_index]

        header = get_chunk_header(
            document_title=document_title,
            document_summary=document_summary,
            section_title=section_title,
            section_summary=section_summary
        )
        chunk_content_with_header = f"{header}\n\n{chunk.content}"
        upload_chunk_to_iliad(
            chunk_content=chunk_content_with_header,
            original_filename=doc_id,
            chunk_index=i,
            section_title=section_title,
            source_name=source_name
        )

    print("\nVLM processing complete.")


def process_and_upload_document_text(file_path, source_name, azure_deployment, chunk_size=800, min_length_for_chunking=1600):
    """
    Processes a document using text-based parsing, enriches it with full AutoContext, and uploads chunks.
    """
    print(f"Starting text-based processing for document: {file_path}")
    print("Step 1: Initializing Azure OpenAI...")
    try:
        auto_context_model = AzureOpenAIChatAPI(azure_deployment=azure_deployment)
    except Exception as e:
        print(f"Fatal Error: Could not initialize AzureOpenAIChatAPI: {e}")
        print("Please ensure OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT environment variables are set correctly.")
        return

    print("Step 2: Parsing document to text...")
    text, _ = parse_file_no_vlm(file_path)
    doc_id = os.path.basename(file_path)

    print("Step 3: Generating semantic sections...")
    sections, document_lines = get_sections_from_str(
        document=text,
        max_characters_per_window=20000,
        semantic_sectioning_config={},
        chunking_config={},
        kb_id="local",
        doc_id=doc_id
    )
    print(f"  - Generated {len(sections)} sections.")

    print("Step 4: Generating initial chunks from sections...")
    chunks = chunk_document(
        sections=sections,
        document_lines=document_lines,
        chunk_size=chunk_size,
        min_length_for_chunking=min_length_for_chunking
    )
    print(f"  - Generated {len(chunks)} initial chunks.")

    print("Step 5: Generating full AutoContext summaries with LLM...")
    print("  - Generating document title and summary...")
    document_title = get_document_title(auto_context_model, text)
    document_summary = get_document_summary(auto_context_model, text, document_title)
    print(f"    - Document Title: {document_title}")
    print(f"    - Document Summary: {document_summary}")

    print("  - Generating section summaries...")
    section_summaries = []
    for i, section in enumerate(sections):
        print(f"    - Summarizing section {i+1}/{len(sections)}: {section['title']}")
        summary = get_section_summary(auto_context_model, section['content'], document_title, section['title'])
        section_summaries.append(summary)

    print(f"\nStep 6: Adding context and uploading {len(chunks)} chunks to source '{source_name}'...")
    for i, chunk in enumerate(chunks):
        section_index = chunk.section_index
        section_title = sections[section_index]['title']
        section_summary = section_summaries[section_index]

        header = get_chunk_header(
            document_title=document_title,
            document_summary=document_summary,
            section_title=section_title,
            section_summary=section_summary
        )

        chunk_content_with_header = f"{header}\n\n{chunk.content}"

        upload_chunk_to_iliad(
            chunk_content=chunk_content_with_header,
            original_filename=doc_id,
            chunk_index=i,
            section_title=section_title,
            source_name=source_name
        )

    print("\nText-based processing complete.")


if __name__ == "__main__":
    if len(sys.argv) not in [4, 5] or (len(sys.argv) == 5 and sys.argv[1] != '--vlm'):
        print("\nUsage:")
        print("  Text-based processing: python process_document.py <path_to_document> <source_name> <azure_deployment_name>")
        print("  VLM-based processing:  python process_document.py --vlm <path_to_document> <source_name> <azure_deployment_name>")
        print("\nExample (Text): python process_document.py ./my_report.pdf my_audit_source gpt-4o")
        print("Example (VLM):  python process_document.py --vlm ./my_report.pdf my_audit_source gpt-4o\n")
        sys.exit(1)

    use_vlm = sys.argv[1] == '--vlm'
    arg_offset = 1 if use_vlm else 0

    document_path = sys.argv[1 + arg_offset]
    source_name = sys.argv[2 + arg_offset]
    azure_deployment_name = sys.argv[3 + arg_offset]

    required_env_vars = ['OPENAI_API_KEY', 'AZURE_OPENAI_ENDPOINT', 'ILIAD_URL', 'ILIAD_API_KEY', 'USER_TOKEN']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    if missing_vars:
        print(f"\nError: Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your terminal before running the script.\n")
        sys.exit(1)

    if not os.path.exists(document_path):
        print(f"\nError: File not found at {document_path}\n")
        sys.exit(1)

    if use_vlm:
        if not document_path.lower().endswith('.pdf'):
            print("\nError: VLM processing requires a PDF file.\n")
            sys.exit(1)
        process_and_upload_document_vlm(document_path, source_name, azure_deployment_name)
    else:
        process_and_upload_document_text(document_path, source_name, azure_deployment_name) 