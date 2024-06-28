import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import os
import logging

from langchain_core.output_parsers import StrOutputParser


from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

os.environ["LANGCHAIN_TRACING_V2"] = "True"
os.environ["OPENAI_API_KEY"] = "sk-"

st.markdown("## Regulatory Knowledge Expert")

final_summary = ""

model = ChatOpenAI(temperature=0, model="gpt-4o")

def get_pdf_pages(file_path):
    """
    Load a PDF file and obtain text for each page.

    Parameters:
    - pdf_file_path: str, path to the PDF file.

    Returns:
    - List of strings, where each string is the text of a page.
    """
    # Initialize the PDFLoader with the given file path
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    return pages



def summarise(document):
    """
    Summarize the given document.

    Parameters:
    - document: str, the document to be summarized.

    Returns:
    - str, the summarized document.
    """
    # Define prompt
    prompt_template = """
    You are a regulatory reporing assistant. Assess this document and 
    provide a summary of the key details to be reported to the 
    regulatory body {document}. The summary should have bullet points and proper formatting with less than 200 words
    
    First we need to identify the names of all the reports to be generated. 
    Each document does may or may not have information regarding report names
    
    For each report, identify the reportable fields:
    Field name, Field value, Field format rules, Sample Data Reportable, Client data type, Mandatory
    
    """
    prompt = PromptTemplate.from_template(prompt_template)
    

    output_parser = StrOutputParser()

    chain = prompt | model | output_parser
    return chain.invoke({"document": document})


def combine_summaries(documents):
    """
    Combine multiple document summaries into a single summary.

    Parameters:
    - documents: List[str], the document summaries to be combined.

    Returns:
    - str, the combined summary of all documents.
    """
    logging.info("Combining Summaries")
    # Define prompt
    prompt_template = """
    You are a regulatory reporting assistant. Given these document summaries:
    {summaries}
    Combine them into a comprehensive summary that captures the key details from all documents.
    
    Make sure to include all relevant information give it in a table format with separation by report
    """
    
    # Join the document summaries into a single string for the prompt
    summaries_text = "\n\n".join(f"- {doc}" for doc in documents)
    prompt = prompt_template.format(summaries=summaries_text)
    
    # Assuming the existence of a PromptTemplate, model, and StrOutputParser setup similar to the `summarise` function
    prompt_instance = PromptTemplate.from_template(prompt)
    output_parser = StrOutputParser()
    chain = prompt_instance | model | output_parser
    
    # Invoke the chain with the combined summaries text
    combined_summary = chain.invoke({"summaries": summaries_text})
    return combined_summary


def summarise_pages(pages_text) -> str:
    """
    Summarize multiple pages of a PDF and combine the summaries.

    Parameters:
    - pages_text: List[str], texts of each page in the PDF.

    Returns:
    - str, combined summary of all pages.
    """
    summaries = []
    # Map Step: Summarize each page
    for i, page_text in enumerate(pages_text):
        with st.spinner(f"Summarizing page {i+1} of {len(pages_text)}..."):
            summaries.append(summarise(page_text))

    with st.expander("Summarized Documents"):
        st.markdown(summaries)

    # Reduce Step: Combine the summaries. This could be as simple as concatenating them,
    # or more complex logic could be applied to structure the combined summary.
    with st.spinner("Generating combined summary..."):
        combined_summary = combine_summaries(summaries)

    return combined_summary


if st.button("Summarize Policy"):
    with st.spinner("Extracting text from pdf..."):
        pages_text = get_pdf_pages("./policy_pdfs/main_1_cic-policy.pdf")
    final_summary = summarise_pages(pages_text)
    st.write(final_summary)