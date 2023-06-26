import os
import requests
import fnmatch
import argparse
import base64
import streamlit as st

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)
#streamlit configs and title
st.set_page_config(page_title="Github Automated Analysis App")
st.title('Github Automated Analysis App')
st.caption("""This is a Langchain + OpenAI app to anlayze and summarize any github project's README.MD all_files, you have to just enter the Repo's link and it would automatically scrape all the files(*.MD)  and summarize and explain those for you. Keep supporting!!""")

def parse_github_url(url):
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo

def get_files_from_github_repo(owner, repo, token):
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/main?recursive=1"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        return content["tree"]
    else:
        raise ValueError(f"Error fetching repo contents: {response.status_code}")

def fetch_md_contents(files):
    md_contents = []
    for file in files:
        if file["type"] == "blob" and fnmatch.fnmatch(file["path"], "*.md"):
            response = requests.get(file["url"])
            if response.status_code == 200:
                content = response.json()["content"]
                decoded_content = base64.b64decode(content).decode('utf-8')
                print("Fetching Content from ", file['path'])
                md_contents.append(Document(page_content=decoded_content, metadata={"source": file['path']}))
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return md_contents

def get_source_chunks(files):
    print("In get_source_chunks ...")
    source_chunks = []
    md_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0, length_function=len,
    )
    for source in fetch_md_contents(files):
        for chunk in md_splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadate=source.metadata))
    return source_chunks


def main():
    #openai_api_key='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'
    # taking user input
    openai_api_key = st.sidebar.text_input('OpenAI API key', type='password')
    username = st.text_input("Username: ", key="first")
    GITHUB_TOKEN = st.text_input("Github Personal Access Token")
    submit_button = st.button("Submit")
    
    if submit_button and username and GITHUB_TOKEN:  
        GITHUB_OWNER, GITHUB_REPO = parse_github_url(username)
        #GITHUB_TOKEN = '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'
        all_files = get_files_from_github_repo(GITHUB_OWNER, GITHUB_REPO, GITHUB_TOKEN)

        CHROMA_DB_PATH = f'./chroma/{os.path.basename(GITHUB_REPO)}'

        chroma_db = None

        if not os.path.exists(CHROMA_DB_PATH):
            print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
            source_chunks = get_source_chunks(all_files)
            chroma_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(openai_api_key=openai_api_key), persist_directory=CHROMA_DB_PATH)
            chroma_db.persist()
        else:
            print(f'Loading Chroma DB from {CHROMA_DB_PATH} ... ')
            chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key))


        qa_chain = load_qa_chain(OpenAI(temperature=1, openai_api_key=openai_api_key), chain_type="stuff")
        qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=chroma_db.as_retriever())

        prompt = "You will summarize and explain all the details of a project in a professional and easy to understand manner"
        answer = qa.run(prompt)
        st.response(answer)


if __name__ == "__main__":
    main()