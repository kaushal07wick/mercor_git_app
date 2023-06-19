#basic imports
import requests
import base64
import os
from dotenv import load_dotenv
import streamlit as st

# Import related to openAI and langchain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

#streamlit configs and title
st.set_page_config(page_title="Github Automated Analysis App")
st.title('Github Automated Analysis App')

#function to extract the repository names from the Github username or URL
def get_github_repo_names(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    
    if response.status_code == 200:
        json_data = response.json()
        repo_names = [repo["name"] for repo in json_data]
        return repo_names
    else:
        raise ValueError(f"Error fetching GitHub repositories: {response.status_code}")


#function to extract all the files from a repository
def get_files_from_github_repo(owner, repo, token):
    file_urls =[]
    
    headers = {
        "Authorization": f"token {token}",    #authorization using the github rest api token
        "Accept": "application/vnd.github+json"
    }
    for repo_name in repo:
        url = f"https://api.github.com/repos/{owner}/{repo_name}/git/trees/main?recursive=1"
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            content = response.json()
            tree = content["tree"]
            for item in tree:
                if item["type"] == "blob":   #we are accessing here all types of files
                    file_urls.append(item["url"])
            return content["tree"]
    else:
        raise ValueError(f"Error fetching repo contents: {response.status_code}")
    return file_urls


#function to fetch all the file content from each file inside a repo
def fetch_all_contents(files):
    all_contents = []
    for file in files:
        if file["type"] == "blob":
            response = requests.get(file["url"])
            response = requests.get(file["url"])
            if response.status_code == 200:
                content = response.json()["content"]
                try:
                    decoded_content = base64.b64decode(content).decode('utf-8')
                    print("Fetching content from", file['path'])
                    all_contents.append(Document(page_content=decoded_content, metadata={"source": file['path']}))
                except UnicodeDecodeError:
                    print(f"Error decoding file {file['path']}: Unable to decode content as UTF-8 .... Skipping (it's either image or logo)")   #we are ignoring these files and decode other, more important ones (code files)
            else:
                print(f"Error downloading file {file['path']}: {response.status_code}")
    return all_contents


#Memory Management
#here, this function chunks or breaks the files in small parts to be easy to ingest in the database and also the LLM can easily analyze it
def get_source_chunks(files):
    print("In get source chunks ...")
    source_chunks=[]
    splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in fetch_all_contents(files):
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    return source_chunks


#main function     
def main():
    # taking user input
    # taking user input
    with st.form('theform'):
        username = st.text_input("Enter the Username: ")
        submitted = st.form_submit_button('Submit')
    with st.form("the_decond-form"):
        GITHUB_TOKEN = st.text_input("Enter GitHub Token ")
        submitted1 = st.form_submit_button("Submit")
    
    repo_names = get_github_repo_names(username) #username oR URL

    ans = get_files_from_github_repo(username, repo_names, GITHUB_TOKEN) #extracting out the files

    #creating a chroma_db path
    CHROMA_DB_PATH = f'./chroma/{os.path.basename("Code-Data-1")}'
    
    chroma_db = None
    
    #checking if the database is not present already, if not then create new one
    if not os.path.exists(CHROMA_DB_PATH):
        print(f'Creating Chroma DB at {CHROMA_DB_PATH}...')
        source_chunks = get_source_chunks(ans) #get the chunked data
        chroma_db = Chroma.from_documents(source_chunks, OpenAIEmbeddings(), persist_directory=CHROMA_DB_PATH)  #embedding the data using OpenAI embedding
    else:
        print(f'Loading Chroma DB from {CHROMA_DB_PATH} ... ')
        chroma_db = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=OpenAIEmbeddings()) #loading the data from ChromDB

    prompt="Develop an algorithm to analyze a vector database of repositories and identify the most technically complex repository. The complexity should be determined based on the size and structure of the codebase, the number of dependencies, algorithmic complexity, integration and interoperability requirements, performance and optimization considerations, as well as the complexity of the domain. The repositories are represented in the vector database using OpenAI embeddings, and there are no specific language preferences. Once identified, provide the link to the repository."

    #we will use the RetrievalQA chain to prompt the GPT to answer according to our need
    qa_chain = load_qa_chain(OpenAI(temperature=1), chain_type="stuff")
    qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=chroma_db.as_retriever())
    answer = qa.run(prompt) #the answer
    st.response(answer)


#runner code
if __name__ == "__main__":
    main()