from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from pinecone import Pinecone
import os
import requests
import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
import time
from dotenv import load_dotenv
from st_clickable_images import clickable_images
import shutil
import sys
import nltk
import streamlit.components.v1 as components

def add_file_text_to_corpus(uploaded_files):
    with st.spinner('Please wait, your file(s) is/are being processed'):
        new_files= [uf_obj for uf_obj in uploaded_files if uf_obj not in st.session_state.ufs]
        appended_file_name=""
        for uploaded_file in new_files:
            file_name=uploaded_file.name
            appended_file_name=appended_file_name+"_"+file_name
        appended_file_name=appended_file_name[1:]
        appended_file_name=st.session_state.appended_file_name+"_"+appended_file_name
        print(appended_file_name)
        st.session_state.appended_file_name=appended_file_name

        root_dir= "uploaded_files"
        dir = f"{root_dir}/{appended_file_name}"
        if not os.path.isdir(dir):
            os.mkdir(dir)

        file_path_list=[]
        for uploaded_file in uploaded_files:
            file_name=uploaded_file.name
            file_path = os.path.join(dir, file_name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_path_list.append(f"{dir}/{file_name}")

@st.cache_data(ttl=3600)  # cache for 1 hour
def get_groq_models():
    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    # filter out non-chat models
    models = [
        m['id'] for m in data['data']
        if not any(x in m['id'] for x in ['whisper', 'orpheus', 'guard', 'safeguard', 'compound'])
    ]
    return sorted(models)

# Loading the api keys
load_dotenv()
# del os.environ['NVIDIA_API_KEY']  ## delete key and reset
os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-")
pinecone_api_key=os.getenv('PINECONE_API_KEY')

st.set_page_config(
    page_title="AskGroq",

    page_icon="icons/AG_black_4_3.png",  # or use a .png file
    # page_icon="icons/AG.png",  # or use a .png file

    # layout="wide"
    menu_items={
            #  'Get Help': 'https://www.extremelycoolapp.com/help',
             'Report a bug': "mailto:suvojeethaldar4@gmail.com",
            #  'About': "# This is a header. This is an *extremely* cool app!"
         }
) 

# Header
st.title("AskGroq")
st.write("Upload any file(s) & use the model of your choice to get answers from it.")


components.html("""
    <div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="https://i.postimg.cc/d15b1GQT/pdf.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/rwq7Shdt/word.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/mZHxR1fh/txt.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/mDpnRRVY/powerpoint.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/0jgNtgmf/excel.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/Yqr7tqpF/image-gallery.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/rFBDhrGT/csv.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/bw0w3FN0/epub.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/Cx6SPTpq/gmail.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/T2qG8WXG/html.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/PqGFr2m7/markdown.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/ZKs5gyrn/odt.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/Pr0kYbWW/org.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/P5bFJ12r/rst.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/50XhX6X7/rtf.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/G26ZGdYc/tsv.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/MHkFXt2H/xml.png" style="margin: 15px; height: 75px;">
    </div>
""", height=310)




# Upload widget
uploaded_files = st.file_uploader("Upload any file(s) & use the model of your choice to get answers from it.", accept_multiple_files=True, label_visibility="collapsed", type=['bmp', 'csv', 'doc', 'docx', 'eml', 'epub', 'heic', 'html', 'jpeg', 'jpg', 'png', 'md', 'msg', 'odt', 'org', 'p7s', 'pdf', 'png', 'ppt', 'pptx', 'rst', 'rtf', 'tiff', 'txt', 'tsv', 'xls', 'xlsx', 'xml'])

       
if 'retriever_or_corpus' not in st.session_state:
    if uploaded_files:
        st.session_state.ufs=uploaded_files
        with st.spinner('Please wait, your file(s) is/are being processed & uploaded to our secure vector db server'):
            # Process the uploaded file(s)
            appended_file_name=""
            for uploaded_file in uploaded_files:
                file_name=uploaded_file.name
                file_name= ''.join(i for i in file_name if ord(i) < 128)
                appended_file_name=appended_file_name+"_"+file_name
            appended_file_name=appended_file_name[1:]
            print("---------------------------------")
            print(appended_file_name)
            st.session_state.appended_file_name=appended_file_name

            root_dir= "uploaded_files"
            if not os.path.isdir(root_dir):
                os.mkdir(root_dir)

            dir = f"{root_dir}/{appended_file_name}"
            if not os.path.isdir(dir):
                os.mkdir(dir)

            file_path_list=[]
            for uploaded_file in uploaded_files:
                file_name=uploaded_file.name
                file_path = os.path.join(dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_path_list.append(f"{dir}/{file_name}")

            for file_path in file_path_list:
                if file_path.endswith(('.bmp', '.png', '.tiff', '.jpeg', '.heic', '.jpg')):
                   strategy= "hi_res"
                else:
                    strategy= "fast"

            print(strategy)

            server_url = os.getenv('UNSTRUCTURED_API_URL')
            loader_u = UnstructuredLoader(
                file_path = file_path_list,
                api_key=os.getenv('UNSTRUCTURED_API_KEY'),
                partition_via_api=True,
                chunking_strategy="by_title",
                strategy=f"{strategy}",
                url = os.getenv('UNSTRUCTURED_API_URL')
            )

            # loader_p= PyPDFDirectoryLoader(dir)

            docs= loader_u.load()
            
            # text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            # final_documents= text_splitter.split_documents(docs)

            excel_or_xml_flag=0

            corpus=[]
            for doc in docs:
                if doc.metadata['filename'].endswith(('.xlsx', '.xls', 'csv', 'tsv')):
                    corpus.append(doc.metadata['text_as_html'])
                    excel_or_xml_flag=1
                elif doc.metadata['filename'].endswith(('xml')):
                    corpus.append(doc.page_content)
                    excel_or_xml_flag=1
                else:
                    corpus.append(doc.page_content)

            # print(corpus)

            if excel_or_xml_flag==0:
                index_name="custom-gpt"
                # Iniialize the Pinecone client
                pc=Pinecone(api_key=pinecone_api_key)
                index=pc.Index(index_name)

                # use default tf-idf values
                bm25_encoder = BM25Encoder().default()
                nltk.download('punkt_tab')

                # fit tf-idf values on your corpus
                bm25_encoder.fit(corpus)

                # store the values to a json file
                bm25_encoder.dump(f"{dir}/bm25_values.json")

                # load to your BM25Encoder object
                bm25_encoder = BM25Encoder().load(f"{dir}/bm25_values.json")

                # vector embedding and sparse matrix
                embeddings=NVIDIAEmbeddings(model="nvidia/nv-embed-v1")
                
                if appended_file_name in index.describe_index_stats()['namespaces']:
                    index.delete(namespace=appended_file_name, delete_all=True)

                retriever=PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder, index=index, namespace=appended_file_name)

                retriever.add_texts(corpus, namespace=appended_file_name)

                st.session_state.retriever_or_corpus=retriever
            else:
                st.session_state.retriever_or_corpus=corpus
            
            # Delete the directory and its contents post corpus addition to the retriever
            st.session_state.excel_or_xml_flag=excel_or_xml_flag
            shutil.rmtree(dir)
            st.success("File(s) uploaded successfully!")


if 'retriever_or_corpus' in st.session_state:
    if uploaded_files != st.session_state.ufs:
        if st.session_state.excel_or_xml_flag==0:
            st.info('Please reload the page to add new files, as we cannot add new sparse_encoder to a retriever. It is added while instantiating hence you need to reload to proceed.')
        else:
            add_file_text_to_corpus(uploaded_files)

    st.write("Choose a Model:")
    models = get_groq_models()
    model = st.selectbox('Choose a Model:', models, label_visibility="collapsed")
    st.write(f'You selected: {model}')
    st.session_state.model = model

if 'model' in st.session_state:
    input_prompt= st.text_input(f"Enter Your Questions from the documents:")

    if input_prompt:
        start=time.process_time()
        llm = ChatGroq(
                model=st.session_state.model,
                groq_api_key=os.getenv('GROQ_API_KEY')
            )
        if st.session_state.excel_or_xml_flag==1:
            corpus=st.session_state.retriever_or_corpus
            messages = [
                            ("system", f"Answer the questions based on the provided context only. Please provide the most accurate response based on the question <context>{corpus}<context>"),
                            ("human", input_prompt),
                        ]
            
            try:
                response=llm.invoke(messages)
            except Exception as e:
                # Code to handle any exception
                print(f"An error occurred: {e}")
                st.info(f'We deeply regret to inform you that {st.session_state.model} is temporarily unavailable, please select another model to proceed.')
                st.stop()

            st.write(response.content)
            st.write("Response time :", time.process_time()-start)
            st.markdown(":green[If you are not satisfied with the answer, you can choose a different model.]")
        else:
            retriever=st.session_state.retriever_or_corpus
            prompt_template=ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question
            <context>
            {context}
            <context>
            Questions:{input}

            """
            )
            document_chain = create_stuff_documents_chain(llm, prompt_template)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            try:
                response=retrieval_chain.invoke({"input":input_prompt})
            except Exception as e:
                # Code to handle any exception
                print(f"An error occurred: {e}")
                st.info(f'We deeply regret to inform you that {st.session_state.model} is temporarily unavailable, please select another model to proceed.')
                st.stop()

            st.write(response['answer'])
        
            st.write("Response time :", time.process_time()-start)

            # With a streamlit expander 
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            st.markdown(":green[If you are not satisfied with the answer, you can choose a different model.]")


footer_html = """
<div style='text-align: center;'>
  <p>Developed by Suvojeet Haldar</p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)