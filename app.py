from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from pinecone import Pinecone
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import time
from dotenv import load_dotenv
from st_clickable_images import clickable_images
import shutil
import sys
import nltk
import streamlit.components.v1 as components

# Loading the api keys
load_dotenv()
# del os.environ['NVIDIA_API_KEY']  ## delete key and reset
if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
pinecone_api_key=os.getenv('PINECONE_API_KEY')

   
# Header
st.title("AskGroq")
st.write("Upload any file(s) & use the model of your choice to get answers from it.")


components.html("""
    <div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="https://i.postimg.cc/rFBDhrGT/csv.png" title="Image #0" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/bw0w3FN0/epub.png" title="Image #1" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/0jgNtgmf/excel.png" title="Image #2" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/Cx6SPTpq/gmail.png" title="Image #3" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/T2qG8WXG/html.png" title="Image #4" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/Yqr7tqpF/image-gallery.png" title="Image #5" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/PqGFr2m7/markdown.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/ZKs5gyrn/odt.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/Pr0kYbWW/org.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/d15b1GQT/pdf.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/mDpnRRVY/powerpoint.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/P5bFJ12r/rst.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/50XhX6X7/rtf.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/G26ZGdYc/tsv.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/mZHxR1fh/txt.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/rwq7Shdt/word.png" style="margin: 15px; height: 75px;">
    <img src="https://i.postimg.cc/MHkFXt2H/xml.png" style="margin: 15px; height: 75px;">
    </div>
""", height=310)




# Upload widget
uploaded_files = st.file_uploader("", accept_multiple_files=True, type=['bmp', 'csv', 'doc', 'docx', 'eml', 'epub', 'heic', 'html', 'jpeg', 'png', 'md', 'msg', 'odt', 'org', 'p7s', 'pdf', 'png', 'ppt', 'pptx', 'rst', 'rtf', 'tiff', 'txt', 'tsv', 'xls', 'xlsx', 'xml'])

       
if 'retriever' not in st.session_state:
    if uploaded_files:
        st.session_state.ufs=uploaded_files
        with st.spinner('Please wait, your file(s) is/are being processed & uploaded to our secure vector db server'):
            # Process the uploaded file(s)
            appended_file_name=""
            for uploaded_file in uploaded_files:
                file_name=uploaded_file.name
                appended_file_name=appended_file_name+"_"+file_name
            appended_file_name=appended_file_name[1:]
            print(appended_file_name)
            st.session_state.appended_file_name=appended_file_name

            dir = appended_file_name
            os.mkdir(dir)

            file_path_list=[]
            for uploaded_file in uploaded_files:
                file_name=uploaded_file.name
                file_path = os.path.join(dir, file_name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                file_path_list.append(f"{dir}/{file_name}")

            server_url = os.getenv('UNSTRUCTURED_API_URL')
            loader_u = UnstructuredLoader(
                file_path = file_path_list,
                api_key=os.getenv('UNSTRUCTURED_API_KEY'),
                partition_via_api=True,
                chunking_strategy="by_title",
                strategy="fast",
                url = os.getenv('UNSTRUCTURED_API_URL')
            )

            loader_p= PyPDFDirectoryLoader(dir)

            docs= loader_u.load()
            
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents= text_splitter.split_documents(docs)
            corpus = [doc.page_content for doc in final_documents]

            # print(corpus)

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
            
            # Delete the directory and its contents post corpus addition to the retriever
            shutil.rmtree(dir)
            

            st.session_state.retriever=retriever
            st.success("File(s) uploaded successfully!")


if 'retriever' in st.session_state:
    if uploaded_files != st.session_state.ufs:
        st.info('Please reload the page to add new files, as we cannot add new sparse_encoder to a retriever. It is added while instantiating hence you need to reload to proceed.')

    st.write("Choose a Developer:")
    clicked = clickable_images(
        [
            "https://i.postimg.cc/fRkjg9x0/meta-Large.png",
            "https://i.postimg.cc/T1RnjLd1/google.png",
            "https://i.postimg.cc/sfcYjcmh/groq-black.png",
            "https://i.postimg.cc/d3HZ85NN/mistral.png"
        ],
        titles=[f"Image #{str(i)}" for i in range(6)],
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap", "id": "responsiveDiv"},
        img_style={"margin": "15px", "height": "75px"},
    )

    # Define options for the first dropdown
    Developers = ['Meta', 'Google', 'Groq', 'Mistral']
    Developer=Developers[clicked]

    # Define options for the second dropdown based on the first selection
    if Developer == 'Meta':
        models = ['llama-3.1-70b-versatile', 'llama-3.1-8b-instant', 'llama-guard-3-8b', 'llama3-70b-8192', 'llama3-8b-8192']
    elif Developer == 'Google':
        models = ['gemma2-9b-it', 'gemma-7b-it']
    elif Developer == 'Groq':
        models = ['llama3-groq-70b-8192-tool-use-preview', 'llama3-groq-8b-8192-tool-use-preview']
    elif Developer == 'Mistral':
        models = ['mixtral-8x7b-32768']

    # Create the second dropdown
    model = st.selectbox(f'Choose a Model from {Developer} :', models)

    # Display the selected options
    st.write(f'You selected: {Developer} - {model}')
    st.session_state.model=model

if 'model' in st.session_state:
    input_prompt= st.text_input(f"Enter Your Questions from the documents:")

    if input_prompt:
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
        groq_api_key =os.getenv('GROQ_API_KEY')
        model=st.session_state.model
        llm=ChatGroq(groq_api_key=groq_api_key,
                    model_name=model)
        start=time.process_time()
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        
        try:
            retriever=st.session_state.retriever
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            response=retrieval_chain.invoke({"input":input_prompt})

            st.write(response['answer'])
            
            st.write("Response time :", time.process_time()-start)

            # With a streamlit expander 
            with st.expander("Document Similarity Search"):
                # Find the relevant chunks
                for i, doc in enumerate(response["context"]):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
            st.markdown(":green[If you are not satisfied with the answer, you can choose a different model or a developer.]")
        except AttributeError:
            # Code to handle the exception
            st.markdown(":red[Please first embed document(s) before asking questions.]")