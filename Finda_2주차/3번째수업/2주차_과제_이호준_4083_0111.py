import streamlit as st
import tempfile
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyMuPDFLoader, PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline

# 1. 문서 로드 및 분할

@st.cache_data
def load_and_split_docs(pdf_file, _embeddings):
    # Streamlit 업로드 파일을 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name
    
    try:
        loader = PDFPlumberLoader(tmp_path)
        docs = loader.load()
        semantic_splitter = SemanticChunker(
            _embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
        return semantic_splitter.split_documents(docs)
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)



# 2. 벡터스토어 생성

@st.cache_resource
def create_vectorstore(split_docs, _embeddings):
    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=_embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 1})


# 3. LLM 생성

@st.cache_resource
def create_llm():
    pipe = pipeline(
    "text-generation",
    model="LiquidAI/LFM2.5-1.2B-Instruct",
    trust_remote_code=True,
    max_length=512,
    temperature=0.1,
    do_sample=True,
    )
    return HuggingFacePipeline(pipeline=pipe)


# 4. 프롬프트
prompt = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Use the following context to answer the question accurately.
If the answer is not in the context, say '모르겠습니다.'
Answer in Korean.

#Question:
{question}

#Context:
{context}

#Answer:"""
)



# 5. Streamlit UI

st.title("📄 무료 RAG 문서 질의 시스템")
st.write("PDF 문서를 기반으로 질문을 입력하면 답변을 생성합니다.")

# PDF 업로드
uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file:
    with st.spinner("임베딩 모델 로드 중..."):
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-distilbert-cos-v1")
    with st.spinner("문서 로드 및 분할 중..."):
        split_docs = load_and_split_docs(uploaded_file, embeddings)
    with st.spinner("벡터스토어 생성 중..."):
        retriever = create_vectorstore(split_docs, embeddings)
    llm = create_llm()
    
    # RAG 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    question = st.text_input("질문을 입력하세요:")
    if question:
        with st.spinner("답변 생성 중..."):
            answer = chain.invoke(question)
        st.subheader("답변")
        st.write(answer)