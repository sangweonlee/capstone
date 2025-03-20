import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from io import BytesIO

# 환경 변수 로드
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# 1. 업로드된 PDF 문서 로드
def load_pdf_from_upload(uploaded_file):
    if uploaded_file is not None:
        # Streamlit의 UploadedFile 객체를 PyPDFLoader가 읽을 수 있는 파일로 변환
        bytes_data = uploaded_file.read()
        with open("temp.pdf", "wb") as f:
            f.write(bytes_data)
        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()
        return documents
    return None

# 2. 텍스트 분할
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# 3. 벡터 스토어 생성
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# 4. Gemini를 사용한 답변 생성
def get_gemini_response(question, context):
    model = genai.GenerativeModel('gemini-2.0-flash-001')  # 모델명은 상황에 따라 변경
    prompt = f"다음 질문에 대해 주어진 문서 내용을 기반으로 답변하세요.\n문서 내용: {context}\n질문: {question}"
    response = model.generate_content(prompt)
    return response.text

# 5. 챗봇 설정
def setup_chatbot(vector_store):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    def chat_with_memory(question):
        docs = vector_store.similarity_search(question, k=2)
        context = "\n".join([doc.page_content for doc in docs])
        
        chat_history = memory.load_memory_variables({})["chat_history"]
        if chat_history:
            context += f"\n이전 대화: {chat_history[-2:]}"
        
        answer = get_gemini_response(question, context)
        memory.save_context({"question": question}, {"answer": answer})
        return answer
    
    return chat_with_memory

# Streamlit 앱
def main():
    st.title("PDF 기반 챗봇")
    st.write("PDF 파일을 업로드하여 질문에 답변받아보세요.")

    # PDF 업로드 위젯
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])

    # 세션 상태 초기화
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None

    # 업로드된 파일 처리
    if uploaded_file is not None and st.session_state.chatbot is None:
        with st.spinner("PDF를 로드하고 처리 중..."):
            documents = load_pdf_from_upload(uploaded_file)
            if documents:
                split_docs = split_documents(documents)
                vector_store = create_vector_store(split_docs)
                st.session_state.chatbot = setup_chatbot(vector_store)
                st.success("챗봇 준비 완료!")
            else:
                st.error("PDF 로드에 실패했습니다.")

    # 챗봇이 준비된 경우에만 대화 활성화
    if st.session_state.chatbot:
        # 대화 기록을 세션 상태에 저장
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 이전 대화 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 사용자 입력
        if question := st.chat_input("질문을 입력하세요:"):
            # 사용자 메시지 추가
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            # 챗봇 응답
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    answer = st.session_state.chatbot(question)
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("PDF 파일을 업로드하면 챗봇이 활성화됩니다.")

if __name__ == "__main__":
    main()