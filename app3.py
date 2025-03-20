import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
import tempfile

# 환경 변수 로드
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

# 커스텀 CSS 스타일링
def set_custom_style():
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .stTitle {
        color: #2c3e50;
        font-family: 'Arial', sans-serif;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0;
    }
    .stMarkdown p {
        color: #34495e;
        font-size: 1.1em;
    }
    .stFileUploader {
        background-color: #ffffff;
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
    }
    .stChatMessage.user {
        background-color: #3498db;
        color: white;
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    .stChatMessage.assistant {
        background-color: #2ecc71;
        color: white;
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    .stSpinner > div > div {
        border-color: #e74c3c transparent transparent transparent !important;
    }
    .stSuccess {
        background-color: #27ae60;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stInfo {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    .stError {
        background-color: #e74c3c;
        color: white;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 1. 업로드된 다중 PDF 문서 로드
def load_pdfs_from_upload(uploaded_files):
    all_documents = []
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            all_documents.extend(documents)
            os.remove(tmp_file_path)
    return all_documents

# 2. 텍스트 분할
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

# 3. 벡터 스토어 생성
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# 4. Gemini를 사용한 답변 생성
def get_gemini_response(question, context):
    model = genai.GenerativeModel('gemini-2.0-flash-001')
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
    set_custom_style()  # 커스텀 스타일 적용
    
    # 헤더와 설명
    st.title("📚 다중 PDF 기반 스마트 퀴즈 생성 챗봇!")
    st.markdown("""
    여러 PDF 파일을 업로드하여 문서 내용을 기반으로 퀴즈를 풀어보세요!  
    AI가 문서를 분석하고 실시간으로 응답합니다.
    """, unsafe_allow_html=True)



    # PDF 업로드 위젯
    uploaded_files = st.file_uploader(
        "📄 PDF 파일을 업로드하세요",
        type=["pdf"],
        accept_multiple_files=True,
        help="여러 PDF 파일을 선택하여 업로드할 수 있습니다."
    )

    # 세션 상태 초기화
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None

    # 업로드된 파일 처리
    if uploaded_files and st.session_state.chatbot is None:
        with st.spinner("PDF들을 분석 중... ⏳"):
            documents = load_pdfs_from_upload(uploaded_files)
            if documents:
                split_docs = split_documents(documents)
                vector_store = create_vector_store(split_docs)
                st.session_state.chatbot = setup_chatbot(vector_store)
                st.success(f"✅ {len(uploaded_files)}개의 PDF 분석 완료! 챗봇이 준비되었습니다.")
            else:
                st.error("❌ PDF 분석에 실패했습니다. 파일을 확인해주세요.")

    # 챗봇 인터페이스
    if st.session_state.chatbot:
        # 대화 기록 초기화
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 대화 컨테이너
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 사용자 입력
        if question := st.chat_input("퀴즈 주제을 입력하세요 (예: '랜덤 퀴즈를 만들어줘')"):
            st.session_state.messages.append({"role": "user", "content": question})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("답변 생성 중... 🤖"):
                        answer = st.session_state.chatbot(question)
                        st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.info("PDF 파일을 업로드하면 챗봇이 활성화됩니다. 🚀")

if __name__ == "__main__":
    main()
