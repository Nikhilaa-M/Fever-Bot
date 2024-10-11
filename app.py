import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.schema import Document
from streamlit_chat import message  # Ensure you have streamlit_chat installed

class CustomTextLoader(TextLoader):
    def load(self):
        docs = []
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            docs.append(Document(page_content=text, metadata={"source": self.file_path}))
        return docs

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

PERSIST = False
medical_keywords = ["medicine", "doctor", "treatment", "symptom", "diagnosis", "medication", "hospital", "disease"]

chain = None

@st.cache_resource
def initialize_index():
    if PERSIST and os.path.exists("persist"):
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    else:
        loader = CustomTextLoader("fever.txt")
        vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="persist")
        documents = loader.load()
        vectorstore.add_documents(documents)

    return VectorStoreIndexWrapper(vectorstore=vectorstore)

@st.cache_resource
def get_chain(_index):
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=_index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        max_tokens_limit=4000,  # Use the specified max tokens limit
    )

def main():
    global chain
    st.title("Chatbot for Fever")

    index = initialize_index()
    chain = get_chain(index)

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Chat container for displaying history
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            message(user_msg, is_user=True, key=f"user_msg_{i}")
            message(bot_msg, key=f"bot_msg_{i}")

    # User input at the bottom
    with st.container():
        st.text_input("Type something...", key="user_input", on_change=process_input)

def process_input():
    global chain
    user_input = st.session_state.user_input
    if user_input:
        if user_input.lower() in ['quit', 'exit', 'bye', 'thank you']:
            bot_response = "Thank you for using this bot. Have a great day!"
        else:
            # Get the result from the chain
            result = chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})

            # Process the result
            if result['answer'] and result['answer'].strip():
                if "fever" in result['answer'].lower():
                    bot_response = result['answer']
                else:
                    bot_response = "I am a fever bot. Please ask about fever."
            else:
                if any(keyword in user_input.lower() for keyword in medical_keywords):
                    bot_response = "Sorry, I don't have information about your query. I will let the doctor know."
                else:
                    bot_response = "I am a fever bot. Please ask about fever."

        st.session_state.chat_history.append((user_input, bot_response))
        st.session_state.user_input = ""  # Clear the input

if __name__ == "__main__":
    main()
