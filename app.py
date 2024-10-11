import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from streamlit_chat import message
from dotenv import load_dotenv  # Import load_dotenv
from langchain.schema import Document
import tiktoken  # Import token counting library

# Custom TextLoader to handle different encodings
from langchain_community.document_loaders.text import TextLoader

class CustomTextLoader(TextLoader):
    def load(self):
        docs = []
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
            docs.append(Document(page_content=text, metadata={"source": self.file_path}))
        return docs
    
# Load environment variables from .env file
load_dotenv()

# Access the OPENAI_API_KEY from the environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

# Define a list of medical-related keywords to detect medical queries
medical_keywords = ["medicine", "doctor", "treatment", "symptom", "diagnosis", "medication", "hospital", "disease"]

chain = None

@st.cache_resource
def initialize_index():
    if PERSIST and os.path.exists("persist"):
        st.write("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    else:
        loader = CustomTextLoader("fever.txt")  # Ensure this path is correct
        vectorstore = Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="persist")
        
        # Load documents and add them to the vector store
        documents = loader.load()
        vectorstore.add_documents(documents)  # This will create the collection if it doesn't exist

    return VectorStoreIndexWrapper(vectorstore=vectorstore)

@st.cache_resource
def get_chain(_index):
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=_index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

def token_count(messages):
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    total_tokens = sum(len(tokenizer.encode(msg[0])) + len(tokenizer.encode(msg[1])) for msg in messages)
    return total_tokens

def main():
    global chain
    st.title("Chatbot for Fever")

    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize index and chain
    index = initialize_index()
    chain = get_chain(index)

    # Chat container
    chat_container = st.container()

    # Display chat history
    with chat_container:
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
            # Limit the number of messages sent in the history to the last 5
            max_history_length = 5
            chat_history_to_send = st.session_state.chat_history[-max_history_length:]

            # Check the total token count
            total_tokens = token_count(chat_history_to_send) + len(tiktoken.encoding_for_model("gpt-3.5-turbo").encode(user_input))

            if total_tokens > 16385:
                st.warning("Your conversation is too long. Please shorten it.")
                return

            # Get the result from the chain
            result = chain.invoke({"question": user_input, "chat_history": chat_history_to_send})

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
