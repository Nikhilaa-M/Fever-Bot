import os
import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.schema import Document

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
        max_tokens_limit=4000,  # Add this line
    )

# ... existing code ...

def main():
    global chain
    st.title("Chatbot for Fever")

    index = initialize_index()
    chain = get_chain(index)

    # Initialize an empty list for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Type something...")

    if user_input:
        if user_input.lower() in ['quit', 'exit', 'bye', 'thank you']:
            st.write("Thank you for using this bot. Have a great day!")
        else:
            try:
                # Call the chain with chat history
                result = chain.invoke({"question": user_input, "chat_history": st.session_state.chat_history})

                if result['answer'] and result['answer'].strip():
                    if "fever" in result['answer'].lower():
                        st.write(result['answer'])
                    else:
                        st.write("I am a fever bot. Please ask about fever.")
                else:
                    if any(keyword in user_input.lower() for keyword in medical_keywords):
                        st.write("Sorry, I don't have information about your query. I will let the doctor know.")
                    else:
                        st.write("I am a fever bot. Please ask about fever.")

                # Update the chat history after processing input
                st.session_state.chat_history.append((user_input, result['answer']))

                # Limit the chat history to the last 5 interactions
                st.session_state.chat_history = st.session_state.chat_history[-5:]

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
