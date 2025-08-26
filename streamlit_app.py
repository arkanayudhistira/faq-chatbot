import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()

# Import helper functions
from rag.preprocessing import load_faq, insert_to_vectorstore
from rag.agent import create_faq_agent


# Set streamlit app name
st.set_page_config(page_title='Nawatech Chatbot')                   

# Load and preprocess FAQ dataset
df = load_faq('data/FAQ_Nawa.xlsx')

# Insert FAQ data to Pinecone vectorstore
index_name = 'nawa-faq'
vectorstore = insert_to_vectorstore(df, index_name=index_name)

# Create chatbot agent
faq_agent = create_faq_agent(vectorstore)

# Initialize chat history in streamlit session
chat_history = StreamlitChatMessageHistory(key='session')
        
# Display past chat messages
for message in chat_history.messages:
    with st.chat_message(message.type):
        st.markdown(message.content)

# Chat box for user input
prompt = st.chat_input("Ask your question here!")

if prompt:
    # Display user message
    with st.chat_message("human"):
        st.markdown(prompt)

    # Display AI response
    with st.chat_message("ai"):
        response = faq_agent.invoke({"input": prompt})
        st.markdown(response['output'])

# Sidebar UI for instructions
with st.sidebar:
    st.title("Nawatech Chatbot")
    with st.expander("How to Use"):
        st.markdown(
            """ 
            1. Ketik pertanyaan Anda di kolom chat.  
            2. Klik tombol send atau tekan Enter untuk mengirim chat.
            3. Kami akan menjawab pertanyaan Anda.
            """
        )

    with st.expander("Example Questions"):
        st.markdown(
            """ 
            - Pada tahun berapa Nawatech didirikan?  
            - Layanan apa saja yang ditawarkan oleh nawatech?
            - Bagaimana cara menghubungi Nawatech?
            """
        )
