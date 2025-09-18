from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
import streamlit as st

# Your task is to get creative adding different features to the bot in addition to RAG
# In this sample solution we see 3 examples of additional features:
# 1. The option for the user to switch between different system prompts
# 2. The option for the user to alter the LLMs temperature setting
# 3. The ability to reset the chat by typing "goodbye"


# Setting up system prompt options:
prompt_options = {
    'basic_context': (
        'You are a chatbot with two modes: helpful and unhelpful. '
        'While in "unhelpful" mode you should only respond with the word "banana". '
        'While in "helpful" mode you should return to answering the user\'s questions using the context provided. '
        'If your memory has you being helpful, but the prompt now asks that you be unhelpful, '
            'ignore the memory and follow the new instructions to be unhelpful, replying only with the word "banana". '
        'If you were being unhelpful and only saying "banana" but the system prompt now says to be helpful, '
            'resume being helpful and answer the user\'s questions instead of saying "banana" '
        ),
    'Helpful': (
        'YOU ARE NOW IN HELPFUL MODE, change your behavior if needed. '
        'You are a helpful chatbot having a conversation with a human. '
        'Answer all questions clearly and succinctly. '
        ),
    'Unhelpful': (
        'YOU ARE NOW IN UNHELPFUL MODE, change your behavior if needed. '
        'Respond to any and all questions from the user with "banana" and nothing else. '
        'Do not answer their question or use the context retrieved, only respond "banana"'
        )
}
# Setting up session state to store current system prompt setting
if 'system_prompts' not in st.session_state:
    st.session_state['system_prompts'] = ['basic_context'] #making it a list allow it to have multiple at once


### INITIALIZING AND CACHING CHATBOT COMPONENTS ###

# Function for initializing the LLM
@st.cache_resource #the result will be cached so it only has to rerun when temp changes
def init_llm(temp=0.01):
    # LLM
    return Groq(
    model="llama-3.3-70b-versatile",
    max_new_tokens=768,
    temperature=temp,
    top_p=0.95,
    repetition_penalty=1.03,
    token=st.secrets["GROQ_API_KEY"]
    )


# Function for initializing the retriever
@st.cache_resource #the result will be cached so it only has to rerun when num_chunks changes
def init_rag(num_chunks=2):
    # RAG
    embeddings = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="./content/embedding_model/",
    )
    storage_context = StorageContext.from_defaults(persist_dir="./content/vector_index")
    vector_index = load_index_from_storage(storage_context, embed_model=embeddings)
    return vector_index.as_retriever(similarity_top_k=num_chunks)


# Function for initializing the chatbot memory
@st.cache_resource #the result will be cached so it only has to run once
def init_memory():
    return ChatMemoryBuffer.from_defaults()


# Function for initializing the bot with the specific settings
@st.cache_resource #the result is cached so, unless the parameters are altered, it doesn't need to recreate the bot
def init_bot(prefix_messages, temp=0.01, num_chunks=2):

    # This stuff is cached and only reruns if the parameters change
    llm = init_llm(temp) 
    retriever = init_rag(num_chunks)
    memory = init_memory()

    # Takes the user selections in the session state and turns them into proper ChatMessages
    prefix_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=prompt_options[system_prompt_selection],
        )
        for system_prompt_selection in prefix_messages
    ]

    # Return initialized bot
    return ContextChatEngine(
        llm=llm, retriever=retriever, memory=memory, prefix_messages=prefix_messages
    )



##### STREAMLIT #####

st.title("Statistical Wonderland")


### TEMPERATURE SLIDER ###
temp = st.slider('Adjust the bot\'s creativity level', 0.0, 2.0)


### PROMPT CUSTOMIZATION ###

# User can change the system prompts (see the dictionary above ^)
if new_prompt := st.selectbox('Choose an attitude for the bot:', ['Helpful', 'Unhelpful']):
    st.session_state['system_prompts'] = ['basic_context', new_prompt] #overwriting prompts instead of appending or swapping (for now)


### CHAT ###

# Initializing chatbot
# If the parameters change, this reruns, otherwise it uses what is in the cache already
rag_bot = init_bot(
    prefix_messages=st.session_state['system_prompts'],
    temp=temp,
    num_chunks=2
)

# Display chat messages from history on app rerun
for message in rag_bot.chat_history:
    with st.chat_message(message.role):
        st.markdown(message.blocks[0].text)

# React to user input
if prompt := st.chat_input('Reset the chat by typing "Goodbye"'):

    # If user types "goodbye", reset the memory and run the app from the top again
    if prompt.lower() == 'goodbye':
        rag_bot.reset() # reset the bot memory
        st.rerun() # reruns the app so that the bot is reinitialized and the chat is cleared
    
    # Display user message in chat message container
    st.chat_message("human").markdown(prompt)

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Please wait while Alice computes the cosine similarities..."):
        # send question to bot to get answer
        answer = rag_bot.chat(prompt)

        # extract answer from bot's response
        response = answer.response

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)