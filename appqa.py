
#%% libs
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Pinecone
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message
import openai
import pinecone
from utils import *

#%% utils
# def get_conversation_string():
#     conversation_string = ""
#     for i in range(len(st.session_state['responses'])-1):        
#         conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
#         conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
#     return conversation_string

# def query_refiner(conversation, query, model_name):
#     response = openai.Completion.create(
#                                         model="text-davinci-003",
#                                         prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
#                                         temperature=0.7,
#                                         max_tokens=256,
#                                         top_p=1,
#                                         frequency_penalty=0,
#                                         presence_penalty=0
#                                         )
#     return response['choices'][0]['text']

# # find chunk matchs 
# def find_match(model, input):
#     input_em = model.encode(input).tolist()
#     result = index.query(input_em, top_k=2, includeMetadata=True)
#     return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']




#%% app layout
with st.sidebar:
    st.title('🤖💬 Q & A App')
    st.subheader("Chat with your own documents and private knowledge base")
    # openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==51):
        st.warning('Please set your OpenAI API key!', icon='⚠️')
    else:
        st.success('Your OpenAI API key is valid.', icon='👍')
    
    # select the model
    selected_model = st.selectbox("**Select a model**", ["gpt-3.5-turbo", "gpt-4"])
    st.write("Selected Model:", selected_model)


if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

llm = ChatOpenAI(model_name=selected_model, openai_api_key="")

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)

system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text below, say 'I don't know'""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)





#%% Pinecone vector DB
embeddings = OpenAIEmbeddings(model_name="ada")
pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENVIRONMENT'))
index_name = 'pdf-qa-demo'
index = Pinecone.from_existing_index(index_name, embeddings)


################################################################################################################
#%% new functions
def get_similiar_docs(vdb_index, query, k=2, score=False):
  if score:
    similar_docs = vdb_index.similarity_search_with_score(query, k=k)
  else:
    similar_docs = vdb_index.similarity_search(query, k=k)
  return similar_docs

model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_name)

chain = load_qa_chain(llm, chain_type="stuff")

def get_answer(query, index):
  similar_docs = get_similiar_docs(index, query)
  answer = chain.run(input_documents=similar_docs, question=query)
  return answer
################################################################################################################


#%% container for chat history
response_container = st.container()

#%% container for text box
textcontainer = st.container()


query = "How can I assist you?"
# answer = get_answer(query, index)

with textcontainer:
    query = st.text_input("Query👇: ")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            # refined_query = query_refiner(conversation_string, query)
            # st.subheader("Refined Query:")
            # st.write(refined_query)
            # context = find_match(input=query)
            context = get_similiar_docs(index, query)
            # print(context)  
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response) 

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i)+ '_user')