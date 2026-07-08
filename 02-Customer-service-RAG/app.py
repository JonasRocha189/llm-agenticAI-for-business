import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="SafeBank Customer Service 🤖", page_icon="🤖")
st.title("SafeBank Customer Service")

id_model = "llama-3.3-70b-versatile"
temperature = 0.7
path = Path(__file__).parent

### Load LLM
def load_llm(id_model, temperature):
  llm = ChatGroq(
    model=id_model,
    temperature=temperature,
    max_tokens=None,
    timeout=None,
    max_retries=2,
  )
  return llm

llm = load_llm(id_model, temperature)

### Show result
def show_res(res):
  from IPython.display import Markdown
  if "</think>" in res:
    res = res.split("</think>")[-1].strip()
  else:
    res = res.strip()  # fallback if there is no tag
  display(Markdown(res))

### Content extraction
def extract_text_pdf(file_path):
  loader = PyMuPDFLoader(file_path)
  doc = loader.load()
  content = "\n".join([page.page_content for page in doc])
  return content

### Indexing and retrieval
def config_retriever(folder_path):
  # Load documents
  docs_path = Path(folder_path)
  pdf_files = [f for f in docs_path.glob("*.pdf")]

  if len(pdf_files) < 1:
    st.error("No PDF file uploaded")
    st.stop()

  loaded_documents = [extract_text_pdf(pdf) for pdf in pdf_files]

  # Splitting into chunks
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=200
  )
  chunks = []
  for doc in loaded_documents:
      chunks.extend(text_splitter.split_text(doc))

  # Embeddings
  embedding_model = "BAAI/bge-m3" #sentence-transformers/all-mpnet-base-v2

  embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

  # Storage
  vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

  vectorstore.save_local('index_faiss')

  # Configuring the text retriever
  retriever = vectorstore.as_retriever(
      search_type='mmr',
      search_kwargs={'k':3, 'fetch_k':4}
  )

  return retriever

### RAG Chain
def config_rag_chain(llm, retriever):

  # Contextualization prompt
  context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."

  context_q_system_prompt = context_q_system_prompt
  context_q_user_prompt = "Question: {input}"
  context_q_prompt = ChatPromptTemplate.from_messages(
      [
          ("system", context_q_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", context_q_user_prompt),
      ]
  )

  # Contextualization Chain
  history_aware_retriever = create_history_aware_retriever(
    llm=llm, retriever=retriever, prompt=context_q_prompt
  )

  # Prompt para perguntas e respostas (Q&A)
  system_prompt = """You are a helpful virtual assistant answering general questions about a company's services.
  Use the following retrieved context snippets to answer the question.
  If you do not know the answer, simply state that you cannot say for sure.
  However, if it is a very common question, you may suggest a possible solution as an alternative.
  Keep the answer concise.
  Respond in English. \n\n"""

  qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "Question: {input}\n\n Context: {context}"),
  ])

  # Configure the LLM and the Chain for Question and Answering (Q&A)

  qa_chain = create_stuff_documents_chain(llm, qa_prompt)

  rag_chain = create_retrieval_chain(
    history_aware_retriever,
    qa_chain,
  )

  return rag_chain

### Chat interaction
def chat_llm(rag_chain, input):

  st.session_state.chat_history.append(HumanMessage(content=input))

  response = rag_chain.invoke({
      "input": input,
      "chat_history": st.session_state.chat_history
  })

  res = response["answer"]
  res = res.split("</think>")[-1].strip() if "</think>" in res else res.strip()

  st.session_state.chat_history.append(AIMessage(content=res))

  return res

input = st.chat_input("Type your message here...")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [
      AIMessage(content = "Hello, I am your virtual assistant! How can I help you?"),
  ]

if "retriever" not in st.session_state:
  st.session_state.retriever = None

for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)

if input is not None:
  with st.chat_message("Human"):
    st.markdown(input)

  with st.chat_message("AI"):
    if st.session_state.retriever is None:
      st.session_state.retriever = config_retriever(path)
    rag_chain = config_rag_chain(llm, st.session_state.retriever)
    res = chat_llm(rag_chain, input)
    st.write(res)