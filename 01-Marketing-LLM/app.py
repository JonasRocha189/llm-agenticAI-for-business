import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

## Connection to the LLM
id_model = "llama-3.3-70b-versatile"
llm = ChatGroq(
    model=id_model,
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

## Generating function
def llm_generate(llm, prompt):
  template = ChatPromptTemplate.from_messages([
      ("system", "You are a digital marketing specialist focused on SEO and persuasive copywriting."),
      ("human", "{prompt}"),
  ])

  chain = template | llm | StrOutputParser()

  res = chain.invoke({"prompt": prompt})
  return res

st.set_page_config(page_title = "Content Generator 🤖", page_icon="🤖")
st.title("Gerador de conteúdo")

# Form inputs
topic = st.text_input("Topic:", placeholder="e.g., mental health, healthy eating, prevention, etc.")
platform = st.selectbox("Platform:", ['Instagram', 'Facebook', 'LinkedIn', 'Blog', 'Email'])
tone = st.selectbox("Tone:", ['Standard', 'Informative', 'Inspirational', 'Urgent', 'Informal'])
length = st.selectbox("Length:", ['Short', 'Medium', 'Long'])
audience = st.selectbox("Target Audience:", ['General', 'Young adults', 'Families', 'Seniors', 'Teenagers'])
cta = st.checkbox("Include CTA")
hashtags = st.checkbox("Include Hashtags")
keywords = st.text_area("Keywords (SEO):", placeholder="e.g., well-being, preventive medicine...")

if st.button("Generate content"):
  prompt = f"""
  Write an SEO-optimized text about the topic '{topic}'.
  Return only the final text in your response, without enclosing it in quotation marks.
  - Where it will be published: {platform}.
  - Tone: {tone}.
  - Target audience: {audience}.
  - Length: {length}.
  - {"Include a clear call to action." if cta else "Do not include a call to action."}
  - {"Include relevant hashtags at the end of the text." if hashtags else "Do not include hashtags."}
  {"- Keywords that must be included in the text (for SEO): " + keywords if keywords else ""}
  """
  try:
      res = llm_generate(llm, prompt)
      st.markdown(res)
  except Exception as e:
      st.error(f"Erro: {e}")