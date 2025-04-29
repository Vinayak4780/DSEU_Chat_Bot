from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from langchain.sql_database import SQLDatabase
from langchain.prompts import PromptTemplate
import atexit

from crewai import Agent, Task
sql_prompt = PromptTemplate.from_template("""
You are an expert AI assistant helping users query a university's course catalog database using SQL.

Use only the following table and columns:

**Table: courses**
- course_type
- course_code
- course_name
- L (lecture credits)
- T (tutorial credits)
- P (practical credits )
- credits
- program
- academic_year
- semester
- syllabus (a link)
- category

You must only use these columns and assume the table is named `courses`.

Question: {question}
SQLQuery:
""")
LLM_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# llm = LlamaCpp(
#     model_path=LLM_PATH,
#     n_ctx=4096,
#     temperature=0.2,
#     max_tokens=300,
#     n_gpu_layers= 20
# )
from langchain.chains import RetrievalQA
from shared_llms import shared_llm as llm
# Load FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("data/syllabus_sql_output", embeddings=embedding_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(k=3)

# Load local Mistral model (path to .gguf model)

# Setup RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)
def syllabus_bot(user_input):
    return qa.run(user_input)
# # interactive mode
# while True:
#     query = input("Ask your data: ").lower()
#     if query.lower() in ["exit", "quit"]:
#         break
#     result = qa.run(query)
    print("→", result)
syllabus_agent = Agent(
    role="DSEU Syllabus Bot",
    goal="Answer student queries related to courses and syllabus accurately using retrieval-based memory.",
    backstory="You are the expert on DSEU syllabus and course structures. Use your knowledge base to help students navigate the academic structure and curriculum of the university.",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

# ✅ Define CrewAI Task
syllabus_task = Task(
    description="Answer questions about course syllabus and academic structure at DSEU.",
    expected_output="A helpful and fact-based answer referencing the syllabus.",
    agent=syllabus_agent,
    async_execution=False,
    callback=lambda x: syllabus_bot(x.input)
)