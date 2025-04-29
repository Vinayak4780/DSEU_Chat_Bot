# import os
# import re
# import pandas as pd
# from thefuzz import process  # Fuzzy string matching

# # ✅ Updated imports for LangChain v0.2+
# from langchain_community.vectorstores import FAISS
# from langchain.chains import ConversationalRetrievalChain
# from langchain.memory import ConversationBufferMemory
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.llms import LlamaCpp


# # ✅ Create and Store FAISS Index
# FAISS_INDEX_PATH = "faiss_index_faculty_data"
# memory = ConversationBufferMemory()
# if not os.path.exists(FAISS_INDEX_PATH):
#     print(f"⚠️ FAISS index not found at {FAISS_INDEX_PATH}, creating a new one...")
#     df = pd.read_excel("agents/ProcessedData/final_combined_name_user_data.xlsx")
#     df["Normalized Name"] = df["firstName"].str.lower().str.strip()
#     df["Normalized Email"] = df["email"].str.lower().str.strip()
#     df["Normalized Department"] = df["organizationUnit"].str.lower().str.strip()
#     df["Normalized Designation"] = df["designation"].str.lower().str.strip()
#     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
#     vectorstore = FAISS.from_texts(df["Normalized Name"].tolist() + df["Normalized Email"].tolist() + df["Normalized Department"].tolist() + df["Normalized Designation"].tolist(), embeddings)
#     vectorstore.save_local(FAISS_INDEX_PATH)
#     print("✅ FAISS index created and saved successfully.")
# else:
#     embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")
#     vectorstore = FAISS.load_local(
#         FAISS_INDEX_PATH,
#         embeddings,
#         allow_dangerous_deserialization=True
#     )

# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})  # 🔹 FAISS optimized retrieval

# # ✅ Load Faculty Data
# df = pd.read_excel("agents/ProcessedData/final_combined_name_user_data.xlsx")
# df["Normalized Name"] = df["firstName"].str.lower().str.strip()
# df["Normalized Email"] = df["email"].str.lower().str.strip()
# df["Normalized Department"] = df["organizationUnit"].str.lower().str.strip()
# df["Normalized Designation"] = df["designation"].str.lower().str.strip()

# # ✅ Function to Format Output
# def format_output(row):
#     return f"""
#     👤 **Name**: {row.get('firstName','Unknown')}
#     📧 **Email**: {row.get('email', 'N/A')}
#     🏢 **Department**: {row.get('organizationUnit', 'N/A') if pd.notna(row.get('organizationUnit')) else 'N/A'}
#     🏷️ **Designation**: {row.get('designation', 'N/A') if pd.notna(row.get('designation')) else 'N/A'}
#     """

# # ✅ Function to Search Faculty Data
# def search_faculty(query, df):
#     query = query.lower().strip()

#     # ✅ Step 1: Exact Match - Search by Name, Email, Department, or Designation
#     exact_match_df = df[(df["Normalized Name"] == query) | (df["Normalized Email"] == query) | (df["Normalized Department"] == query) | (df["Normalized Designation"] == query)]
#     if not exact_match_df.empty:
#         row = exact_match_df.iloc[0]
#         return format_output(row)

#     # ✅ Step 2: If No Exact Match, Use FAISS for Retrieval
#     faiss_results = retriever.invoke(query)
#     if faiss_results:
#         for doc in faiss_results:
#             faculty_name = doc.page_content.strip()
#             row = df[(df['Normalized Name'].str.contains(faculty_name, case=False, na=False)) |
#                      (df['Normalized Email'].str.contains(faculty_name, case=False, na=False)) |
#                      (df['Normalized Department'].str.contains(faculty_name, case=False, na=False)) |
#                      (df['Normalized Designation'].str.contains(faculty_name, case=False, na=False))]
#             if not row.empty:
#                 return format_output(row.iloc[0])

#     # ✅ Step 3: If No FAISS Results, Use Fuzzy Matching
#     for column in ["Normalized Name", "Normalized Email", "Normalized Department", "Normalized Designation"]:
#         best_match, score = process.extractOne(query, df[column].unique())
#         if best_match and score > 75:
#             fuzzy_match_df = df[df[column].str.contains(best_match, na=False)]
#             row = fuzzy_match_df.iloc[0]
#             return format_output(row)
#         else:
#             response = "⚠️ No faculty found matching your query!"

#     memory.chat_memory.add_user_message(query)
#     memory.chat_memory.add_user_message(response)
# # ✅ Load Free Local LLM (Mistral, Llama-2, GPT4All)
# LLM_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # 🔹 Replace with actual model path

# if not os.path.exists(LLM_PATH):
#     print(f"❌ Error: Model file not found at {LLM_PATH}")
#     exit(1)

# llm = LlamaCpp(model_path=LLM_PATH, n_ctx=4096,temperature=0.1,max_tokens=100,n_gpu_layers = 16)

# from crewai import Agent, Task, Crew
# FacultySearchAgent = Agent(
#     role = "Find and retrieve details based on userr queries",
#     goal = "Help users find faculty contact and department dedatils efficenty.",
#     backstory = "You are a knowledgeable assistant at DSEU who has worked closely with department heads and faculty members across all campuses. You have detailed information about faculty expertise, designations, departments, and contact details. You are the go-to person when someone needs faculty-related information for any course or college.",
#     verbose = True,
#     allow_delegation = False,
#     llm=llm
# )
# def campus_bot(user_query):
#     response = search_faculty(user_query, df)

#     if response:
#         return f"Thought: I found a matching faculty.\nFinal Answer: {response}"
#     else:
#         return "Thought: No faculty matched the query.\nFinal Answer: Please refine your search."

# faculty_task  = Task(
#     description="Engage in an interactive conversation about DSEU admissions and answer queries accurately.",
#     expected_output="A helpful and complete interaction with the student, providing all requested information.",
#     agent=FacultySearchAgent ,
#     async_execution=False,
#     tools=[],
#     callback=lambda x: campus_bot(x.input),  # Trigger your CLI chatbot
# )











































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
You are an expert AI assistant helping users query a university's faculty catalog database using SQL.

Use only the following table and columns:

**Table: courses**
- name
- email
- organization
- designation
You must only use these columns and assume the table is named `faculty-data`.

Question: {question}
SQLQuery:
""")
LLM_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# llm = LlamaCpp(
#     model_path=LLM_PATH,
#     n_ctx=4096,
#     temperature=0.2,
#     max_tokens=300,
#     n_gpu_layers=20
# )
from shared_llms import shared_llm as llm
from langchain.chains import RetrievalQA

# Load FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("data/faculty_sql_output", embeddings=embedding_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(k=3)

# Load local Mistral model (path to .gguf model)

# Setup RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)
def search_faculty(user_input):
    return qa.run(user_input)
# interactive mode
# while True:
#     query = input("Ask your data: ").lower()
#     if query.lower() in ["exit", "quit"]:
#         break
#     result = qa.run(query)
#     print("→", result)
FacultySearchAgent = Agent(
    role = "Find and retrieve details based on userr queries",
    goal = "Help users find faculty contact and department dedatils efficenty.",
    backstory = "You are a knowledgeable assistant at DSEU who has worked closely with department heads and faculty members across all campuses. You have detailed information about faculty expertise, designations, departments, and contact details. You are the go-to person when someone needs faculty-related information for any course or college.",
    verbose = True,
    allow_delegation = False,
    llm=llm
)
faculty_task  = Task(
    description="Engage in an interactive conversation about DSEU admissions and answer queries accurately.",
    expected_output="A helpful and complete interaction with the student, providing all requested information.",
    agent=FacultySearchAgent ,
    async_execution=False,
    tools=[],
    callback=lambda x: search_faculty(x.input),  # Trigger your CLI chatbot
)
