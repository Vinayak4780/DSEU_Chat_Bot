# import re
# import os
# import pickle
# import pandas as pd
# import faiss
# import numpy as np

# # ✅ LangChain memory module
# from langchain.memory import ConversationBufferMemory

# # ✅ Use community version for embeddings (LangChain v0.2+)
# from langchain.embeddings import HuggingFaceEmbeddings

# # ✅ Direct use of SentenceTransformer (if you use raw embeddings outside LangChain)
# from sentence_transformers import SentenceTransformer

# # ✅ Updated LlamaCpp import for LangChain v0.2+
# from langchain_community.llms import LlamaCpp
# # ✅ Load Data
# data_path = "agents/ProcessedData/Campuses Data (Responses)(1).xlsx"
# df1 = pd.read_excel(data_path)

# # ✅ Normalize campus names for better search
# df1["Normalized Campus Name"] = df1["Name of the Campus"].str.replace(r"\bDSEU\b", "", regex=True, case=False).str.strip()

# # ✅ Select the problematic column name
# labs_col = "Labs In the Campus(Provide Labs' description with Labs' name including the departments they fall in)"

# # ✅ Initialize Hugging Face Embeddings (Improved Model)
# embedding_model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

# # ✅ Paths for FAISS indexes and pickle storage
# index_paths = {
#     "campus": "faiss_campus.index",
#     "course": "faiss_course.index",
#     "location": "faiss_location.index",
# }
# pickle_path = "campus_data.pkl"

# # ✅ Function to create FAISS index
# def create_faiss_index(vectors):
#     dimension = vectors.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(vectors)
#     return index

# # ✅ Check if FAISS indexes already exist
# if all(os.path.exists(path) for path in index_paths.values()) and os.path.exists(pickle_path):
#     print("🔄 Loading FAISS indexes and data from storage...")
    
#     # Load FAISS indexes
#     faiss_campus = faiss.read_index(index_paths["campus"])
#     faiss_course = faiss.read_index(index_paths["course"])
#     faiss_location = faiss.read_index(index_paths["location"])
    
#     # Load DataFrame with stored vectors
#     with open(pickle_path, "rb") as f:
#         df1 = pickle.load(f)
# else:
#     print("📌 Generating new FAISS indexes and storing data...")

#     # Compute vector embeddings
#     df1["Campus Vector"] = df1["Normalized Campus Name"].apply(lambda x: embedding_model.encode(x))
#     df1["Course Vector"] = df1["Courses Offered by the Campus"].fillna("").apply(lambda x: embedding_model.encode(x))
#     df1["Location Vector"] = df1["Location of the campus"].fillna("").apply(lambda x: embedding_model.encode(x))

#     # Stack vectors into NumPy arrays
#     campus_vectors = np.stack(df1["Campus Vector"].values)
#     course_vectors = np.stack(df1["Course Vector"].values)
#     location_vectors = np.stack(df1["Location Vector"].values)

#     # Create FAISS indexes
#     faiss_campus = create_faiss_index(campus_vectors)
#     faiss_course = create_faiss_index(course_vectors)
#     faiss_location = create_faiss_index(location_vectors)

#     # Save FAISS indexes to files
#     faiss.write_index(faiss_campus, index_paths["campus"])
#     faiss.write_index(faiss_course, index_paths["course"])
#     faiss.write_index(faiss_location, index_paths["location"])

#     # Save DataFrame with vectors to a pickle file
#     with open(pickle_path, "wb") as f:
#         pickle.dump(df1, f)

# # ✅ Function to format output
# def format_output(row):
#     return f"""
#     🎓 **{row.get('Name of the Campus', 'Unknown Campus')}**\n
#     📍 **Campus Name**: {row.get('Name of the Campus', 'N/A')}\n
#     📧 **Email**: {row.get('Email Address', 'N/A')}\n
#     📌 **Location**: {row.get('Location of the campus', 'N/A')}\n
#     📚 **Courses Offered**: {row.get('Courses Offered by the Campus', 'N/A')}\n
#     🏛️ **Labs & Descriptions**: {row.get(labs_col, 'N/A')}\n
#     📸 **Campus Photos**: {row.get('Upload the Photos of the Campus', 'N/A')}\n
#     """

# # ✅ Function to search using FAISS
# def search_campus(query, faiss_index, vector_column, df, top_k=1):
#     query_vector = embedding_model.encode(query).reshape(1, -1)
#     D, I = faiss_index.search(query_vector, k=top_k)  # Retrieve top-k matches
#     results = []

#     for idx in I[0]:
#         if idx == -1:
#             continue
#         row = df.iloc[idx]
#         results.append(format_output(row))

#     return results if results else ["⚠️ No matching results found!"]

# # ✅ Enhanced Function to Handle Multi-Query Search
# def search_university(query):
#     query = query.lower().strip()

#     # 🔍 **Check for course-related search**
#     if "course" in query:
#         match = re.search(r"course (.+)", query)
#         if match:
#             course_query = match.group(1).strip()
#             return search_campus(course_query, faiss_course, "Course Vector", df1)

#     # 🔍 **Check for location-related search**
#     if "location" in query:
#         match = re.search(r"location (.+)", query)
#         if match:
#             location_query = match.group(1).strip()
#             return search_campus(location_query, faiss_location, "Location Vector", df1)
#     if "college" in query:
#         match = re.search(r"college (.+)", query)
#         if match:
#             college_query = match.group(1).strip()
#             response = search_campus(college_query, faiss_campus, df1)
#             memory.chat_memory.add_ai_message(response)
#             return response

#     # 🔍 **Default: Search by Campus Name**
#     return search_campus(query, faiss_campus, "Campus Vector", df1)
# # ✅ Load Free Local LLM (Mistral, Llama-2, GPT4All)
# LLM_PATH = "/home/ubuntu/project_dseu/mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # 🔹 Replace with actual model path

# if not os.path.exists(LLM_PATH):
#     print(f"❌ Error: Model file not found at {LLM_PATH}")
#     exit(1)

# llm = LlamaCpp(model_path=LLM_PATH, n_ctx=4096,temperature=0.1,max_tokens=100,n_gpu_layers = 3 )

# # ✅ Initialize LangChain Memory for Chat History
# memory = ConversationBufferMemory()


# from crewai import Agent, Task, Crew

# CampusSearchAgent = Agent(
#     role="DSEU Admissions Guide",
#     goal="Assist users with all admission-related queries using a conversational interface.",
#     backstory=(
#         "You are a campus expert who knows all DSEU campuses inside out — including locations, courses, facilities, and key events. You have supported admission and outreach teams in resolving complex student queries. If a question isn’t campus-specific, you help coordinate with the faculty, forms, or links agents to ensure users still get the right information."
#     ),
#     verbose=True,
#     allow_delegation=False,
#     llm=llm
# )

#     # ✅ Main Chatbot Loop
# print("🏫 Welcome to the University Campus Chatbot! Type 'exit' to stop.")
# def campus_agent_bot(query):
#     memory.chat_memory.add_user_message(query)
#     responses = search_university(query)

#     if responses:
#         final = "\n\n".join(responses)
#         memory.chat_memory.add_ai_message(final)
#         return f"Thought: I found the campus details.\nFinal Answer: {final}"
#     else:
#         return "Thought: No matching campus found.\nFinal Answer: Please try again."



# from crewai import Agent

# campus_Task  = Task(
#     description="Engage in an interactive conversation about campus and answer queries accurately.",
#     expected_output="A helpful and complete interaction with the student, providing all requested information.",
#     agent=CampusSearchAgent,
#     async_execution=False,
#     tools=[],
#     callback=lambda x: campus_agent_bot(x.input),  # Trigger your CLI chatbot
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
You are an expert AI assistant helping users query a university's campus catalog database using SQL.

Use only the following table and columns:

**Table: courses**
  - Timestamp 
  - Email Address
  - Name of the Campus 
  - Location of the campus 
  - About the Campus(Description of the campus
  - Courses Offered by the Campus 
  - Labs In the Campus(Provide Labs' description with Labs' name including the departments they fall in
  - Upload the Photos of the Campus 
You must only use these columns and assume the table is named `campus-data`.

Question: {question}
SQLQuery:
""")
LLM_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# llm = LlamaCpp(
#     model_path=LLM_PATH,
#     n_ctx=8192,
#     temperature=0.2,
#     max_tokens=300,
#     n_gpu_layers=20
# )
from langchain.chains import RetrievalQA
from shared_llms import shared_llm as llm
# Load FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("data/campus_sql_output", embeddings=embedding_model, allow_dangerous_deserialization=True)

retriever = vectorstore.as_retriever(k=3)

# Load local Mistral model (path to .gguf model)

# Setup RetrievalQA Chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)
def campus_agent_bot(user_input):
    response = qa.run(user_input) # or summarize if needed
    return response
# interactive mode
# while True:
#     query = input("Ask your data: ").lower()
#     if query.lower() in ["exit", "quit"]:
#         break
#     result = qa.run(query)
#     print("→", result)
from crewai import Agent, Task, Crew

CampusSearchAgent = Agent(
    role="DSEU Admissions Guide",
    goal="Assist users with all admission-related queries using a conversational interface.",
    backstory=(
        "You are a campus expert who knows all DSEU campuses inside out — including locations, courses, facilities, and key events. You have supported admission and outreach teams in resolving complex student queries. If a question isn’t campus-specific, you help coordinate with the faculty, forms, or links agents to ensure users still get the right information."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)
campus_Task  = Task(
    description="Engage in an interactive conversation about campus and answer queries accurately.",
    expected_output="A helpful and complete interaction with the student, providing all requested information.",
    agent=CampusSearchAgent,
    async_execution=False,
    tools=[],
    callback=lambda x: campus_agent_bot(x.input),  # Trigger your CLI chatbot
)