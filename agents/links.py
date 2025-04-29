import difflib
import json
import os

# ✅ Updated imports to follow LangChain v0.2+ standards
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.document import Document


from crewai import Agent, Task, Crew
from langchain_community.llms import LlamaCpp


FAISS_DIR = "faiss_link_index"

def load_links_from_file(file_path):
    """Loads a dictionary of links from a file, supporting JSON, dictionary, and plain text formats."""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        try:
            evaluated = eval(content)
            if isinstance(evaluated, dict):
                return evaluated
        except Exception:
            pass

        name_link_dict = {}
        for line in content.splitlines():
            if ":" in line:
                key, value = map(str.strip, line.split(":", 1))
                name_link_dict[key] = value

        return name_link_dict if name_link_dict else None

def build_or_load_faiss(name_link_dict):
    """Builds or loads a FAISS index using LangChain."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(FAISS_DIR):
        return FAISS.load_local(FAISS_DIR, embeddings=embeddings,allow_dangerous_deserialization=True)

    documents = [
        Document(page_content=key, metadata={"link": value})
        for key, value in name_link_dict.items()
    ]
    db = FAISS.from_documents(documents, embedding=embeddings)
    db.save_local(FAISS_DIR)
    return db

def find_best_match_fuzzy(query, name_link_dict):
    """Fuzzy match fallback if FAISS fails."""
    query = query.lower()
    best_match = difflib.get_close_matches(query, list(name_link_dict.keys()), n=1, cutoff=0.5)
    if best_match:
        return best_match[0], name_link_dict[best_match[0]]
    return None, None
# ✅ Load Free Local LLM (Mistral, Llama-2, GPT4All)
LLM_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # 🔹 Replace with actual model path

if not os.path.exists(LLM_PATH):
    print(f"❌ Error: Model file not found at {LLM_PATH}")
    exit(1)

# llm = LlamaCpp(model_path=LLM_PATH, n_ctx=4096,temperature=0.1,max_tokens=200,n_gpu_layers= 20)
from shared_llms import shared_llm as llm
print("[DEBUG] LLM Type Used in Agent:", type(llm))


# Run the chatbot with the fi
dLinkSearchAgent = Agent(
    role = "Find revelant links on user queries",
    goal = "Retrive and suggest the best matching links",
    backstory="You are a digital navigator for DSEUs official online resources. You manage updated URLs for admission portals, document downloads, program details, events, and notices. You ensure users always get the correct and most recent links for their queries.",
    verbose = True,
    allow_delegation= False,
    llm = llm
)

def Link_predict(user_query, file_path="agents/EDITED_LINKS(1).txt"):
    """Runs the AI Link Chatbot with FAISS + Fuzzy fallback."""
    name_link_dict = load_links_from_file(file_path)

    if not name_link_dict:
        return "Thought: Link data is not available.\nFinal Answer: Please check the file format or try again later."

    db = build_or_load_faiss(name_link_dict)

    docs = db.similarity_search(user_query, k=1)
    if docs:
        doc = docs[0]
        best_name = doc.page_content
        best_link = doc.metadata.get("link", "Link not found")
        return f"Thought: I found a matching link for '{best_name}'.\nFinal Answer: {best_link}"
    else:
        # Fallback to fuzzy matching
        best_name, best_link = find_best_match_fuzzy(user_query, name_link_dict)
        if best_link:
            return f"Thought: I found a fuzzy match for '{best_name}'.\nFinal Answer: {best_link}"
        else:
            return "Thought: I couldn't find any match.\nFinal Answer: Please rephrase your query or try again."



Link_task = Task(
    description="Retrieve relevant DSEU link based on user query.",
    expected_output="A correct and useful link for the student.",
    agent=dLinkSearchAgent,
    tools=[],
    async_execution=False,
    callback=lambda x: Link_predict(x.input)  # ✅ Correct fix
)
