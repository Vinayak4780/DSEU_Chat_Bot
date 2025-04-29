import os
import pytesseract
import matplotlib.pyplot as plt
from PIL import Image
from fuzzywuzzy import fuzz

# ✅ Use community version to avoid deprecation warnings
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp

from langchain.memory import ConversationBufferMemory
from crewai import Agent, Task, Crew

INPUT_FOLDER = "data/forms"
VECTORSTORE_PATH = "data/embeddings_db/vectorstore.index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Memory (optional for future dialogue)
memory = ConversationBufferMemory()

def extract_text_from_top_quarter(image_path):
    try:
        image = Image.open(image_path)
        width, height = image.size
        cropped = image.crop((0, 0, width, height // 4))

        # OCR
        text = pytesseract.image_to_string(cropped, lang="eng")
        lines = [line.strip() for line in text.split("\n") if line.strip()]

        # Debug log
        print(f"🔍 Lines from {image_path}: {lines}")

        heading = "No Heading Found"
        subheading = "No Subheading Found"

        # Step 1: Find the line that contains 'ANNEXURE'
        for i, line in enumerate(lines):
            if "ANNEXURE" in line.upper():
                # Step 2: The next line is heading
                if i + 1 < len(lines):
                    heading = lines[i + 1]
                # Step 3: The line after that is subheading
                if i + 2 < len(lines):
                    subheading = lines[i + 2]
                break

        return heading, subheading

    except Exception as e:
        return "Error", str(e)

def build_or_load_faiss_index():
    if os.path.exists(VECTORSTORE_PATH):
        print("📦 FAISS index loaded from disk.")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embedding_model, allow_dangerous_deserialization=True)
        return vectorstore
        
    
    texts, metadatas = [], []

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(INPUT_FOLDER, filename)
            heading, subheading = extract_text_from_top_quarter(path)
            texts.append(f"{heading}. {subheading}")
            metadatas.append({"image_path": path})

    if texts:
        vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metadatas)
        vectorstore.save_local(VECTORSTORE_PATH)
        print("✅ FAISS index built and saved.")
        return vectorstore
    else:
        print("⚠️ No valid texts found.")
        return None

def fuzzy_match(user_input):
    best_match = None
    best_score = 0

    for filename in os.listdir(INPUT_FOLDER):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(INPUT_FOLDER, filename)
            heading, subheading = extract_text_from_top_quarter(path)
            score = max(
                fuzz.ratio(user_input.lower(), heading.lower()),
                fuzz.ratio(user_input.lower(), subheading.lower())
            )
            if score > best_score:
                best_score = score
                best_match = path

    return best_match

def faiss_search(user_input, vectorstore):
    try:
        results = vectorstore.similarity_search(user_input, k=1)
        if results:
            return results[0].metadata["image_path"]
    except Exception as e:
        print(f"⚠️ FAISS search error: {e}")
    return None

def display_image(image_path):
    try:
        image = Image.open(image_path)
        plt.close("all")
        fig, ax = plt.subplots(figsize=(10, 12))
        ax.imshow(image)
        ax.axis("off")
        plt.show()
    except Exception as e:
        print(f"⚠️ Error displaying image: {e}")
# ✅ Load Free Local LLM (Mistral, Llama-2, GPT4All)
LLM_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"  # 🔹 Replace with actual model path

if not os.path.exists(LLM_PATH):
    print(f"❌ Error: Model file not found at {LLM_PATH}")
    exit(1)

# llm = LlamaCpp(model_path=LLM_PATH, n_ctx=4096,temperature=0.1,max_tokens=100,n_gpu_layers= 20)
from shared_llms import shared_llm as llm
# === MAIN ===
vectorstore = build_or_load_faiss_index()
Form_Agent = Agent(
    role = "Find and retrive scanned forms based on queries.",
    goal = "Help users find relevant forms by anylyzing headings",
    backstory="ou are a resource librarian and administrator at DSEU who specializes in managing all types of student and admission-related forms — from application to grievance and fee submission. You maintain the latest versions and are capable of guiding users to the right form based on their needs.",
    verbose = True,
    allow_delegation = False,
    llm= llm
)
print("🤖 Image Chatbot Ready. Type 'exit' to quit.\n")
def forms_predict(query):
    image_path = faiss_search(query, vectorstore)

    if not image_path:
        image_path = fuzzy_match(query)

    if image_path:
        return f"Thought: I found a matching form for the query.\nFinal Answer: Here is the form: {image_path}"
    else:
        return "Thought: I couldn't find a relevant form.\nFinal Answer: Please try a different query."


FormsTask= Task(
    description="Engage in an interactive conversation about DSEU admissions and answer queries accurately.",
    expected_output="A helpful and complete interaction with the student, providing all requested information.",
    agent=Form_Agent,
    async_execution=False,
    tools=[],
    callback=lambda x: forms_predict(x.input),  # Trigger your CLI chatbot
)





























# import os

# # LangChain imports
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.chains import RetrievalQA
# from langchain.llms import LlamaCpp

# # CrewAI imports
# from crewai import Agent, Task

# # Fuzzy matching
# from rapidfuzz import process

# # ── CONFIG ───────────────────────────────────────────────────────────────────────
# LLM_PATH            = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
# EMBEDDING_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"
# FAISS_INDEX_PATH    = "data/embedding_db"
# # ────────────────────────────────────────────────────────────────────────────────

# def build_or_load_faiss_index():
#     """Load an existing FAISS index or build & save one from scratch."""
#     embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     if os.path.isdir(FAISS_INDEX_PATH):
#         return FAISS.load_local(FAISS_INDEX_PATH, embeddings=embedder)
#     # — If you have documents to index, replace this stub:
#     # docs = []  # ← load your documents here, each with `.page_content` and `.metadata["path"]`
#     # vs = FAISS.from_documents(docs, embedder)
#     # vs.save_local(FAISS_INDEX_PATH)
#     # return vs

# def faiss_search(query: str, vs: FAISS, top_k: int = 1) -> str | None:
#     """Return metadata['path'] of the top-k most similar doc, if any."""
#     results = vs.similarity_search(query, k=top_k)
#     if results:
#         return results[0].metadata.get("path")
#     return None

# def fuzzy_match(query: str, choices: list[str]) -> str | None:
#     """Fallback: return the best match from a list of file-paths."""
#     match = process.extractOne(query, choices)
#     return match[0] if match and match[1] > 60 else None

# # ── VERIFY MODEL ────────────────────────────────────────────────────────────────
# if not os.path.isfile(LLM_PATH):
#     raise FileNotFoundError(f"Model file not found at {LLM_PATH}")

# # ── INSTANTIATE LLM ─────────────────────────────────────────────────────────────
# llm = LlamaCpp(
#     model_path=LLM_PATH,
#     n_ctx=4096,
#     temperature=0.1,
#     max_tokens=100,
#     n_gpu_layers=20
# )

# # ── BUILD/LOAD VECTORSTORE ───────────────────────────────────────────────────────
# vectorstore = build_or_load_faiss_index()

# # ── OPTIONAL: A QA CHAIN FOR TEXT QA ─────────────────────────────────────────────
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     retriever=vectorstore.as_retriever(k=3),
#     return_source_documents=False
# )

# # ── DEFINE YOUR FORM-FINDING AGENT ──────────────────────────────────────────────
# Form_Agent = Agent(
#     role="Find and retrieve scanned forms based on queries.",
#     goal="Help users find relevant forms by analyzing headings.",
#     backstory=(
#         "You are a resource librarian and administrator at DSEU "
#         "who specializes in managing all types of student and admission-related forms."
#     ),
#     llm=llm,
#     verbose=True,
#     allow_delegation=False,
# )

# def forms_predict(query: str) -> str:
#     # 1) Try FAISS
#     path = faiss_search(query, vectorstore)
#     # 2) Fallback to fuzzy matching filenames
#     if not path:
#         choices = [doc.metadata.get("path", "") for doc in vectorstore.docs]
#         path = fuzzy_match(query, choices)
#     if path:
#         return (
#             "Thought: I found a matching form for the query.\n"
#             f"Final Answer: Here is the form: {path}"
#         )
#     return (
#         "Thought: I couldn't find a relevant form.\n"
#         "Final Answer: Please try a different query."
#     )

# FormsTask = Task(
#     description="Engage in an interactive conversation about DSEU admissions and answer queries accurately.",
#     expected_output="A helpful and complete interaction with the student, providing all requested information.",
#     agent=Form_Agent,
#     async_execution=False,
#     tools=[],
#     callback=lambda inp: forms_predict(inp),
# )

# # ── SIMPLE CLI LOOP ──────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     print("🤖 Image Chatbot Ready. Type 'exit' to quit.\n")
#     while True:
#         user_input = input("You: ").strip()
#         if user_input.lower() in {"exit", "quit"}:
#             break
#         print("Bot:", forms_predict(user_input), "\n")
