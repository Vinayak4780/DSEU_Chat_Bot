import os
import re
import string
import logging
import nltk
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from crewai import Agent, Task
from fuzzywuzzy import fuzz

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)
def fuzzy_fallback(query: str, documents: list, top_k: int = 1) -> str:
    best_match = ""
    best_score = 0

    for doc in documents:
        content = doc.page_content
        score = fuzz.partial_ratio(query.lower(), content.lower())
        if score > best_score:
            best_score = score
            best_match = content

    logging.info(f"🔁 Fuzzy fallback score: {best_score}")
    return best_match.strip()[:500] if best_score > 60 else "No close fuzzy match found."


stop_words = set(stopwords.words('english'))

# ✅ Paths
TEXT_FILE_PATH = "/home/ubuntu/ChatBot/project_dseu/agents/merged_text.txt"
FAISS_INDEX_DIR = "data/faiss_index_dseu_admissions"
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")
LLM_PATH = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

# ✅ Ensure agents directory exists
os.makedirs("agents", exist_ok=True)

# ✅ Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    return " ".join(tokens)

# ✅ Build or Load FAISS Index
if os.path.exists(FAISS_INDEX_FILE):
    logging.info(f"📦 FAISS index already exists at '{FAISS_INDEX_FILE}'. Loading...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
    vectorstore = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True
    )
else:
    if not os.path.exists(TEXT_FILE_PATH):
        raise FileNotFoundError(f"Text file not found: {TEXT_FILE_PATH}")

    loader = TextLoader(TEXT_FILE_PATH, encoding="utf-8")
    documents = loader.load()
    for doc in documents:
        doc.page_content = preprocess_text(doc.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")
    logging.info("🔧 Building FAISS index...")
    vectorstore = FAISS.from_documents(texts, embedding_model)
    vectorstore.save_local(FAISS_INDEX_DIR)
    logging.info(f"✅ FAISS index saved to '{FAISS_INDEX_DIR}'")

# ✅ Load LLM
# llm = LlamaCpp(
#     model_path=LLM_PATH,
#     n_ctx=4096,
#     temperature=0.2,
#     max_tokens=300,
#     n_gpu_layers=30
# )
from shared_llms import shared_llm as llm
# ✅ LangChain memory
memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

# ✅ Prompt template
custom_prompt = PromptTemplate.from_template("""
You are an expert assistant for DSEU Admissions.

Answer the following question as accurately as possible using only the provided context.
"

--------------------
Context:
{context}
--------------------
Question: {question}
Answer:
""")

# ✅ RAG chain using FAISS + LLM
retriever = vectorstore.as_retriever(
    search_type="mmr",  # instead of similarity
    search_kwargs={"k": 7, "fetch_k": 20}
)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": custom_prompt}
)

# ✅ FAISS-only chatbot handler
def summrization_bot(query: str) -> str:
    try:
        logging.info(f"📩 Query received: {query}")

        # ✅ Clean query (lightly, to retain structure)
        cleaned_query = re.sub(rf"[{re.escape(string.punctuation)}]", " ", query.lower())
        cleaned_query = " ".join([word for word in word_tokenize(cleaned_query) if word not in stop_words and word.isalnum()])

        logging.debug(f"🔍 Cleaned Query: {cleaned_query}")

        # ✅ Test retrieval
        docs = retriever.get_relevant_documents(cleaned_query)
        logging.info(f"🔍 Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs[:3]):
            logging.info(f"Doc {i+1}: {doc.page_content[:200]}...")

        # ✅ Step 1: Try answering using FAISS RAG
        response = qa.invoke({"question": cleaned_query})
        answer = response.get("answer", "").strip()

        if not answer or answer.lower() in ["i do not know.", "unknown", ""]:
            logging.warning("🤷 No answer from context. Trying fuzzy fallback...")

            # 🧠 Try fuzzy match
            fuzzy_match = fuzzy_fallback(query, docs)

            if fuzzy_match != "No close fuzzy match found.":
                return (
                        f"Thought: Couldn't find a strong match using FAISS RAG, but found a fuzzy match.\n"
                        f"Fuzzy Match Extract: {fuzzy_match}"
                    ) 

            # 📉 If fuzzy also fails, use raw LLM fallback
            logging.info("💬 Switching to LLM fallback after fuzzy failed.")
            fallback_prompt = f"You are a helpful assistant for Delhi Skill and Entrepreneurship University. Try your best to answer: {query}"
            fallback_answer = llm.invoke(fallback_prompt).strip()

            return (
                 f"Thought: I couldn't find an exact match in the official documents.\n"
                 f"Final Answer (LLM Guess): {fallback_answer}"
                )


        return f"Thought: I found the answer from documents.\nFinal Answer: {answer}"

    except Exception as e:
        logging.error(f"❌ Summarization Error: {str(e)}")
        return f"Thought: Something went wrong.\nFinal Answer: Error: {str(e)}"



# ✅ CrewAI Agent
summary_agent = Agent(
    role="DSEU Admissions Guide",
    goal="Assist users with all admission-related queries using a conversational interface.",
    backstory=(
        "You are the master resolver and knowledge integrator at DSEU. You have a broad understanding of all systems — faculty, campuses, forms, and digital resources — and are trained to handle complex or ambiguous queries that may fall outside specialized agents’ scopes."
    ),
    verbose=True,
    allow_delegation=False,
    llm=llm
)

summary_task = Task(
    description="Engage in an interactive conversation about DSEU admissions and answer queries accurately.",
    expected_output="A helpful and complete interaction with the student, providing all requested information.",
    agent=summary_agent,
    async_execution=False,
    callback=lambda x: summrization_bot(x.input)
)

# ✅ Terminal Chatbot UI
if __name__ == "__main__":
    print("\n🤖 DSEU Admissions Expert Chatbot (Terminal Mode)")
    print("Type your admission query below (or type 'exit' to quit):\n")

    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("Chatbot: Goodbye! 👋")
                logging.info("💬 Session ended by user.")
                break

            response = summrization_bot(user_input)
            print(f"Chatbot: {response}\n")

    except KeyboardInterrupt:
        print("\n👋 Chatbot stopped. Goodbye!")
        logging.info("🛑 Chatbot terminated by keyboard interrupt.")
