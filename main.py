# import os
# import logging
# import re
# import string
# import concurrent.futures
# import threading
# from crewai import Crew
# from agents.summrization import summary_agent, summary_task, summrization_bot
# from agents.links import dLinkSearchAgent, Link_task, Link_predict
# from agents.forms_agent import Form_Agent, FormsTask, forms_predict
# from agents.faculty import FacultySearchAgent, faculty_task, search_faculty
# from agents.campus_agent import CampusSearchAgent, campus_Task, campus_agent_bot
# from agents.Syllabus import syllabus_agent, syllabus_task, syllabus_bot
# from pathlib import Path
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk
# from autocorrect import Speller
# os.environ["NLTK_DATA"] = "/app/nltk_data"

# from llama_patch import patch_llama
# patch_llama()

# nltk.download('punkt')
# nltk.download('stopwords')

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[
#         logging.FileHandler("chatbot.log", mode="a"),
#         logging.StreamHandler()
#     ],
#     force=True       # <<— this forces reconfiguration
# )

# logging.info("Chatbot is starting...")

# MultimodalCrew = Crew(
#     agents=[
#         summary_agent,
#         dLinkSearchAgent,
#         Form_Agent,
#         FacultySearchAgent,
#         CampusSearchAgent,
#         syllabus_agent
#     ],
#     description="A multimodal AI chatbot that handles summarization, links, forms, faculty details, and campus queries.",
#     goal="Provide an intelligent, context-aware AI chatbot that efficiently answers user queries."
# )

# handler_keywords = {
#     Link_predict: ["link", "website", "site", "url"],
#     forms_predict: ["form", "document", "application", "pdf"],
#     search_faculty: ["name", "teacher", "professor", "mail", "email", "vice chancellor", "pro vice chancellor",
#                      "controller finance", "deputy registrar", "assistant registrar", "training placement officer",
#                      "junior mechanic", "assistant professor", "associate professor", "store keeper",
#                      "caretaker", "campus director", "assistant section officer", "office superintendent",
#                      "lecturer", "workshop instructor", "senior assistant", "executive engineer",
#                      "workshop attendant", "senior mechanic", "senior accounts officer", "accounts officer",
#                      "assistant accounts officer", "deputy director technical", "assistant director technical",
#                       "department"],
#     campus_agent_bot: ["campus", "university", "college", "institute", "course related", "location",
#                        "lab", "photo campus", "about campus"],
#     syllabus_bot: [ "lecture credit", "tutorial credit",
#                    "practical credit", "credit",'syllabus']
# }

# # Semaphore to limit concurrent access to GPU
# gpu_semaphore = threading.BoundedSemaphore(value=1)

# def wrap_with_semaphore(func):
#     def wrapped(text):
#         with gpu_semaphore:
#             return func(text)
#     return wrapped

# # Wrapped agent functions
# async_link_predict = wrap_with_semaphore(Link_predict)
# async_forms_predict = wrap_with_semaphore(forms_predict)
# async_search_faculty = wrap_with_semaphore(search_faculty)
# async_campus_bot = wrap_with_semaphore(campus_agent_bot)
# async_summarize = wrap_with_semaphore(summrization_bot)
# handle_syllabus = wrap_with_semaphore(syllabus_bot)

# def handle_user_input_parallel(user_input):
#     logging.info(f"User Query: {user_input}")
#     user_input = user_input.lower()

#     with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
#         tasks = []

#         # Queue tasks based on keyword matches
#         if any(kw in user_input for kw in handler_keywords[Link_predict]):
#             tasks.append(async_link_predict)

#         if any(kw in user_input for kw in handler_keywords[forms_predict]):
#             tasks.append(async_forms_predict)

#         if any(kw in user_input for kw in handler_keywords[search_faculty]):
#             tasks.append(async_search_faculty)

#         if any(kw in user_input for kw in handler_keywords[campus_agent_bot]):
#             tasks.append(async_campus_bot)

#         if any(kw in user_input for kw in handler_keywords[syllabus_bot]):
#             tasks.append(handle_syllabus)

#         results = []

#         # Run one task at a time, in the same thread pool (sequential execution)
#         for task in tasks:
#             future = executor.submit(task, user_input)
#             try:
#                 result = future.result(timeout=120)  # Wait until done
#                 if result:
#                     results.append(result)
#             except Exception as e:
#                 logging.warning(f"Agent failed: {e}")

#         # If no match, fallback to summarization
#         if not results:
#             try:
#                 results.append(async_summarize(user_input))
#             except Exception as e:
#                 logging.error(f"Fallback summarizer failed: {e}")
#                 results.append("Sorry, I couldn't understand that.")

#     final_response = "\n".join([f"{i+1}. {res}" for i, res in enumerate(results)])
#     logging.info(f"Chatbot Response: {final_response}")
#     return final_response

# def preprocess_text(text):
#     spell = Speller(lang='en')
#     text = text.lower()
#     text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
#     text = " ".join([spell(word) for word in text.split()])
#     tokens = word_tokenize(text)
#     stop_words = set(stopwords.words("english"))
#     tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
#     return " ".join(tokens)

# if __name__ == "__main__":
#     print("\n\U0001F916 AI Chatbot Ready! Type 'exit' to quit.\n")

#     try:
#         while True:
#             user_input = input("You: ").strip()
#             user_input = preprocess_text(user_input)
#             if user_input.lower() in ["exit", "quit", "bye"]:
#                 print("Chatbot: Goodbye! \U0001F44B")
#                 break

#             response = handle_user_input_parallel(user_input)
#             print(f"Chatbot:\n{response}\n")

#     except KeyboardInterrupt:
#         print("\n\U0001F44B Chatbot stopped. Goodbye!")
#         logging.info("Chatbot terminated with keyboard interrupt.") 





























import os
import logging
import re
import string
import concurrent.futures
import threading
from crewai import Crew
from agents.summrization import summary_agent, summary_task, summrization_bot
from agents.links import dLinkSearchAgent, Link_task, Link_predict
from agents.forms_agent import Form_Agent, FormsTask, forms_predict
from agents.faculty import FacultySearchAgent, faculty_task, search_faculty
from agents.campus_agent import CampusSearchAgent, campus_Task, campus_agent_bot
from agents.Syllabus import syllabus_agent, syllabus_task, syllabus_bot
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from autocorrect import Speller

# Ensure NLTK data location
os.environ["NLTK_DATA"] = "/app/nltk_data"

from llama_patch import patch_llama
patch_llama()

nltk.download('punkt')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chatbot.log", mode="a"),
        logging.StreamHandler()
    ],
    force=True
)
logging.info("Chatbot is starting...")

# Initialize Crew with agents
MultimodalCrew = Crew(
    agents=[
        summary_agent,
        dLinkSearchAgent,
        Form_Agent,
        FacultySearchAgent,
        CampusSearchAgent,
        syllabus_agent
    ],
    description="A multimodal AI chatbot that handles summarization, links, forms, faculty details, and campus queries.",
    goal="Provide an intelligent, context-aware AI chatbot that efficiently answers user queries."
)

# Keyword mappings for handlers
handler_keywords = {
    Link_predict: ["link", "website", "site", "url"],
    forms_predict: ["form", "document", "application", "pdf"],
    search_faculty: [
        "name", "teacher", "professor", "mail", "email", "vice chancellor",
        "pro vice chancellor", "controller finance", "deputy registrar",
        "assistant registrar", "training placement officer", "junior mechanic",
        "assistant professor", "associate professor", "store keeper", "caretaker",
        "campus director", "assistant section officer", "office superintendent",
        "lecturer", "workshop instructor", "senior assistant", "executive engineer",
        "workshop attendant", "senior mechanic", "senior accounts officer",
        "accounts officer", "assistant accounts officer", "deputy director technical",
        "assistant director technical", "department"
    ],
    campus_agent_bot: ["campus", "university", "college", "institute", "course related", "location", "lab", "photo campus", "about campus"],
    syllabus_bot: ["lecture credit", "tutorial credit", "practical credit", "credit", "syllabus"]
}

# Semaphore to serialize GPU access
gpu_semaphore = threading.BoundedSemaphore(value=1)

def wrap_with_semaphore(func):
    def wrapped(text):
        with gpu_semaphore:
            return func(text)
    return wrapped

# Wrapped agent callables
async_link_predict   = wrap_with_semaphore(Link_predict)
async_forms_predict  = wrap_with_semaphore(forms_predict)
async_search_faculty = wrap_with_semaphore(search_faculty)
async_campus_bot     = wrap_with_semaphore(campus_agent_bot)
async_summarize      = wrap_with_semaphore(summrization_bot)
handle_syllabus      = wrap_with_semaphore(syllabus_bot)

def handle_user_input_parallel(user_input: str) -> str:
    logging.info(f"User Query: {user_input}")
    text = user_input.lower()

    # 1. Build the list of matched tasks
    tasks = []
    if any(kw in text for kw in handler_keywords[Link_predict]):
        tasks.append(async_link_predict)
    if any(kw in text for kw in handler_keywords[forms_predict]):
        tasks.append(async_forms_predict)
    if any(kw in text for kw in handler_keywords[search_faculty]):
        tasks.append(async_search_faculty)
    if any(kw in text for kw in handler_keywords[campus_agent_bot]):
        tasks.append(async_campus_bot)
    if any(kw in text for kw in handler_keywords[syllabus_bot]):
        tasks.append(handle_syllabus)

    # 2. If no handlers matched, use fallback summarizer
    if not tasks:
        try:
            response = async_summarize(text)
            logging.info(f"Chatbot Response: {response}")
            return response
        except Exception as e:
            logging.error(f"Fallback summarizer failed: {e}")
            return "Sorry, I couldn't understand that."

    results = []
    # 3. Execute each matched task in its own thread, up to timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_to_task = {executor.submit(task, text): task for task in tasks}
        done, not_done = concurrent.futures.wait(
            future_to_task.keys(),
            timeout=120,
            return_when=concurrent.futures.ALL_COMPLETED
        )
        # Collect successful results
        for fut in done:
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                logging.warning(f"{future_to_task[fut].__name__} failed: {e}")
        # Cancel any that timed out
        for fut in not_done:
            fut.cancel()
            logging.warning(f"{future_to_task[fut].__name__} timed out and was cancelled")

    # 4. Aggregate numbered responses
    final_response = "\n".join(f"{i+1}. {r}" for i, r in enumerate(results))
    logging.info(f"Chatbot Response: {final_response}")
    return final_response

def preprocess_text(text: str) -> str:
    spell = Speller(lang='en')
    text = text.lower()
    text = re.sub(rf"[{re.escape(string.punctuation)}]", " ", text)
    text = " ".join(spell(w) for w in text.split())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [w for w in tokens if w.isalnum() and w not in stop_words]
    return " ".join(filtered)

if __name__ == "__main__":
    print("\n\U0001F916 AI Chatbot Ready! Type 'exit' to quit.\n")
    try:
        while True:
            user_input = input("You: ").strip()
            clean = preprocess_text(user_input)
            if clean.lower() in {"exit", "quit", "bye"}:
                print("Chatbot: Goodbye! \U0001F44B")
                break
            response = handle_user_input_parallel(clean)
            print(f"Chatbot:\n{response}\n")
    except KeyboardInterrupt:
        print("\n\U0001F44B Chatbot stopped. Goodbye!")
        logging.info("Chatbot terminated with keyboard interrupt.")
