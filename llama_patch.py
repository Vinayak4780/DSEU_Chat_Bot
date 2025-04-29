# llama_patch.py
import logging
from langchain_community.llms import LlamaCpp
import crewai.llm as crewai_llm

def patch_llama():
    original_call = crewai_llm.LLM.call

    def llama_call(self, prompt, **kwargs):
        # Unwrap and flatten any list prompt
        if isinstance(prompt, list):
            prompt = "\n".join([msg.get("content", "") for msg in prompt if isinstance(msg, dict)])

        # Add final answer instruction
        prompt += "\n\nIMPORTANT:\nRespond in this format only:\nThought: your reasoning\nFinal Answer: your final answer"

        # ✅ Check for wrapped LlamaCpp
        wrapped = getattr(self, "llm", None) or getattr(self, "_llm", None)
        if isinstance(wrapped, LlamaCpp):
            logging.info("[PATCH] Using wrapped LlamaCpp model.")
            return wrapped.invoke(prompt)

        # ✅ Direct LlamaCpp
        if isinstance(self, LlamaCpp):
            logging.info("[PATCH] Using direct LlamaCpp model.")
            return self.invoke(prompt)

        # ❌ Unknown model – fallback gracefully
        logging.warning(f"[PATCH] Unknown LLM type {type(self)}; fallback to original.")
        try:
            return original_call(self, prompt, **kwargs)
        except Exception as e:
            return f"Thought: Failed.\nFinal Answer: LLM failed internally. ({str(e)})"

    crewai_llm.LLM.call = llama_call
