import os
import torch
import streamlit as st
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

class ResponseGenerator:
    """
    Loads a GGUF language model using llama-cpp-python for generation.
    """
    def __init__(self, model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"):
        hf_token = st.secrets.get("HF_TOKEN")
        if not hf_token:
            st.warning("Warning: Hugging Face token not found in Streamlit secrets.")

        print(f"Downloading model file for '{model_name}'...")
        model_path = hf_hub_download(
            repo_id=model_name,
            filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            token=hf_token
        )

        print(f"Loading generator model from '{model_path}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        CONTEXT_LENGTH = 4096

        self.model = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_LENGTH,
            n_gpu_layers=-1 if self.device == "cuda" else 0,
            verbose=False
        )
        
        print("Loading tokenizer...")
        tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
        
        self.max_positions = CONTEXT_LENGTH
        print(f"Model and Tokenizer loaded. Max positions: {self.max_positions}")

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Generates a final answer using the retrieved context.
        """
        context_passages = [c["text"] for c in retrieved_chunks]
        
        # --- FIX: Removed the leading "<s>" token ---
        # The llama-cpp-python library adds this automatically.
        prompt_tmpl = (
            "[INST] Answer the following question based on the context provided below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question} [/INST]"
        )

        final_prompt = prompt_tmpl.format(context="\n".join(context_passages), question=query)
        
        response = self.model.create_completion(
            prompt=final_prompt,
            max_tokens=250,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["</s>", "[/INST]"]
        )
        
        answer = response['choices'][0]['text']
        
        if not answer:
            answer = "I couldn’t produce a confident answer with the provided context."
            
        return answer.strip()