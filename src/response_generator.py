import os
import torch
import streamlit as st
from transformers import AutoTokenizer
from ctransformers import AutoModelForCausalLM

class ResponseGenerator:
    """
    Loads a GGUF language model for generation and a separate transformers
    tokenizer for reliable text processing.
    """
    def __init__(self, model_name: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"):
        # Read token from Streamlit's secrets
        hf_token = st.secrets.get("HF_TOKEN")
        if not hf_token:
            st.warning("Warning: Hugging Face token not found in Streamlit secrets.")

        print(f"Loading generator model '{model_name}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        CONTEXT_LENGTH = 4096

        # The ctransformers library is used to load the GGUF model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            model_type="mistral",
            gpu_layers=50 if self.device == "cuda" else 0,
            hf=True,
            context_length=CONTEXT_LENGTH
        )
        
        print("Loading separate tokenizer...")
        tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
        
        # Set a pad token to enable correct attention masking if it's missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_positions = CONTEXT_LENGTH
        print(f"Model and Tokenizer loaded. Max positions: {self.max_positions}")

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Generates a final answer using the retrieved context.
        """
        context_passages = [c["text"] for c in retrieved_chunks]
        prompt_tmpl = (
            "<s>[INST] Answer the following question based on the context provided below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question} [/INST]"
        )

        final_prompt = prompt_tmpl.format(context="\n".join(context_passages), question=query)
        
        # Tokenize the final prompt, returning PyTorch tensors
        inputs = self.tokenizer(
            final_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=self.max_positions - 250 # Leave room for generation
        )

        # We select the first (and only) item from the batch ([0]) to convert
        # the 2D tensor of shape [1, sequence_length] to the 1D tensor of
        # shape [sequence_length] that the model expects.
        input_ids_tensor = inputs['input_ids'][0]
        
        output_token_ids = self.model(
            input_ids_tensor,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            stop_tokens=[self.tokenizer.eos_token_id]
        )
        
        answer = self.tokenizer.decode(output_token_ids, skip_special_tokens=True)
        
        if not answer:
            answer = "I couldn’t produce a confident answer with the provided context."
            
        return answer.strip()