# response_generator.py

import streamlit as st
from transformers import AutoTokenizer
from ctransformers import AutoModelForCausalLM

class ResponseGenerator:
    """Loads a GGUF language model and tokenizer for text generation."""
    
    def __init__(self, model_name: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"):
        hf_token = st.secrets.get("HF_TOKEN")
        if not hf_token:
            st.warning("Hugging Face token not found in Streamlit secrets.")

        print(f"Loading generator model '{model_name}'...")
        CONTEXT_LENGTH = 2048

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            model_file="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            model_type="llama",
            gpu_layers=0, 
            context_length=CONTEXT_LENGTH
        )
        
        print("Loading tokenizer...")
        tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.max_positions = CONTEXT_LENGTH
        print(f"Model and Tokenizer loaded. Max positions: {self.max_positions}")

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        context_passages = [c["text"] for c in retrieved_chunks]
        
        # TinyLlama chat format for RAG
        prompt_tmpl = (
            "<|system|>\n"
            "You are a helpful assistant. Use the context provided to answer the user's question.</s>\n"
            "<|user|>\n"
            "Context: {context}\n\n"
            "Question: {question}</s>\n"
            "<|assistant|>"
        )
        final_prompt = prompt_tmpl.format(
            context="\n".join(context_passages),
            question=query
        )

        # Ensure prompt fits within the model's context window
        enc = self.tokenizer(final_prompt, return_tensors=None, add_special_tokens=False)
        max_input = self.max_positions - 250 # Reserve 250 tokens for generation
        if len(enc["input_ids"]) > max_input:
            trimmed_ids = enc["input_ids"][-max_input:]
            final_prompt = self.tokenizer.decode(trimmed_ids, skip_special_tokens=True)

        output_text = self.model(
            final_prompt,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
            stop=[self.tokenizer.eos_token]
        )
        
        answer = (output_text or "").strip()
        if not answer:
            answer = "I could not generate a confident answer with the provided context."
        return answer