# response_generator.py

import streamlit as st
from transformers import AutoTokenizer
# MODIFIED: Removed ctransformers and added llama-cpp-python components
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

class ResponseGenerator:
    """Loads a GGUF language model and tokenizer for text generation."""

    def __init__(self, model_name: str = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"):
        hf_token = st.secrets.get("HF_TOKEN")
        if not hf_token:
            st.warning("Hugging Face token not found in Streamlit secrets.")

        print(f"Loading generator model '{model_name}'...")
        CONTEXT_LENGTH = 2048
        MODEL_FILE = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

        # MODIFIED: Download the GGUF model file from Hugging Face Hub
        # This will cache the model file locally, preventing re-downloads
        try:
            model_path = hf_hub_download(
                repo_id=model_name,
                filename=MODEL_FILE,
                token=hf_token
            )
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            raise

        # MODIFIED: Load the GGUF model using llama-cpp-python.
        # The n_gpu_layers=0 parameter ensures the model runs on the CPU.
        self.model = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_LENGTH,
            n_gpu_layers=0, # Use CPU
            n_batch=512,
            verbose=False,
        )

        # Tokenizer is still needed for prompt construction and token counting
        print("Loading tokenizer...")
        tokenizer_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=hf_token)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_positions = CONTEXT_LENGTH
        print(f"Model and Tokenizer loaded. Max positions: {self.max_positions}")

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Builds a prompt that respects the model's context window by iteratively
        adding context chunks until the token limit is reached.
        """
        # Reserve 250 tokens for the model to generate its response
        max_input_tokens = self.max_positions - 250

        # TinyLlama chat format for RAG
        prompt_tmpl = (
            "<|system|>\n"
            "You are a helpful assistant. Use the context provided to answer the user's question.</s>\n"
            "<|user|>\n"
            "Context: {context}\n\n"
            "Question: {question}</s>\n"
            "<|assistant|>"
        )

        # 1. Calculate tokens used by the prompt template and query
        base_prompt = prompt_tmpl.format(context="", question=query)
        base_tokens = self.tokenizer(base_prompt, add_special_tokens=False)['input_ids']
        available_tokens_for_context = max_input_tokens - len(base_tokens)

        # 2. Iteratively build context until available tokens are used
        context_passages = []
        current_token_count = 0
        for chunk in retrieved_chunks:
            chunk_text = chunk["text"] + "\n"
            chunk_tokens = self.tokenizer(chunk_text, add_special_tokens=False)['input_ids']

            if current_token_count + len(chunk_tokens) <= available_tokens_for_context:
                context_passages.append(chunk_text)
                current_token_count += len(chunk_tokens)
            else:
                break # Stop if adding the next chunk would exceed the limit

        final_context = "".join(context_passages)

        # 3. Format the final prompt with the curated context
        final_prompt = prompt_tmpl.format(context=final_context, question=query)

        # 4. MODIFIED: Generate the response using llama-cpp-python's interface
        output = self.model(
            final_prompt,
            max_tokens=250,
            temperature=0.7,
            top_p=0.95,
            repeat_penalty=1.1,
            stop=["</s>", "<|user|>", self.tokenizer.eos_token] # More robust stop tokens
        )

        output_text = output['choices'][0]['text']

        answer = (output_text or "").strip()
        if not answer:
            answer = "I could not generate a confident answer with the provided context."
        return answer