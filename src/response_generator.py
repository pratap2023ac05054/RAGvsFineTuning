# response_generator.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ResponseGenerator:
    """Loads a GPT-2 model and tokenizer for text generation."""

    def __init__(self, model_name: str = "openai-community/gpt2-medium"):
        print(f"Loading generator model '{model_name}'...")
        
        # Determine the device to use (GPU if available, otherwise CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer and model from Hugging Face
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            raise

        # Set the pad token to the end-of-sequence token if it's not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Get the maximum context length from the model's configuration
        self.max_positions = self.model.config.n_positions
        print(f"Model and Tokenizer loaded. Max positions: {self.max_positions}")

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Builds a prompt that respects the model's context window by iteratively
        adding context chunks until the token limit is reached.
        """
        # Reserve 250 tokens for the model to generate its response
        max_input_tokens = self.max_positions - 250

        # A simple and effective prompt template for GPT-2 in a RAG setup
        prompt_tmpl = (
            "Use the context provided below to answer the user's question. "
            "If the context does not contain the answer, state that you don't know.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )

        # 1. Calculate tokens used by the prompt template and query
        base_prompt = prompt_tmpl.format(context="", question=query)
        base_tokens = self.tokenizer(base_prompt, return_tensors="pt")['input_ids']
        available_tokens_for_context = max_input_tokens - base_tokens.shape[1]

        # 2. Iteratively build context until available tokens are used
        context_passages = []
        current_token_count = 0
        for chunk in retrieved_chunks:
            chunk_text = chunk["text"] + "\n"
            chunk_tokens = self.tokenizer(chunk_text)['input_ids']

            if current_token_count + len(chunk_tokens) <= available_tokens_for_context:
                context_passages.append(chunk_text)
                current_token_count += len(chunk_tokens)
            else:
                break # Stop if adding the next chunk would exceed the limit

        final_context = "".join(context_passages)

        # 3. Format the final prompt with the curated context
        final_prompt = prompt_tmpl.format(context=final_context, question=query)

        # 4. Generate the response using the transformers `generate` method
        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
        
        output_sequences = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Decode the output and extract only the newly generated text
        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        # Find the start of the answer and slice the string
        answer_start_pos = generated_text.find("Answer:") + len("Answer:")
        output_text = generated_text[answer_start_pos:]


        answer = (output_text or "").strip()
        if not answer:
            answer = "I could not generate a confident answer with the provided context."
        return answer