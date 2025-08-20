# response_generator.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class ResponseGenerator:
    """
    A class to handle the loading of a language model and the generation of answers
    based on a query and provided context.
    """
    def __init__(self, model_name: str = "HuggingFaceH4/zephyr-7b-beta"):
        """
        Initializes the generator by loading the model and tokenizer.

        Args:
            model_name (str): The name of the Hugging Face model to use.
        """
        print(f"Loading generator model '{model_name}'...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        if self.device == "cuda":
            # Configure 4-bit quantization to load the large model efficiently on GPU
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto", # Automatically use GPU
            )
        else:
            # Load the model without quantization for CPU
            print("CUDA not available. Loading model on CPU. This may be slow and memory-intensive.")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name
            ).to(self.device)

        print(f"Model loaded successfully on device: {self.device}")

    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        """
        Generates a final answer using the retrieved context.

        Args:
            query (str): The user's original question.
            retrieved_chunks (list[dict]): A list of dictionaries, each containing a text chunk.

        Returns:
            str: The generated, cleaned-up answer.
        """
        context_passages = [chunk['text'] for chunk in retrieved_chunks]
        
        max_length = 4096 
        packed_context = ""
        for passage in context_passages:
            if len(self.tokenizer.encode(packed_context + passage)) < max_length - 512: # Reserve space for prompt/query
                packed_context += passage + "\n\n"
            else:
                break
        
        if not packed_context.strip():
            return "Could not generate an answer because no relevant context was found."

        # Using the prompt format for Zephyr models
        messages = [
            {
                "role": "system",
                "content": "You are a friendly and helpful assistant that answers questions based on the provided context.",
            },
            {
                "role": "user",
                "content": f"Context:\n{packed_context}\n\nBased on the context provided, please answer the following question:\n{query}",
            },
        ]
        
        final_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)

        output_sequences = self.model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        generated_text = self.tokenizer.decode(output_sequences[0], skip_special_tokens=True)
        
        # Clean the output to remove the prompt part
        # The answer is what comes after the final <|assistant|> tag
        parts = generated_text.split("<|assistant|>\n")
        if len(parts) > 1:
            return parts[-1].strip()
        else:
            # Fallback if the model doesn't follow the template perfectly
            # This can happen if the generated text is very short
            user_prompt_end = "please answer the following question:\n" + query
            if user_prompt_end in generated_text:
                return generated_text.split(user_prompt_end)[-1].strip()
            return "Could not extract a clear answer from the model's response."
