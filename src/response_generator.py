import os
from dotenv import load_dotenv
from ctransformers import AutoModelForCausalLM
from huggingface_hub import login

class ResponseGenerator:
    """
    Loads the Mistral 7B GGUF language model by downloading it from the
    Hugging Face Hub and running it on the CPU.
    """
    _model = None  # Class-level variable to hold the model singleton

    def __init__(self, model_repo: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF", model_file: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        if ResponseGenerator._model is None:
            load_dotenv()
            hf_token = os.getenv("HUGGING_FACE_HUB_TOKEN")

            if not hf_token:
                raise ValueError("Hugging Face token not found. Please add it to your .env file.")

            print("Authenticating with Hugging Face Hub...")
            login(token=hf_token)
            # FIX: Replaced emoji with standard text to prevent encoding errors
            print("(SUCCESS) Authentication successful.")

            print(f"Downloading/loading model: {model_repo}/{model_file}")
            try:
                # Load the model only once and store it in the class variable
                ResponseGenerator._model = AutoModelForCausalLM.from_pretrained(
                    model_path_or_repo_id=model_repo,
                    model_file=model_file,
                    model_type="mistral",
                    gpu_layers=0,  # Set to 0 for CPU to avoid CUDA errors
                    context_length=4096,
                )
                # FIX: Replaced emoji with standard text
                print("(SUCCESS) Model loaded successfully on CPU.")
            except Exception as e:
                print(f"--- ERROR LOADING MODEL ---: {e}")
                raise
        
        self.model = ResponseGenerator._model


    def generate(self, query: str, retrieved_chunks: list[dict]) -> str:
        context_passages = [c["text"] for c in retrieved_chunks]
        packed_context = "\n".join(context_passages)

        if not packed_context.strip():
            return "Could not generate an answer because no relevant context was found."

        user_prompt = (
            f"[INST] Answer the question based on the context.\n\n"
            f"Context:\n{packed_context.strip()}\n\n"
            f"Question: {query} [/INST]"
        )

        answer = self.model(
            user_prompt,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.95,
        )

        return answer.strip() if answer else "The model generated an empty response."