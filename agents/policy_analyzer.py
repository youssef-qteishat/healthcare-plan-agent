# agents/policy_analyzer.py
import requests
from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_core.language_models import LLM
import os
from dotenv import load_dotenv
import shutil

load_dotenv()

class HuggingFaceLLM(LLM):
    def _call(self, prompt: str, stop=None, run_manager=None) -> str:
        HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

        headers = {
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json"
        }

        # System prompt definition
        system_prompt = (
            "You are an expert in insurance policy analysis. "
            "Given a user's query about a medical procedure and a healthcare summary of benifits document, "
            "provide a concise answer indicating whether the procedure is covered, and explain your reasoning clearly."
        )

        # Chat-style formatting (Mistral uses `### Instruction:` or similar prompt templates)
        full_prompt = f"""[INST] <<SYS>>
                    {system_prompt}
                    <</SYS>>
                    {prompt} [/INST]"""

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7
            }
        }

        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()

        data = response.json()
        if isinstance(data, list) and "generated_text" in data[0]:
            return data[0]["generated_text"]
        elif isinstance(data, dict) and "error" in data:
            raise RuntimeError(f"Hugging Face API Error: {data['error']}")
        else:
            raise ValueError(f"Unexpected response format: {data}")

    @property
    def _llm_type(self) -> str:
        return "huggingface-inference-api"


def analyze_policy(surgery: str, policy_text: str) -> str:
    """
    Uses embedding-based RAG to semantically analyze if the user's query (e.g., ACL surgery)
    is covered in the given healthcare policy text.
    """
    persist_directory = os.path.expanduser("~/.chroma_policy_index")

    # Clean up Chroma persistent memory safely
    if os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
        except Exception as e:
            print(f"⚠️ Error deleting Chroma index: {e}")
            raise RuntimeError("Unable to clear Chroma index; disk may be full or write-protected.")


    # os.makedirs(persist_directory, exist_ok=True)

    # Step 1: Split the policy text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([policy_text])

    # Step 2: Create Chroma vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_directory)

    # Step 3: Create retriever
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)

    # Step 4: Load local LLM
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in environment.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe)

    # Step 5: Create QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # Step 6: Ask question
    result = qa_chain.run(f"Does the following insurance policy cover: {surgery}?")

    return result