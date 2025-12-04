"""Google Gemini LLM client using GCP credits (no OpenAI needed)."""

from typing import Dict, List, Optional, Union
from websocietysimulator.llm import LLMBase
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import logging
import os
import time
from dotenv import load_dotenv
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    GOOGLE_EMBEDDINGS_AVAILABLE = True
except ImportError:
    GOOGLE_EMBEDDINGS_AVAILABLE = False

load_dotenv()
logger = logging.getLogger("websocietysimulator")


class GoogleGeminiLLM(LLMBase):
    """Google Gemini LLM with embeddings - uses only Google APIs."""
    
    def __init__(
        self, 
        api_key: str = None,
        model: str = "gemini-2.0-flash",
        embedding_model: str = "models/embedding-001"
    ):
        """
        Initialize Google Gemini LLM.
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: "gemini-2.0-flash" (default), "gemini-1.5-flash", "gemini-1.5-pro"
            embedding_model: Google embedding model
        """
        super().__init__(model)
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError(
                "Google API key required. Get at: https://aistudio.google.com/app/apikey"
            )
        
        genai.configure(api_key=self.api_key)
        self.safety_settings = {
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
        self.client = genai.GenerativeModel(model, safety_settings=self.safety_settings)
        
        if GOOGLE_EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model=embedding_model, google_api_key=self.api_key
                )
            except Exception:
                self.embedding_model = None
        else:
            self.embedding_model = None
    
    @retry(
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted),
        wait=wait_exponential(multiplier=2, min=5, max=120),
        stop=stop_after_attempt(6),
    )
    def __call__(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
        stop_strs: Optional[List[str]] = None,
        n: int = 1
    ) -> Union[str, List[str]]:
        """Call Gemini API with retry logic for rate limits."""
        time.sleep(3)
        client = genai.GenerativeModel(model) if model and model != self.model else self.client
        
        gemini_messages = []
        system_instruction = None
        for msg in messages:
            role, content = msg.get('role', 'user'), msg.get('content', '')
            if role == 'system':
                system_instruction = content
            elif role == 'user':
                gemini_messages.append({'role': 'user', 'parts': [content]})
            elif role == 'assistant':
                gemini_messages.append({'role': 'model', 'parts': [content]})
        
        config = genai.types.GenerationConfig(temperature=temperature, max_output_tokens=max_tokens)
        
        responses = []
        for _ in range(n):
            try:
                if system_instruction:
                    temp_client = genai.GenerativeModel(
                        model or self.model, system_instruction=system_instruction,
                        safety_settings=self.safety_settings
                    )
                    response = temp_client.generate_content(
                        gemini_messages, generation_config=config, safety_settings=self.safety_settings
                    )
                elif len(gemini_messages) == 1 and gemini_messages[0]['role'] == 'user':
                    response = client.generate_content(
                        gemini_messages[0]['parts'][0], generation_config=config,
                        safety_settings=self.safety_settings
                    )
                else:
                    chat = client.start_chat(history=gemini_messages[:-1])
                    response = chat.send_message(
                        gemini_messages[-1]['parts'][0], generation_config=config,
                        safety_settings=self.safety_settings
                    )
                
                if response.candidates and response.candidates[0].content.parts:
                    responses.append(response.text)
                else:
                    responses.append("[Response blocked by safety filters]")
            except google_exceptions.ResourceExhausted:
                raise
            except Exception:
                raise
        
        return responses[0] if n == 1 else responses
    
    def get_embedding_model(self):
        """Get embedding model (GoogleGenerativeAIEmbeddings or None)."""
        return self.embedding_model
