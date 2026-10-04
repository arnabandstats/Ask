"""LLM client factory.

`client_create()` is copied verbatim from the original single-file app
(lines 763-806), together with the environment globals it reads. Every LLM
call in the app goes through it (see ask/agent/llm_call.py).
"""
from __future__ import annotations

import os

import openai
from dotenv import load_dotenv

load_dotenv()

USE_AZURE = os.getenv("USE_AZURE_OPENAI", "false").lower() == "true"


def client_create():
    """
    Return an OpenAI-compatible client. The ONLY place the client is instantiated.
    Shared by the repo assessor and the Pandas Chatbot.

    Standard OpenAI  → set USE_AZURE_OPENAI=false (default)
                        requires: OPENAI_API_KEY

    Azure OpenAI     → set USE_AZURE_OPENAI=true
                        requires: AZURE_OPENAI_TOKEN  (or AZURE_OPENAI_KEY)
                                  AZURE_OPENAI_BASE_URL
                                  AZURE_OPENAI_VERSION  (default: 2024-02-01)
                        MODEL / OPENAI_DEPLOYMENT_NAME must match your deployment name.
    """
    if USE_AZURE:
        from openai import AzureOpenAI
        api_key  = (os.getenv("AZURE_OPENAI_TOKEN")
                    or os.getenv("AZURE_OPENAI_KEY")
                    or os.getenv("OPENAI_API_KEY"))
        endpoint = os.getenv("AZURE_OPENAI_BASE_URL")
        version  = os.getenv("AZURE_OPENAI_VERSION", "2024-02-01")
        if not api_key:
            raise EnvironmentError(
                "Azure OpenAI key not found. "
                "Set AZURE_OPENAI_TOKEN (or AZURE_OPENAI_KEY) in your environment."
            )
        if not endpoint:
            raise EnvironmentError(
                "AZURE_OPENAI_BASE_URL not set. "
                "Set it to your Azure OpenAI endpoint URL."
            )
        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=version,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not found. "
                "Add it to your .env file as:  OPENAI_API_KEY=sk-..."
            )
        return openai.OpenAI(api_key=api_key)
