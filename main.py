import asyncio
import os
from dotenv import load_dotenv

from schemas import ChatMessage, ModelConfig
from clients import BaseLLMClient, OpenAIClient, AnthropicClient

async def run_client_demo(client_name: str, client: BaseLLMClient, messages: list[ChatMessage]):
    print(f"\n{'='*50}\nIniciando test para: {client_name}\n{'='*50}")
    
    print(f"\n--- [1] Generación Estándar ({client_name}) ---")
    response = await client.generate(messages)
    
    if response.error:
        print(f"Error capturado de forma segura: {response.error}")
    else:
        print(f"Resultado: {response.content}\nMetadatos de Uso: {response.usage}")
    
    print(f"\n--- [2] Generación por Streaming ({client_name}) ---")
    print("Resultado: ", end="", flush=True)
    
    async for chunk in client.stream(messages):
        print(chunk, end="", flush=True)
    
    print("\n")

async def main():
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    messages = [
        ChatMessage(role="system", content="Eres un asistente conciso."),
        ChatMessage(role="user", content="¿Qué es la entropía? Explícalo en una sola oración.")
    ]
    
    tasks = []
    
    if openai_key:
        openai_config = ModelConfig(model="gpt-4o-mini", temperature=0.5)
        openai_client = OpenAIClient(api_key=openai_key, config=openai_config)
        tasks.append(run_client_demo("OpenAI (gpt-4o-mini)", openai_client, messages))
    else:
        print("Aviso: OPENAI_API_KEY no encontrada.")

    if anthropic_key:
        anthropic_config = ModelConfig(model="claude-3-haiku-20240307", temperature=0.5)
        anthropic_client = AnthropicClient(api_key=anthropic_key, config=anthropic_config)
        tasks.append(run_client_demo("Anthropic (claude-3-haiku)", anthropic_client, messages))
    else:
        print("Aviso: ANTHROPIC_API_KEY no encontrada.")
        
    if tasks:
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
