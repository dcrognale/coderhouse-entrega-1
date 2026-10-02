# Unified Async LLM Client

Este proyecto proporciona un cliente asíncrono unificado en Python 3.12 para interactuar con los modelos de OpenAI y Anthropic mediante una interfaz común.

## Estructura del proyecto
- `schemas.py`: Modelos Pydantic para los contratos de datos.
- `clients.py`: Implementación de los clientes base, OpenAI y Anthropic.
- `main.py`: Script de prueba de los clientes (estándar y streaming).
- `.env`: Archivo para configurar las variables de entorno de forma segura.

## Instalación y ejecución
1. Ejecuta `python -m venv venv` para crear un entorno virtual.
2. Activa el entorno (`source venv/bin/activate` en Unix o `venv\Scripts\activate` en Windows).
3. Instala las dependencias con `pip install -r requirements.txt`.
4. Rellena el archivo `.env` con tus claves API.
5. Ejecuta `python main.py`.
