from dotenv import load_dotenv
from langsmith import Client

# Load environment variables from .env file
load_dotenv()

client = Client()

promptName = "python_api_explanation"

# Fazer pull de prompts do LangSmith Prompt Hub contendo prompts de baixa qualidade
prompt = client.pull_prompt(promptName)

# Analyze the quality of the prompt right now, if it is not that good, optimize it
# Usually the model that makes the prompt better is cheaper and not same used to give the response

"""
Prompt Enrichement: given more context about who is this user,

I'm pretending it is someone,

Contexto:
- 5+ anos criando APIs
- Principal experiência: .NET
- Conhecimento de Node.js

Preferências:
- Resposta sucinta
- Evitar conceitos básicos
- Comparar Python com .NET
- Mostrar equivalências entre frameworks/conceitos"

Refatorar e otimizar esses prompts usando técnicas avançadas de Prompt Engineering
Avaliar a qualidade através de métricas customizadas 

(Helpfulness, Correctness, F1-Score, Clarity, Precision)
Atingir pontuação mínima de 0.8 (80%) em todas as métricas de avaliação

Helpfulness - a resposta atende bem ao objetivo/contexto do usuário?
Correctness - os conceitos/informações estão tecnicamente corretos?
F1 - equilíbrio entre Precision e Recall.
Clarity - a explicação é clara?
Precision - dos conceitos mencionados pela resposta, quantos são relevantes?
Recall - dos conceitos esperados, quantos foram mencionados?

uma boa resposta deve mencionar:
- Python 3
- FastAPI
- criação de endpoints
- HTTP/REST
- Pydantic
- servidor ASGI/Uvicorn

"""

# Fazer push dos prompts otimizados de volta ao LangSmith
# url = client.push_prompt(
#     promptName, 
#     object=prompt_template, 
#     description=prompt.description,
# )
# print(url)
