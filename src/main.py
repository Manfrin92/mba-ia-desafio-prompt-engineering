from dotenv import load_dotenv
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from metrics import evaluate_response, print_metrics_report

load_dotenv()

client = Client()

# Esse Prompt é simplesmente: "Como fazer uma API"
promptName = "python_api_explanation"

prompt = client.pull_prompt(promptName)

# LLM usado para GERAR as respostas que serão avaliadas (Gemini via GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

# Contexto FAKE de usuário mock somente para exemplificar + regras de estilo + cobertura de tópicos obrigatória
system_context = """Você é um assistente técnico sênior especializado em Python e desenvolvimento de APIs.

Perfil do usuário:
- Mais de 5 anos de experiência criando APIs
- Stack principal: .NET (C#)
- Conhecimento intermediário de Node.js
- NÃO é iniciante — não precisa de explicações de conceitos básicos de programação ou "o que é uma API"
- Atualmente na sua empresa estão migrando tudo para Python 3 por isso é fundamental que as respostas sejam apenas assumindo que ele está trabalhando com Python no momento

Diretrizes de resposta:
- Seja direto e sucinto
- Evite conceitos básicos; vá direto ao ponto técnico
- Compare cada conceito Python com o equivalente em .NET (e em Node.js quando fizer sentido)
- Mostre equivalências práticas entre frameworks (ex: ASP.NET Core Web API ↔ FastAPI, [ApiController]/DTOs ↔ Pydantic models, Kestrel ↔ Uvicorn)

A resposta deve obrigatoriamente cobrir:
- Python 3 (setup mínimo do projeto)
- FastAPI como framework
- Criação de endpoints (rotas e métodos HTTP)
- Conceitos de HTTP/REST aplicados
- Pydantic para validação/serialização
- Execução via servidor ASGI (Uvicorn)
"""

original_human_message = prompt.messages[-1].prompt.template

enriched_prompt = ChatPromptTemplate.from_messages([
    ("system", system_context),
    ("human", original_human_message),
])

print("Executando avaliação dos prompts...")

# ------------------------------------------------------------------
# 1) Avaliação do prompt ORIGINAL (genérico, sem contexto de usuário)
# ------------------------------------------------------------------
resposta_original = (prompt | llm).invoke({}).content

resultado_original = evaluate_response(resposta_original, user_context=None)
print_metrics_report(resultado_original, prompt_label=f"{promptName} (original)")

# ------------------------------------------------------------------
# 2) Avaliação do prompt ENRIQUECIDO (com contexto de usuário)
# ------------------------------------------------------------------
resposta_enriquecida = (enriched_prompt | llm).invoke({}).content

resultado_enriquecido = evaluate_response(resposta_enriquecida, user_context=system_context)
print_metrics_report(resultado_enriquecido, prompt_label=f"{promptName} (enriquecido)")

# Fazer push dos prompts otimizados de volta ao LangSmith
url = client.push_prompt(
    promptName,
    object=enriched_prompt,
    description=(
        "Prompt otimizado com contexto de usuário sênior (.NET/Node.js): "
        "system prompt com role, preferências de estilo e cobertura garantida "
        "de tópicos-chave (FastAPI, Pydantic, Uvicorn, REST) para maximizar "
        "Helpfulness, Correctness, Clarity, Precision e Recall."
    ),
)
print(url)