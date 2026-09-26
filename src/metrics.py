"""
metrics.py

Módulo de avaliação de qualidade para respostas geradas a partir do prompt
"python_api_explanation" no LangSmith.

Métricas Base (determinísticas, calculadas por análise direta do texto):
    - Precision
    - Recall (interna, usada para calcular F1)
    - F1-Score
    - Clarity

Métricas Derivadas (calculadas via LLM-as-judge, que interpreta o texto
para dar um julgamento contextual sobre o usuário/resposta):
    - Helpfulness
    - Correctness

Uso básico:

    from metrics import evaluate_response, print_metrics_report

    resultado = evaluate_response(resposta_do_modelo, user_context=CONTEXTO_USUARIO)
    print_metrics_report(resultado)
"""

import os
import re
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

THRESHOLD = 0.8  # pontuação mínima exigida em todas as métricas


# --------------------------------------------------------------------------
# 1. Dicionários de conceitos (usados nas métricas Base: Precision/Recall/F1)
# --------------------------------------------------------------------------

# Conceitos que uma "boa resposta" DEVE mencionar (usados no Recall)
EXPECTED_CONCEPTS: dict[str, list[str]] = {
    "Python 3": [r"\bpython\s*3\b", r"\bpython\b"],
    "FastAPI": [r"\bfastapi\b", r"\bfast\s*api\b"],
    "Endpoints/Rotas": [
        r"\bendpoint", r"\brota", r"\broute", r"@app\.(get|post|put|delete|patch)",
        r"\bpath\s*operation",
    ],
    "HTTP/REST": [
        r"\bhttp\b", r"\brest(ful)?\b", r"\bstatus\s*code", r"\bget\b", r"\bpost\b",
        r"\bput\b", r"\bdelete\b",
    ],
    "Pydantic": [r"\bpydantic\b", r"\bbasemodel\b"],
    "ASGI/Uvicorn": [r"\buvicorn\b", r"\basgi\b"],
}

# Conceitos "bônus" relevantes dado o perfil do usuário (comparações .NET/Node.js).
# Contam a favor da Precision (são relevantes), mas não entram no cálculo de Recall,
# já que não fazem parte da lista fixa de conceitos esperados.
RELEVANT_COMPARISON_CONCEPTS: dict[str, list[str]] = {
    "Comparação .NET": [r"\.net\b", r"\basp\.net\b", r"\bc#\b", r"\bkestrel\b", r"\bcontroller\b"],
    "Comparação Node.js": [r"\bnode\.?js\b", r"\bexpress\b", r"\bmiddleware\b"],
}

# Conceitos considerados "ruído" para este usuário: básicos demais ou fora de escopo.
# Se aparecerem, penalizam a Precision (mencionados, mas não relevantes para um sênior).
NOISE_CONCEPTS: dict[str, list[str]] = {
    "Explicação básica de programação": [
        r"o que é uma vari[aá]vel", r"o que é um loop", r"o que é uma fun[cç][aã]o",
        r"introdu[cç][aã]o a programa[cç][aã]o",
    ],
    "O que é uma API (básico)": [r"o que é uma api\b", r"api significa"],
    "Frameworks fora de escopo": [r"\bdjango\b", r"\bflask\b", r"\bspring\b", r"\bjava\b(?!script)"],
}


def _normalize(text: str) -> str:
    return text.lower()


def _detect_concepts(text: str, concepts: dict[str, list[str]]) -> set[str]:
    """Retorna o conjunto de chaves de `concepts` cujos padrões aparecem em `text`."""
    normalized = _normalize(text)
    found = set()
    for name, patterns in concepts.items():
        if any(re.search(p, normalized) for p in patterns):
            found.add(name)
    return found


# --------------------------------------------------------------------------
# 2. Métricas Base: Precision, Recall, F1
# --------------------------------------------------------------------------

def calculate_precision_recall_f1(response_text: str) -> dict:
    """
    Precision = conceitos relevantes mencionados / total de conceitos mencionados
                (relevantes + ruído)
    Recall    = conceitos esperados mencionados / total de conceitos esperados
    F1        = média harmônica entre Precision e Recall
    """
    expected_found = _detect_concepts(response_text, EXPECTED_CONCEPTS)
    comparison_found = _detect_concepts(response_text, RELEVANT_COMPARISON_CONCEPTS)
    noise_found = _detect_concepts(response_text, NOISE_CONCEPTS)

    relevant_found = expected_found | comparison_found
    total_mentioned = relevant_found | noise_found

    precision = len(relevant_found) / len(total_mentioned) if total_mentioned else 0.0
    recall = len(expected_found) / len(EXPECTED_CONCEPTS) if EXPECTED_CONCEPTS else 0.0

    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1_score": round(f1, 2),
        "expected_concepts_found": sorted(expected_found),
        "expected_concepts_missing": sorted(set(EXPECTED_CONCEPTS) - expected_found),
        "comparison_concepts_found": sorted(comparison_found),
        "noise_concepts_found": sorted(noise_found),
    }


# --------------------------------------------------------------------------
# 3. Métrica Base: Clarity (heurística, sem depender de LLM)
# --------------------------------------------------------------------------

def calculate_clarity(response_text: str) -> float:
    """
    Heurística de clareza baseada em:
      - estrutura (uso de blocos de código, listas, cabeçalhos)
      - comprimento médio das frases (frases muito longas prejudicam clareza)
      - densidade de texto corrido vs. elementos organizados

    Retorna um score entre 0 e 1.
    """
    text = response_text.strip()
    if not text:
        return 0.0

    sentences = [s for s in re.split(r"[.!?\n]+", text) if s.strip()]
    words = text.split()

    avg_sentence_len = len(words) / len(sentences) if sentences else len(words)

    # Frases entre ~8 e ~25 palavras tendem a ser mais claras; penaliza fora dessa faixa
    if 8 <= avg_sentence_len <= 25:
        length_score = 1.0
    else:
        distance = min(abs(avg_sentence_len - 8), abs(avg_sentence_len - 25))
        length_score = max(0.0, 1.0 - distance / 25)

    has_code_block = "```" in text
    has_bullets = bool(re.search(r"(^|\n)\s*[-*•]\s+", text)) or bool(re.search(r"(^|\n)\s*\d+\.\s+", text))
    has_headers = bool(re.search(r"(^|\n)#{1,3}\s+", text))

    structure_score = sum([has_code_block, has_bullets, has_headers]) / 3

    clarity = 0.6 * length_score + 0.4 * structure_score
    return round(min(clarity, 1.0), 2)


# --------------------------------------------------------------------------
# 4. Métricas Derivadas: Helpfulness e Correctness (LLM-as-judge)
# --------------------------------------------------------------------------

class _JudgeScore(BaseModel):
    score: float = Field(..., ge=0, le=1, description="Nota entre 0 e 1")
    justification: str = Field(..., description="Breve justificativa da nota")


def _get_judge_llm(model: str = "gemini-3.8-flash"):
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY não encontrada no ambiente. Defina no .env para usar as "
            "métricas derivadas (Helpfulness/Correctness), ou chame evaluate_response "
            "com use_llm=False."
        )
    return ChatGoogleGenerativeAI(model=model, temperature=0).with_structured_output(_JudgeScore)


def calculate_helpfulness(
    response_text: str,
    user_context: Optional[str] = None,
    recall_hint: Optional[float] = None,
    model: str = "gemini-3.8-flash",
) -> dict:
    """
    Métrica derivada: combina o julgamento de um LLM (o quanto a resposta atende
    ao objetivo/perfil do usuário) com o Recall (base), já que "ser útil" também
    depende de cobrir os conceitos que o usuário realmente precisa.
    """
    judge = _get_judge_llm(model)

    prompt = f"""Avalie o quão ÚTIL (helpfulness) a resposta abaixo é, dado o perfil do usuário.

Perfil do usuário:
{user_context or "Não informado."}

Resposta a ser avaliada:
\"\"\"{response_text}\"\"\"

Dê uma nota de 0 a 1, onde:
- 1.0 = perfeitamente adequada ao nível e preferências do usuário
- 0.0 = completamente inadequada (ex: explica conceitos básicos demais para um sênior)

Considere: nível técnico adequado, uso de comparações pedidas, objetividade."""

    result: _JudgeScore = judge.invoke(prompt)
    llm_score = result.score

    if recall_hint is not None:
        final_score = round(0.6 * llm_score + 0.4 * recall_hint, 2)
    else:
        final_score = round(llm_score, 2)

    return {
        "helpfulness": final_score,
        "llm_score": round(llm_score, 2),
        "justification": result.justification,
    }


def calculate_correctness(response_text: str, model: str = "gemini-3.8-flash") -> dict:
    """
    Métrica derivada: julgamento de um LLM sobre a correção técnica dos conceitos
    e afirmações presentes na resposta (Python 3, FastAPI, Pydantic, ASGI, etc.).
    """
    judge = _get_judge_llm(model)

    prompt = f"""Avalie a CORREÇÃO TÉCNICA (correctness) da resposta abaixo sobre criação
de APIs com Python 3.

Resposta a ser avaliada:
\"\"\"{response_text}\"\"\"

Dê uma nota de 0 a 1, onde:
- 1.0 = todas as afirmações técnicas estão corretas (sintaxe, conceitos, nomes de bibliotecas)
- 0.0 = contém erros técnicos graves ou informações incorretas

Verifique especificamente, se mencionados: sintaxe do FastAPI, uso do Pydantic,
comandos de execução do Uvicorn, e a precisão de qualquer comparação com .NET/Node.js."""

    result: _JudgeScore = judge.invoke(prompt)
    return {"correctness": round(result.score, 2), "justification": result.justification}


# --------------------------------------------------------------------------
# 5. Orquestração: evaluate_response + relatório formatado
# --------------------------------------------------------------------------

def evaluate_response(
    response_text: str,
    user_context: Optional[str] = None,
    use_llm: bool = True,
    model: str = "gemini-3.8-flash",
) -> dict:
    """
    Calcula todas as métricas (base + derivadas) para uma resposta.

    Se use_llm=False, pula Helpfulness/Correctness (úteis para testes rápidos
    sem custo de API), retornando apenas as métricas base.
    """
    base = calculate_precision_recall_f1(response_text)
    base["clarity"] = calculate_clarity(response_text)

    result = {
        "base_metrics": {
            "precision": base["precision"],
            "f1_score": base["f1_score"],
            "clarity": base["clarity"],
        },
        "details": {
            "recall": base["recall"],
            "expected_concepts_found": base["expected_concepts_found"],
            "expected_concepts_missing": base["expected_concepts_missing"],
            "comparison_concepts_found": base["comparison_concepts_found"],
            "noise_concepts_found": base["noise_concepts_found"],
        },
        "derived_metrics": {},
    }

    if use_llm:
        helpfulness = calculate_helpfulness(
            response_text, user_context=user_context, recall_hint=base["recall"], model=model
        )
        correctness = calculate_correctness(response_text, model=model)
        result["derived_metrics"] = {
            "helpfulness": helpfulness["helpfulness"],
            "correctness": correctness["correctness"],
        }
        result["details"]["helpfulness_justification"] = helpfulness["justification"]
        result["details"]["correctness_justification"] = correctness["justification"]

    return result


def print_metrics_report(result: dict, prompt_label: Optional[str] = None) -> None:
    """Imprime o relatório no formato solicitado, com ✓/✗ baseado no THRESHOLD.

    Se `prompt_label` for informado, imprime um cabeçalho:
        ==================================================
        Prompt: {prompt_label}
        ==================================================
    antes das métricas.
    """

    def flag(value: float) -> str:
        return "✓" if value >= THRESHOLD else "✗"

    if prompt_label is not None:
        print("=" * 50)
        print(f"Prompt: {prompt_label}")
        print("=" * 50)

    print()
    if result.get("derived_metrics"):
        print("Métricas Derivadas:")
        h = result["derived_metrics"]["helpfulness"]
        c = result["derived_metrics"]["correctness"]
        print(f"  - Helpfulness: {h:.2f} {flag(h)}")
        print(f"  - Correctness: {c:.2f} {flag(c)}")
        print()

    print("Métricas Base:")
    f1 = result["base_metrics"]["f1_score"]
    clarity = result["base_metrics"]["clarity"]
    precision = result["base_metrics"]["precision"]
    print(f"  - F1-Score: {f1:.2f} {flag(f1)}")
    print(f"  - Clarity: {clarity:.2f} {flag(clarity)}")
    print(f"  - Precision: {precision:.2f} {flag(precision)}")
    print()

    missing = result["details"]["expected_concepts_missing"]
    if missing:
        print(f"  ⚠ Conceitos esperados ausentes: {', '.join(missing)}")
    noise = result["details"]["noise_concepts_found"]
    if noise:
        print(f"  ⚠ Conceitos de ruído detectados: {', '.join(noise)}")
    print()


if __name__ == "__main__":
    # Teste rápido e isolado do módulo (sem precisar do pipeline do LangSmith)
    exemplo_resposta = """
    Para criar uma API em Python 3, use o FastAPI, que é conceitualmente parecido
    com o ASP.NET Core Web API que você já conhece.

    ```python
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI()

    class Item(BaseModel):
        name: str
        price: float

    @app.post("/items")
    def create_item(item: Item):
        return item
    ```

    Aqui, `Item(BaseModel)` do Pydantic equivale a um DTO/Controller do .NET.
    Para rodar, use `uvicorn main:app --reload` (equivalente ao Kestrel).
    Os endpoints seguem o padrão REST com métodos HTTP (GET, POST, PUT, DELETE).
    """

    resultado = evaluate_response(exemplo_resposta, use_llm=False)
    print_metrics_report(resultado)