"""
Module 7 — Generation: grounded RAG ответы с цитатами и стримингом.

Стратегии генерации:
  1. baseline    — LLM + контекст, свободный ответ
  2. citations   — обязательные inline-цитаты [n](msmarco#id)
  3. structured  — Pydantic-карточка (answer, confidence, sources)

Бэкенд: Qdrant rag_production (hybrid search из Module 5).

Запуск:
  python scripts/module_7_generation.py                # benchmark
  python scripts/module_7_generation.py --interactive   # интерактив
"""

import os, sys, time, json, argparse, re
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from common import get_chat_llm
from module_5_retrieval import search_hybrid, get_client

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(BASE, "..", "..", "data"))
COLLECTION = "rag_v2"

BASELINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Ты — RAG-ассистент. Отвечай на основе предоставленного контекста. "
     "Если в контексте нет информации — честно скажи, что не можешь ответить. "
     "Не выдумывай факты. "
     "СТРОГОЕ ПРАВИЛО ЯЗЫКА: отвечай ТОЛЬКО на языке вопроса."),
    ("user", "Контекст:\n{context}\n\nВопрос: {question}"),
])

CITATIONS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a RAG assistant. Answer based ONLY on the provided context. "
     "ALWAYS add citations [n](msmarco#id) after every fact taken from the context. "
     "The number n is the document number in the context block. "
     "If the context lacks the information, honestly say you cannot answer. "
     "Do not invent facts. "
     "STRICT LANGUAGE RULE: write the entire answer - text AND citations - "
     "ONLY in the language of the question. Paraphrase context fragments in the "
     "language of the question, never quote them verbatim."),
    ("user", "Context:\n{context}\n\nQuestion: {question}"),
])


class AnswerCard(BaseModel):
    answer: str = Field(description="краткий ответ на вопрос")
    confidence: str = Field(description="high / medium / low — уверенность в ответе")
    sources: list[str] = Field(description="список цитат вида msmarco#id")


STRUCTURED_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Извлеки из контекста структурированный ответ. "
     "Если факта нет — answer='не могу ответить', confidence='low'. "
     "СТРОГОЕ ПРАВИЛО ЯЗЫКА: отвечай ТОЛЬКО на языке вопроса."),
    ("user", "Контекст:\n{context}\n\nВопрос: {question}\n\n{format_instructions}"),
])

FAITHFULNESS_TPL = ("Ты — judge. Оцени faithfulness ответа (0.0 - 1.0): "
                    "насколько ответ опирается на контекст, а не на внутренние знания модели. "
                    "Верни ТОЛЬКО число от 0.0 до 1.0, ничего больше.\n\n"
                    "Контекст:\n$CONTEXT$\n\nОтвет:\n$ANSWER$")

TEST_QUERIES = [
    {"query": "what teas are good for what",            "qids": ["900736"]},
    {"query": "what was the immediate impact of the success of the manhattan project", "qids": ["1185869"]},
    {"query": "what color is amber urine",              "qids": ["597651"]},
    {"query": "is autoimmune hepatitis a bile acid synthesis disorder", "qids": ["403613"]},
    {"query": "elegxo meaning",                         "qids": ["1183785"]},
    {"query": "how much does an average person make for tutoring", "qids": ["312651"]},
    {"query": "can you use a calculator on the compass test", "qids": ["80385"]},
    {"query": "what does physical medicine do",          "qids": ["645590"]},
    {"query": "weather history in amsterdam",            "qids": ["543163"]},
    {"query": "justice is designed to repair the harm to victim the community and the offender", "qids": ["1185868"]},
]


def reorder_lost_in_middle(docs):
    if len(docs) <= 2:
        return docs
    ranked = sorted(docs, key=lambda x: x[2] if len(x) > 2 else 0, reverse=True)
    return [ranked[0]] + ranked[2:] + [ranked[1]]


def format_context(docs, reorder=True):
    if reorder:
        docs = reorder_lost_in_middle(docs)
    lines = []
    for i, d in enumerate(docs):
        doc_id, text = d[0], d[1]
        lines.append(f"[{i+1}](msmarco#{doc_id})\n{text}")
    return "\n\n".join(lines)


def count_tokens(text):
    return max(1, len(text.encode("utf-8")) // 3)


def generate_baseline(query, context, llm):
    prompt = BASELINE_PROMPT.invoke({"context": context, "question": query})
    return llm.invoke(prompt).content


def generate_citations(query, context, llm, config=None):
    prompt = CITATIONS_PROMPT.invoke({"context": context, "question": query})
    return llm.invoke(prompt, config=config or {}).content


def generate_structured(query, context, llm):
    parser = PydanticOutputParser(pydantic_object=AnswerCard)
    prompt = STRUCTURED_PROMPT.invoke({
        "context": context,
        "question": query,
        "format_instructions": parser.get_format_instructions(),
    })
    try:
        card = parser.parse(llm.invoke(prompt).content)
        return card
    except Exception:
        return AnswerCard(answer="(error parsing)", confidence="low", sources=[])


def extract_citations(text):
    return re.findall(r'\[(\d+)\]\(msmarco#([^)]+)\)', text)


def verify_grounding(answer, retrieved_ids, num_docs):
    cited = extract_citations(answer)
    if not cited:
        return ["no citations"], [], True
    positions = [int(n) for n, _ in cited]
    doc_ids = [d for _, d in cited]
    issues = [n for n in positions if n < 1 or n > num_docs]
    for n, d in cited:
        if str(d) not in retrieved_ids:
            issues.append(f"doc {d} not in retrieved set")
    return issues, doc_ids, bool(issues)


def _parse_faithfulness(response: str) -> float:
    """Парсит число из ответа judge, устойчиво к лишнему тексту."""
    response = response.strip()
    m = re.search(r'([0-9]*\.?[0-9]+)', response)
    if m:
        val = float(m.group(1))
        return max(0.0, min(1.0, val))
    return 0.0


def check_faithfulness(context, answer, llm, config=None):
    prompt = FAITHFULNESS_TPL.replace("$CONTEXT$", context).replace("$ANSWER$", answer)
    try:
        resp = llm.invoke([HumanMessage(content=prompt)], config=config or {}).content
        return _parse_faithfulness(resp)
    except Exception:
        return 0.0


def stream_answer(prompt_template, query, context, llm):
    prompt = prompt_template.invoke({"context": context, "question": query})
    for chunk in llm.stream(prompt):
        yield chunk.content


def interactive():
    print("\n=== Module 7: Generation (interactive) ===")
    print("Commands: /help /quit\n")
    llm = get_chat_llm(temperature=0)

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q == "/quit":
            break
        if q == "/help":
            print("  <query>       — RAG + citations (default)")
            print("  /baseline <q> — без цитат")
            print("  /citations <q> — с цитатами")
            print("  /structured <q> — Pydantic-карточка")
            print("  /quit")
            continue

        if q.startswith("/baseline "):
            query, strategy = q[10:], "baseline"
        elif q.startswith("/citations "):
            query, strategy = q[11:], "citations"
        elif q.startswith("/structured "):
            query, strategy = q[12:], "structured"
        else:
            query, strategy = q, "citations"

        t0 = time.time()
        docs = search_hybrid(query, k=5)
        rt = time.time() - t0
        context = format_context(docs)
        print(f"  retrieved {len(docs)} docs in {rt:.2f}s\n")

        prompt_t = {"baseline": BASELINE_PROMPT, "citations": CITATIONS_PROMPT, "structured": STRUCTURED_PROMPT}[strategy]

        t0 = time.time()
        if strategy == "structured":
            card = generate_structured(query, context, llm)
            print(f"[structured] answer: {card.answer}")
            print(f"[structured] confidence: {card.confidence}")
            print(f"[structured] sources: {card.sources}")
        else:
            print(f"[{strategy}] ", end="", flush=True)
            full = ""
            for chunk in stream_answer(prompt_t, query, context, llm):
                print(chunk, end="", flush=True)
                full += chunk
            print()
        gt = time.time() - t0
        print(f"  generation: {gt:.1f}s")

        if strategy == "citations":
            retrieved_ids = [str(d[0]) for d in docs]
            issues, cited, has_issues = verify_grounding(full, retrieved_ids, len(docs))
            if has_issues:
                print(f"  ⚠️  Citation issues: {issues}")
            else:
                print(f"  ✅ Citations: {cited}")
            faithfulness = check_faithfulness(context, full, llm)
            print(f"  Faithfulness: {faithfulness:.2f}")


def run_benchmark():
    client = get_client()
    info = client.get_collection(COLLECTION)
    print(f"Collection: {COLLECTION} — {info.points_count} points, status={info.status}")

    llm = get_chat_llm(temperature=0)
    strategies = ["baseline", "citations", "structured"]
    results = {s: {"retrieve": [], "gen": [], "total": [], "faith": [], "tok_in": 0, "tok_out": 0} for s in strategies}

    for qi, item in enumerate(TEST_QUERIES):
        query = item["query"]
        print(f"\n[{qi+1}/10] \"{query}\"")

        t0 = time.time()
        docs = search_hybrid(query, k=5)
        rt = time.time() - t0
        context = format_context(docs)
        print(f"  retrieve: {len(docs)} docs, {rt*1000:.0f}ms")

        for s in strategies:
            t0 = time.time()
            if s == "baseline":
                answer = generate_baseline(query, context, llm)
            elif s == "citations":
                answer = generate_citations(query, context, llm)
            else:
                card = generate_structured(query, context, llm)
                answer = card.answer
            gt = time.time() - t0

            faithfulness = check_faithfulness(context, answer, llm)

            results[s]["retrieve"].append(rt)
            results[s]["gen"].append(gt)
            results[s]["total"].append(rt + gt)
            results[s]["faith"].append(faithfulness)
            results[s]["tok_in"] += count_tokens(context + query)
            results[s]["tok_out"] += count_tokens(answer)

            preview = answer[:80].replace("\n", " ")
            print(f"  {s:15s}  gen={gt*1000:.0f}ms  faith={faithfulness:.2f}  \"{preview}\"")

    print("\n" + "=" * 80)
    print(f"{'Strategy':15s}  {'Retrieve':>10s}  {'Generate':>10s}  {'Total':>10s}  {'Faithfulness':>12s}  {'Tokens':>10s}")
    print("-" * 80)
    for s in strategies:
        avg_r = sum(results[s]["retrieve"]) / len(results[s]["retrieve"]) * 1000
        avg_g = sum(results[s]["gen"]) / len(results[s]["gen"]) * 1000
        avg_t = sum(results[s]["total"]) / len(results[s]["total"]) * 1000
        avg_f = sum(results[s]["faith"]) / len(results[s]["faith"])
        toks = results[s]["tok_in"] + results[s]["tok_out"]
        print(f"  {s:15s}  {avg_r:8.0f}ms  {avg_g:8.0f}ms  {avg_t:8.0f}ms  {avg_f:10.2f}     {toks:>8d}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()
    if args.interactive:
        interactive()
    else:
        run_benchmark()
