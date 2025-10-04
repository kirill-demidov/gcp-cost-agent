"""
FastAPI Backend для GCP Cost Agent с графиками
"""
import os
import sys
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import logging
# Prometheus and Instrumentator imports (safe, optional)
from fastapi import status
try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest, Gauge, Counter, Histogram
    PROM_ENABLED = True
except Exception:
    PROM_ENABLED = False
try:
    from fastapi_instrumentator import Instrumentator
    INSTRUMENTATOR_AVAILABLE = True
except Exception:
    INSTRUMENTATOR_AVAILABLE = False
import time
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import google.generativeai as genai

# Загружаем переменные окружения из .env
load_dotenv()

# Добавляем путь к агенту
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Настройка Gemini API
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics setup
if PROM_ENABLED:
    registry = CollectorRegistry()
    app_health_gauge = Gauge("gcp_cost_agent_health", "Overall health status (1=ok,0=fail)", registry=registry)
    toolbox_status_gauge = Gauge("gcp_cost_agent_toolbox_status", "Toolbox connectivity (1=ok,0=fail)", registry=registry)
    genai_config_gauge = Gauge("gcp_cost_agent_genai_config", "Gemini API key present (1=present,0=missing)", registry=registry)
    request_latency_hist = Histogram("gcp_cost_agent_request_latency_seconds", "Request latency seconds", registry=registry, buckets=(0.05,0.1,0.25,0.5,1,2,5))
    request_counter = Counter("gcp_cost_agent_requests_total", "Total requests processed", ["endpoint","method","status_code"], registry=registry)
else:
    registry = None

# Пытаемся импортировать агенты (могут не работать в Cloud Run без toolbox)
try:
    from agents.gcp_cost_agent.agent import root_agent, run_agent_query
    from toolbox_core import ToolboxSyncClient
    AGENTS_AVAILABLE = True
except Exception as e:
    logger.warning(f"Agents not available: {e}")
    root_agent = None
    run_agent_query = None
    ToolboxSyncClient = None
    AGENTS_AVAILABLE = False

# Хранилище истории разговоров (session_id -> список сообщений)
# Каждое сообщение: {"role": "user"/"assistant", "content": "...", "parsed": {...}}
conversation_history: Dict[str, List[Dict[str, Any]]] = {}

# Функция для понимания запроса с помощью LLM
def understand_query_with_llm(question: str, history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Использует Gemini для понимания намерения пользователя с учетом контекста"""
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    if not os.getenv('GOOGLE_API_KEY'):
        return {"intent": "unknown", "month": None, "year": None, "date": None, "date_range": None, "service": None, "analysis_type": None, "top_n": None}

    # Формируем контекст из истории
    context = ""
    if history and len(history) > 0:
        # Берем последние 2 обмена (4 сообщения: 2 user + 2 assistant)
        recent_history = history[-4:]
        context = "\n\nКонтекст предыдущего разговора:\n"
        for msg in recent_history:
            if msg['role'] == 'user':
                context += f"Пользователь: {msg['content']}\n"
            else:
                context += f"Ассистент: {msg.get('answer', '')}\n"
                if 'parsed' in msg:
                    context += f"(определено: {msg['parsed']})\n"

    prompt = f"""Проанализируй запрос пользователя о расходах в GCP и извлеки структурированную информацию.{context}

Текущий запрос: "{question}"

ВАЖНО:
1. Если запрос неполный или содержит местоимения/ссылки ("а по X", "а когда", "для него", "по сервисам"), используй контекст предыдущего разговора чтобы понять полное намерение.
2. Если пользователь говорит "всего", "в сумме", "за все время", "за весь период" - это означает year: "all" и month: null (НЕ БЕРИ месяц из контекста!)
3. Только если пользователь НЕ указывает период явно - можешь взять из контекста.
4. Если в контексте был запрос о динамике/трендах и пользователь говорит "по сервисам" - это означает показать разбивку по сервисам для того же периода.

ВНИМАНИЕ: Если вопрос НЕ СВЯЗАН с GCP, облачными расходами, анализом затрат - ВСЕГДА возвращай "unknown"!

Ответь ТОЛЬКО в формате JSON со следующими полями:
{{
  "intent": "costs|trends|comparison|daily|date_breakdown|date_range|service_daily|service_year|projects|forecast|optimization|benchmark|anomaly|efficiency|unknown",
  "month": "Месяц Год (например 'Сентябрь 2025') или null",
  "year": "YYYY (например '2025') или 'all' для всех данных, или null",
  "date": "YYYY-MM-DD или null",
  "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}} или null",
  "service": "storage|compute|bigquery|kubernetes|cloudrun|functions|sql|artifact|vertex|null",
  "analysis_type": "list|growth|decline|average|median|peak|volatility|stats|seasonal|trend|anomaly|null",
  "top_n": "число для топ-N запросов (например 5 для 'топ-5', 3 для 'топ-3') или null"
}}

Правила:
- intent:
  * "costs" если спрашивают о расходах за месяц ИЛИ топ сервисов (БЕЗ анализа динамики по месяцам!)
  * "trends" для динамики ПО МЕСЯЦАМ, вопросов о росте/снижении, статистик по периоду, или когда спрашивают "в какой месяц" ("в какой месяц был пик", "когда был максимум", "динамика по месяцам")
  * "comparison" для сравнения
  * "daily" для запросов по дням месяца (общая статистика)
  * "date_breakdown" для запроса разбивки по сервисам за конкретный день
  * "date_range" для запроса расходов за диапазон дат ("первые три дня", "с 1 по 5", "1-10 сентября")
  * "service_daily" для запроса расходов по дням для конкретного сервиса ЗА МЕСЯЦ
  * "service_year" для запроса расходов по дням для сервиса за ГОД или ЗА ВСЕ ВРЕМЯ
  * "projects" для запросов о расходах по проектам ("расходы по проектам", "какой проект тратит больше", "топ проектов")
  * "forecast" для прогнозирования расходов ("прогноз", "предсказание", "ожидаемые расходы", "следующий месяц", "прогноз на", "предсказать")
  * "optimization" для анализа оптимизации ("где сэкономить", "неиспользуемые ресурсы", "рекомендации", "оптимизация расходов", "как сэкономить", "экономия", "оптимизировать")
  * "benchmark" для бенчмаркинга ("средние расходы", "норма", "базовая линия", "бенчмарк", "среднее значение", "стандарт")
  * "anomaly" для поиска аномалий ("пики", "самые дорогие месяцы", "аномалии", "выбросы", "необычные", "аномальные")
  * "efficiency" для анализа эффективности ("ROI", "стоимость на пользователя", "оптимизация", "эффективность", "эффективность использования")
  * "unknown" если вопрос НЕ СВЯЗАН с GCP, облачными расходами, или если непонятно что хочет пользователь
- month: ВАЖНО - ставь null если период НЕ УКАЗАН явно! Извлекай только если упоминается конкретный месяц (например "May 2025" -> "May 2025", "июль 2024" -> "Июль 2024", "октябрь" -> "Октябрь 2025"). Если период не указан ("какой самый дорогой", "топ сервисов", "расходы по проектам" БЕЗ месяца) -> month = null
- year: если упоминается только год ("в 2025"), укажи "2025"; если "за все время"/"за весь период", укажи "all"
- date: если упоминается конкретная дата (например "13 сентября", "2025-09-13"), извлеки в формате YYYY-MM-DD
- date_range: если упоминается диапазон дат (например "первые три дня сентября", "с 1 по 5 августа", "1-10.9.2024"), извлеки start и end в формате YYYY-MM-DD. Примеры: "первые три дня сентября 2024" -> {{"start": "2024-09-01", "end": "2024-09-03"}}, "1-10 августа 2024" -> {{"start": "2024-08-01", "end": "2024-08-10"}}
- service: если упоминается конкретный сервис (storage, compute, bigquery, vertex и т.д.), иначе null
- analysis_type:
  * "list" - просто показать список (по умолчанию)
  * "growth" - найти самый быстрый рост ("самый быстрый рост", "когда больше всего выросло")
  * "decline" - найти самое большое снижение ("самое большое падение", "когда упало")
  * "average" - среднее значение ТОЛЬКО ЕСЛИ просят ТОЛЬКО среднее ("средний", "в среднем")
  * "median" - медиана ТОЛЬКО ЕСЛИ просят ТОЛЬКО медиану ("медиана", "медианный")
  * "peak" - максимум ТОЛЬКО ЕСЛИ просят ТОЛЬКО максимум ("пик", "максимум", "самый высокий")
  * "volatility" - волатильность ("волатильность", "стабильность", "разброс")
  * "stats" - ВСЕ статистики вместе ЕСЛИ просят несколько метрик ("макс минимум", "max min avg median", "среднее и медиана", "все статистики", "полная статистика")
  * "seasonal" - сезонность ("сезонность", "паттерны", "циклы")
  * "trend" - тренды ("тренд", "направление", "склонность")
  * "anomaly" - аномалии ("аномалии", "выбросы", "необычные")
- top_n: если упоминается "топ", "top", "первые N", "лучшие N" - извлеки число N, иначе null

Примеры:
"my costs for May 2025" -> {{"intent": "costs", "month": "May 2025", "year": null, "date": null, "service": null, "analysis_type": null, "top_n": null}}
"траты за октябрь за storage" -> {{"intent": "costs", "month": "Октябрь 2025", "year": null, "date": null, "service": "storage", "analysis_type": null, "top_n": null}}
"расходы по дням за сентябрь" -> {{"intent": "daily", "month": "Сентябрь 2025", "year": null, "date": null, "service": null, "analysis_type": null, "top_n": null}}
"в какой день в сентябре был самый высокий кост по storage" -> {{"intent": "service_daily", "month": "Сентябрь 2025", "year": null, "date": null, "service": "storage", "analysis_type": null, "top_n": null}}
"когда был самый быстрый рост расходов" -> {{"intent": "trends", "month": null, "year": null, "date": null, "service": null, "analysis_type": "growth", "top_n": null}}
"покажи динамику" -> {{"intent": "trends", "month": null, "year": null, "date": null, "service": null, "analysis_type": "list", "top_n": null}}
"средние расходы за период" -> {{"intent": "trends", "month": null, "year": null, "date": null, "service": null, "analysis_type": "average", "top_n": null}}
"пиковые расходы" -> {{"intent": "trends", "month": null, "year": null, "date": null, "service": null, "analysis_type": "peak", "top_n": null}}
"в какой месяц был самый дорогой чек" -> {{"intent": "trends", "month": null, "year": null, "date": null, "service": null, "analysis_type": "peak", "top_n": null}}
"когда был максимум расходов" -> {{"intent": "trends", "month": null, "year": null, "date": null, "service": null, "analysis_type": "peak", "top_n": null}}
"покажи max min avg median за 2024" -> {{"intent": "trends", "month": null, "year": "2024", "date": null, "service": null, "analysis_type": "stats", "top_n": null}}
"макс минимум медиана за 2024" -> {{"intent": "trends", "month": null, "year": "2024", "date": null, "service": null, "analysis_type": "stats", "top_n": null}}
"среднее и медиана" -> {{"intent": "trends", "month": null, "year": null, "date": null, "service": null, "analysis_type": "stats", "top_n": null}}
"расходы по проектам за сентябрь" -> {{"intent": "projects", "month": "Сентябрь 2025", "year": null, "date": null, "service": null, "analysis_type": null, "top_n": null}}
"какой самый дорогой сервис" -> {{"intent": "costs", "month": null, "year": null, "date": null, "service": null, "analysis_type": null, "top_n": 1}}
"топ-5 самых дорогих сервисов" -> {{"intent": "costs", "month": null, "year": null, "date": null, "service": null, "analysis_type": null, "top_n": 5}}
"топ-3 проекта по расходам" -> {{"intent": "projects", "month": null, "year": null, "date": null, "service": null, "analysis_type": null, "top_n": 3}}
"расходы по проектам" -> {{"intent": "projects", "month": null, "year": null, "date": null, "service": null, "analysis_type": null, "top_n": null}}
"сколько я всего потратил на vertex AI" -> {{"intent": "costs", "month": null, "year": "all", "date": null, "date_range": null, "service": "vertex", "analysis_type": null, "top_n": null}}
"сколько в сумме на storage" -> {{"intent": "costs", "month": null, "year": "all", "date": null, "date_range": null, "service": "storage", "analysis_type": null, "top_n": null}}
"сколько потрачено за первые три дня сентября 2025" -> {{"intent": "date_range", "month": null, "year": null, "date": null, "date_range": {{"start": "2025-09-01", "end": "2025-09-03"}}, "service": null, "analysis_type": null, "top_n": null}}
"1-10 августа 2024 с разбивкой по сервисам" -> {{"intent": "date_range", "month": null, "year": null, "date": null, "date_range": {{"start": "2024-08-01", "end": "2024-08-10"}}, "service": null, "analysis_type": "list", "top_n": null}}
"прогноз расходов на следующий месяц" -> {{"intent": "forecast", "month": null, "year": null, "date": null, "service": null, "analysis_type": null, "top_n": null}}
"где можно сэкономить?" -> {{"intent": "optimization", "month": null, "year": "all", "date": null, "service": null, "analysis_type": null, "top_n": null}}
"как сэкономить на расходах?" -> {{"intent": "optimization", "month": null, "year": "all", "date": null, "service": null, "analysis_type": null, "top_n": null}}
"анализ оптимизации расходов" -> {{"intent": "optimization", "month": null, "year": "all", "date": null, "service": null, "analysis_type": null, "top_n": null}}
"средние расходы в месяц" -> {{"intent": "benchmark", "month": null, "year": "2025", "date": null, "service": null, "analysis_type": "average", "top_n": null}}
"базовая линия расходов" -> {{"intent": "benchmark", "month": null, "year": "2025", "date": null, "service": null, "analysis_type": "average", "top_n": null}}
"какие месяцы были самыми дорогими?" -> {{"intent": "anomaly", "month": null, "year": "2025", "date": null, "service": null, "analysis_type": "peak", "top_n": 3}}
"найди аномалии в расходах" -> {{"intent": "anomaly", "month": null, "year": "2025", "date": null, "service": null, "analysis_type": "anomaly", "top_n": null}}
"есть ли сезонность в расходах?" -> {{"intent": "trends", "month": null, "year": "2025", "date": null, "service": null, "analysis_type": "seasonal", "top_n": null}}
"покажи неиспользуемые ресурсы" -> {{"intent": "optimization", "month": null, "year": "all", "date": null, "service": null, "analysis_type": "efficiency", "top_n": null}}
"анализ эффективности использования ресурсов" -> {{"intent": "efficiency", "month": null, "year": null, "date": null, "service": null, "analysis_type": null, "top_n": null}}
"как настроить wifi" -> {{"intent": "unknown", "month": null, "year": null, "date": null, "service": null, "analysis_type": null, "top_n": null}}
"какой сегодня день" -> {{"intent": "unknown", "month": null, "year": null, "date": null, "service": null, "analysis_type": null, "top_n": null}}
"хочу купить машину" -> {{"intent": "unknown", "month": null, "year": null, "date": null, "service": null, "analysis_type": null, "top_n": null}}
"а теперь по сервисам" (после запроса о динамике) -> {{"intent": "costs", "month": null, "year": "2025", "date": null, "service": null, "analysis_type": "list", "top_n": null}}
"по сервисам" (после запроса о динамике за 2025) -> {{"intent": "costs", "month": null, "year": "2025", "date": null, "service": null, "analysis_type": "list", "top_n": null}}
"""

    try:
        response = model.generate_content(prompt)
        import json

        # Логируем сырой ответ для отладки
        logger.info(f"Raw LLM response: {response.text}")

        # Извлекаем JSON из ответа (может быть обернут в markdown)
        response_text = response.text.strip()

        # Удаляем markdown code blocks если есть
        if response_text.startswith('```'):
            # Найти JSON между ```
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                response_text = response_text[start:end]

        result = json.loads(response_text)
        logger.info(f"LLM parsed: {result}")
        return result
    except Exception as e:
        logger.error(f"LLM parsing error: {e}")
        return {"intent": "unknown", "month": None, "service": None}

# Функция для форматирования месяца в человекочитаемый вид
def format_month_human(month_str: str) -> str:
    """Преобразует 202507 в 'Июль 2025'"""
    month_names = {
        '01': 'Январь', '02': 'Февраль', '03': 'Март', '04': 'Апрель',
        '05': 'Май', '06': 'Июнь', '07': 'Июль', '08': 'Август',
        '09': 'Сентябрь', '10': 'Октябрь', '11': 'Ноябрь', '12': 'Декабрь'
    }
    if len(month_str) == 6:  # YYYYMM
        year = month_str[:4]
        month = month_str[4:6]
        return f"{month_names.get(month, month)} {year}"
    return month_str

def map_service_name(service_short: str) -> str:
    """Преобразует короткое название сервиса в полное название из BigQuery"""
    service_mapping = {
        'storage': 'Cloud Storage',
        'bigquery': 'BigQuery',
        'compute': 'Compute Engine',
        'kubernetes': 'Kubernetes Engine',
        'cloudrun': 'Cloud Run',
        'functions': 'Cloud Functions',
        'sql': 'Cloud SQL',
        'artifact': 'Artifact Registry',
        'reservation': 'BigQuery Reservation API',
        'vertex': 'Vertex AI',
    }
    return service_mapping.get(service_short.lower(), service_short)

# Функции анализа трендов
def analyze_trends_list(data: List[Dict], start_display: str, end_display: str) -> str:
    """Просто показывает список месяцев с расходами"""
    answer = f"Динамика затрат с {start_display} по {end_display}:\n\n"
    for row in data:
        month_formatted = format_month_human(row['month'])
        cost = row['total_cost']
        answer += f"📅 {month_formatted}: {cost:.2f} {row['currency']}\n"
    return answer

def analyze_trends_growth(data: List[Dict], start_display: str, end_display: str) -> str:
    """Находит месяц с самым быстрым ростом"""
    if len(data) < 2:
        return "Недостаточно данных для анализа роста."

    max_growth = 0
    max_growth_month = None
    max_growth_prev = None

    for i in range(1, len(data)):
        prev_cost = data[i-1]['total_cost']
        curr_cost = data[i]['total_cost']
        growth = curr_cost - prev_cost

        if growth > max_growth:
            max_growth = growth
            max_growth_month = data[i]
            max_growth_prev = data[i-1]

    if max_growth_month:
        month_formatted = format_month_human(max_growth_month['month'])
        prev_formatted = format_month_human(max_growth_prev['month'])
        percent = (max_growth / max_growth_prev['total_cost'] * 100) if max_growth_prev['total_cost'] > 0 else 0

        answer = f"Самый быстрый рост расходов был в **{month_formatted}**:\n\n"
        answer += f"📈 {prev_formatted}: {max_growth_prev['total_cost']:.2f} {max_growth_prev['currency']}\n"
        answer += f"📈 {month_formatted}: {max_growth_month['total_cost']:.2f} {max_growth_month['currency']}\n"
        answer += f"💰 Рост: +{max_growth:.2f} {max_growth_month['currency']} (+{percent:.1f}%)"
        return answer

    return "Роста расходов не обнаружено."

def analyze_trends_decline(data: List[Dict], start_display: str, end_display: str) -> str:
    """Находит месяц с самым большим снижением"""
    if len(data) < 2:
        return "Недостаточно данных для анализа снижения."

    max_decline = 0
    max_decline_month = None
    max_decline_prev = None

    for i in range(1, len(data)):
        prev_cost = data[i-1]['total_cost']
        curr_cost = data[i]['total_cost']
        decline = prev_cost - curr_cost

        if decline > max_decline:
            max_decline = decline
            max_decline_month = data[i]
            max_decline_prev = data[i-1]

    if max_decline_month:
        month_formatted = format_month_human(max_decline_month['month'])
        prev_formatted = format_month_human(max_decline_prev['month'])
        percent = (max_decline / max_decline_prev['total_cost'] * 100) if max_decline_prev['total_cost'] > 0 else 0

        answer = f"Самое большое снижение расходов было в **{month_formatted}**:\n\n"
        answer += f"📉 {prev_formatted}: {max_decline_prev['total_cost']:.2f} {max_decline_prev['currency']}\n"
        answer += f"📉 {month_formatted}: {max_decline_month['total_cost']:.2f} {max_decline_month['currency']}\n"
        answer += f"💰 Снижение: -{max_decline:.2f} {max_decline_month['currency']} (-{percent:.1f}%)"
        return answer

    return "Снижения расходов не обнаружено."

def analyze_trends_average(data: List[Dict], start_display: str, end_display: str) -> str:
    """Вычисляет среднее значение расходов"""
    if not data:
        return "Нет данных для расчета среднего."

    total = sum(row['total_cost'] for row in data)
    avg = total / len(data)
    currency = data[0]['currency']

    answer = f"Статистика расходов с {start_display} по {end_display}:\n\n"
    answer += f"📊 Среднее: {avg:.2f} {currency}\n"
    answer += f"📊 Всего месяцев: {len(data)}\n"
    answer += f"💰 Общая сумма: {total:.2f} {currency}"
    return answer

def analyze_trends_median(data: List[Dict], start_display: str, end_display: str) -> str:
    """Вычисляет медиану расходов"""
    if not data:
        return "Нет данных для расчета медианы."

    costs = sorted([row['total_cost'] for row in data])
    n = len(costs)
    median = costs[n // 2] if n % 2 == 1 else (costs[n // 2 - 1] + costs[n // 2]) / 2
    currency = data[0]['currency']

    answer = f"Статистика расходов с {start_display} по {end_display}:\n\n"
    answer += f"📊 Медиана: {median:.2f} {currency}\n"
    answer += f"📊 Всего месяцев: {len(data)}\n"
    answer += f"📊 Минимум: {min(costs):.2f} {currency}\n"
    answer += f"📊 Максимум: {max(costs):.2f} {currency}"
    return answer

def analyze_trends_peak(data: List[Dict], start_display: str, end_display: str) -> str:
    """Находит пиковое значение расходов"""
    if not data:
        return "Нет данных для поиска пика."

    peak_month = max(data, key=lambda x: x['total_cost'])
    month_formatted = format_month_human(peak_month['month'])

    answer = f"Пиковые расходы с {start_display} по {end_display}:\n\n"
    answer += f"📈 Максимум: {peak_month['total_cost']:.2f} {peak_month['currency']}\n"
    answer += f"📅 Месяц: {month_formatted}"
    return answer

def analyze_trends_volatility(data: List[Dict], start_display: str, end_display: str) -> str:
    """Вычисляет волатильность (стандартное отклонение)"""
    if len(data) < 2:
        return "Недостаточно данных для расчета волатильности."

    costs = [row['total_cost'] for row in data]
    avg = sum(costs) / len(costs)
    variance = sum((x - avg) ** 2 for x in costs) / len(costs)
    std_dev = variance ** 0.5
    currency = data[0]['currency']

    answer = f"Волатильность расходов с {start_display} по {end_display}:\n\n"
    answer += f"📊 Среднее: {avg:.2f} {currency}\n"
    answer += f"📊 Стандартное отклонение: {std_dev:.2f} {currency}\n"
    answer += f"📊 Коэффициент вариации: {(std_dev/avg*100):.1f}%\n\n"

    if std_dev / avg < 0.2:
        answer += "Расходы стабильные (низкая волатильность)"
    elif std_dev / avg < 0.5:
        answer += "Расходы умеренно изменчивые"
    else:
        answer += "Расходы сильно изменчивые (высокая волатильность)"

    return answer

def analyze_trends_stats(data: List[Dict], start_display: str, end_display: str) -> str:
    """Показывает все статистики: макс, мин, среднее, медиану"""
    if not data:
        return "Нет данных для расчета статистик."

    costs = sorted([row['total_cost'] for row in data])
    n = len(costs)
    median = costs[n // 2] if n % 2 == 1 else (costs[n // 2 - 1] + costs[n // 2]) / 2
    avg = sum(costs) / len(costs)
    total = sum(costs)
    currency = data[0]['currency']

    answer = f"Полная статистика расходов с {start_display} по {end_display}:\n\n"
    answer += f"📈 Максимум: {max(costs):.2f} {currency}\n"
    answer += f"📉 Минимум: {min(costs):.2f} {currency}\n"
    answer += f"📊 Среднее: {avg:.2f} {currency}\n"
    answer += f"📊 Медиана: {median:.2f} {currency}\n"
    answer += f"💰 Общая сумма: {total:.2f} {currency}\n"
    answer += f"📅 Всего месяцев: {len(data)}"
    return answer

def analyze_seasonality(data: List[Dict], start_display: str, end_display: str) -> str:
    """Анализ сезонности в расходах"""
    if not data:
        return "Нет данных для анализа сезонности."

    # Сортируем по месяцам
    sorted_data = sorted(data, key=lambda x: x['invoice.month'])
    
    answer = f"📊 **Анализ сезонности расходов с {start_display} по {end_display}**\n\n"
    
    # Показываем динамику по месяцам
    answer += "📅 **Динамика по месяцам:**\n"
    for row in sorted_data:
        month_display = format_month_human(row['invoice.month'])
        answer += f"• {month_display}: {row['total_cost']:.2f} {row['currency']}\n"
    
    # Анализ паттернов
    costs = [row['total_cost'] for row in sorted_data]
    avg_cost = sum(costs) / len(costs)
    
    # Находим месяцы выше и ниже среднего
    above_avg = [row for row in sorted_data if row['total_cost'] > avg_cost]
    below_avg = [row for row in sorted_data if row['total_cost'] < avg_cost]
    
    answer += f"\n📈 **Анализ сезонности:**\n"
    answer += f"• Средние расходы: {avg_cost:.2f} ILS\n"
    answer += f"• Месяцев выше среднего: {len(above_avg)}\n"
    answer += f"• Месяцев ниже среднего: {len(below_avg)}\n"
    
    if above_avg:
        answer += f"\n🔺 **Пиковые месяцы (выше среднего):**\n"
        for row in above_avg:
            month_display = format_month_human(row['invoice.month'])
            percentage = ((row['total_cost'] - avg_cost) / avg_cost) * 100
            answer += f"• {month_display}: {row['total_cost']:.2f} ILS (+{percentage:.1f}%)\n"
    
    if below_avg:
        answer += f"\n🔻 **Низкие месяцы (ниже среднего):**\n"
        for row in below_avg:
            month_display = format_month_human(row['invoice.month'])
            percentage = ((avg_cost - row['total_cost']) / avg_cost) * 100
            answer += f"• {month_display}: {row['total_cost']:.2f} ILS (-{percentage:.1f}%)\n"
    
    # Анализ волатильности
    if len(costs) > 1:
        variance = sum((x - avg_cost) ** 2 for x in costs) / len(costs)
        std_dev = variance ** 0.5
        cv = (std_dev / avg_cost) * 100
        
        if cv < 20:
            seasonality = "стабильные (низкая сезонность)"
        elif cv < 50:
            seasonality = "умеренно сезонные"
        else:
            seasonality = "сильно сезонные"
        
        answer += f"\n📊 **Сезонность:** {seasonality} (коэффициент вариации: {cv:.1f}%)"
    
    return answer

# Диспетчер анализаторов
TREND_ANALYZERS = {
    'list': analyze_trends_list,
    'growth': analyze_trends_growth,
    'decline': analyze_trends_decline,
    'average': analyze_trends_average,
    'median': analyze_trends_median,
    'peak': analyze_trends_peak,
    'volatility': analyze_trends_volatility,
    'stats': analyze_trends_stats,
}

def parse_month_to_yyyymm(month_human: str) -> str:
    """Преобразует 'Июль 2025' или 'July 2025' в '202507'"""
    month_mapping = {
        'январь': '01', 'january': '01', 'jan': '01',
        'февраль': '02', 'february': '02', 'feb': '02',
        'март': '03', 'march': '03', 'mar': '03',
        'апрель': '04', 'april': '04', 'apr': '04',
        'май': '05', 'may': '05',
        'июнь': '06', 'june': '06', 'jun': '06',
        'июль': '07', 'july': '07', 'jul': '07',
        'август': '08', 'august': '08', 'aug': '08',
        'сентябрь': '09', 'september': '09', 'sep': '09',
        'октябрь': '10', 'october': '10', 'oct': '10',
        'ноябрь': '11', 'november': '11', 'nov': '11',
        'декабрь': '12', 'december': '12', 'dec': '12',
    }

    # Если уже в формате YYYYMM
    if month_human and month_human.isdigit() and len(month_human) == 6:
        return month_human

    # Парсим "Месяц Год"
    parts = month_human.lower().split()
    if len(parts) >= 2:
        month_name = parts[0]
        year = parts[-1]
        month_num = month_mapping.get(month_name, '01')
        return f"{year}{month_num}"

    return month_human

app = FastAPI(
    title="GCP Cost Agent API",
    description="API для анализа расходов Google Cloud с поддержкой графиков",
    version="1.0.0"
)

# CORS для React фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production указать конкретный домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus/metrics middleware (after CORS, before static)
@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    if PROM_ENABLED:
        try:
            latency = time.perf_counter() - start
            request_latency_hist.observe(latency)
            request_counter.labels(endpoint=request.url.path, method=request.method, status_code=str(response.status_code)).inc()
        except Exception:
            pass
    return response

# Раздача статики
frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

# Fallback /metrics endpoint if needed
if PROM_ENABLED and not INSTRUMENTATOR_AVAILABLE:
    from fastapi import Response
    @app.get("/metrics")
    async def metrics():
        try:
            # Set static gauges
            if os.getenv('GOOGLE_API_KEY'):
                genai_config_gauge.set(1)
            else:
                genai_config_gauge.set(0)
            # toolbox gauge will be updated in /readiness
            output = generate_latest(registry)
            from fastapi.responses import Response
            return Response(content=output, media_type=CONTENT_TYPE_LATEST)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

# Optional Instrumentator metrics exposure
if INSTRUMENTATOR_AVAILABLE:
    try:
        Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
    except Exception:
        pass


class ChatRequest(BaseModel):
    question: str
    language: Optional[str] = "ru"
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    data: Optional[Dict[str, Any]] = None


class DataRequest(BaseModel):
    tool: str
    parameters: Dict[str, str]


@app.get("/")
async def root():
    """Главная страница - редирект на фронтенд"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")


@app.get("/health")
async def health():
    """Проверка здоровья сервиса (liveness, не блокирует)"""
    try:
        genai_key_present = bool(os.getenv('GOOGLE_API_KEY'))
        status_obj = {
            "status": "healthy",
            "version": os.getenv("APP_VERSION", "1.0.0"),
            "env": os.getenv("APP_ENV", "dev"),
            "genai_key_present": genai_key_present,
            "agents_available": AGENTS_AVAILABLE
        }
        if PROM_ENABLED:
            app_health_gauge.set(1)
            genai_config_gauge.set(1 if genai_key_present else 0)
        return status_obj
    except Exception as e:
        if PROM_ENABLED:
            app_health_gauge.set(0)
        raise HTTPException(status_code=503, detail=str(e))


# Kubernetes probe endpoints
@app.get("/liveness")
async def liveness():
    # Pure process-level check
    if PROM_ENABLED:
        app_health_gauge.set(1)
    return {"status":"alive"}

@app.get("/readiness")
async def readiness():
    """
    Readiness checks external dependencies but uses short timeouts to avoid blocking.
    """
    toolbox_ok = False
    try:
        if ToolboxSyncClient:
            toolbox_url = os.getenv('TOOLBOX_URL', 'http://127.0.0.1:5001')
            toolbox = ToolboxSyncClient(toolbox_url, timeout=1.5) if hasattr(ToolboxSyncClient, "__call__") else ToolboxSyncClient(toolbox_url)
            # Try a lightweight call if available, otherwise just instantiate
            toolbox_ok = True
    except Exception as e:
        logger.warning(f"Readiness toolbox check failed: {e}")
        toolbox_ok = False

    if PROM_ENABLED:
        toolbox_status_gauge.set(1 if toolbox_ok else 0)

    ready = toolbox_ok or not AGENTS_AVAILABLE  # If agents are optional, consider ready if they are absent
    return {
        "ready": ready,
        "toolbox_connected": toolbox_ok,
        "agents_available": AGENTS_AVAILABLE
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Отправить вопрос агенту и получить ответ через прямой вызов инструментов
    """
    try:
        logger.info(f"Received question: {request.question}")
        if len(request.question) > 4000:
            return ChatResponse(answer="Вопрос слишком длинный. Пожалуйста, сократите формулировку до 4000 символов.", data=None)

        # Получаем или создаем session_id
        session_id = request.session_id or "default"

        # Получаем историю для этой сессии
        history = conversation_history.get(session_id, [])

        # Проверяем, связан ли запрос с GCP/облачными расходами
        question_lower = request.question.lower()
        gcp_keywords = ['gcp', 'google cloud', 'облако', 'расход', 'затрат', 'стоимост', 'биллинг', 'инвойс', 'счет', 'трат', 'потрач', 'потребил', 'использовал', 'cloud', 'compute', 'storage', 'bigquery', 'vertex', 'artifact', 'run', 'functions', 'kubernetes', 'sql', 'logging', 'monitoring', 'analytics', 'машинное обучение', 'база данных', 'хранение', 'вычисления', 'сервис', 'проект', 'ресурс', 'оптимизац', 'экономи', 'сэконом', 'анализ', 'динамик', 'тренд', 'сравн', 'месяц', 'год', 'день', 'период', 'время', 'дата', 'график', 'диаграмм', 'статистик', 'метрик', 'показател']

        # Если запрос не связан с GCP - сразу возвращаем сообщение
        if not any(keyword in question_lower for keyword in gcp_keywords):
            answer = (
                "❓ **Запрос не распознан**\n\n"
                "Я специализируюсь на анализе расходов Google Cloud Platform (GCP). "
                "Попробуйте переформулировать ваш запрос, например:\n\n"
                "• 'Сколько я потратил в этом месяце?'\n"
                "• 'Покажи расходы по сервисам за август'\n"
                "• 'Какие были затраты на BigQuery?'\n"
                "• 'Сравни расходы июля и августа'\n\n"
                "Или используйте вкладки для просмотра графиков!"
            )

            # Ограничиваем историю последними 10 сообщениями (5 пар вопрос-ответ)
            if len(conversation_history[session_id]) > 10:
                conversation_history[session_id] = conversation_history[session_id][-10:]

            return ChatResponse(answer=answer, data=None)

        # Остальная логика будет добавлена позже
        return ChatResponse(answer="Функция в разработке", data=None)

    except Exception as e:
        logger.error(f"Error in chat: {e}")
        return ChatResponse(answer=f"Ошибка: {str(e)}", data=None)


# Simple version endpoint
@app.get("/version")
async def version():
    return {
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "build": os.getenv("GIT_SHA", "unknown"),
        "env": os.getenv("APP_ENV", "dev")
    }


@app.post("/data/trends")
async def get_cost_trends(start_month: str, end_month: str):
    """
    Получить данные о трендах затрат за период

    Args:
        start_month: Начальный месяц в формате YYYYMM
        end_month: Конечный месяц в формате YYYYMM

    Returns:
        JSON с данными для построения графика
    """
    try:
        toolbox_url = os.getenv('TOOLBOX_URL', 'http://127.0.0.1:5001')

        # Вызываем инструмент напрямую через HTTP
        import requests
        response = requests.post(
            f'{toolbox_url}/api/tool/get_cost_trends/invoke',
            json={'start_month': start_month, 'end_month': end_month}
        )
        response.raise_for_status()
        result = response.json()

        # Парсим результат - Toolbox возвращает {"result": "json_string"}
        import json
        if isinstance(result, dict) and 'result' in result:
            data = json.loads(result['result'])
        else:
            data = result

        return {
            "status": "success",
            "data": data,
            "raw_data": data
        }

    except Exception as e:
        logger.error(f"Error getting cost trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/service-comparison")
async def get_service_comparison(month1: str, month2: str):
    """
    Сравнить затраты по сервисам между двумя месяцами

    Args:
        month1: Первый месяц в формате YYYYMM
        month2: Второй месяц в формате YYYYMM

    Returns:
        JSON с данными для построения графика
    """
    try:
        toolbox_url = os.getenv('TOOLBOX_URL', 'http://127.0.0.1:5001')

        # Вызываем инструмент напрямую через HTTP
        import requests
        response = requests.post(
            f'{toolbox_url}/api/tool/get_service_comparison/invoke',
            json={'month1': month1, 'month2': month2}
        )
        response.raise_for_status()
        result = response.json()

        import json
        if isinstance(result, dict) and 'result' in result:
            data = json.loads(result['result'])
        else:
            data = result

        return {
            "status": "success",
            "data": data,
            "raw_data": data
        }

    except Exception as e:
        logger.error(f"Error getting service comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/monthly-cost")
def get_monthly_cost(month: str):
    """
    Получить данные о затратах за месяц для построения графика

    Args:
        month: Месяц в формате YYYYMM

    Returns:
        JSON с данными для построения графика
    """
    try:
        toolbox_url = os.getenv('TOOLBOX_URL', 'http://127.0.0.1:5001')

        response = requests.post(
            f'{toolbox_url}/api/tool/get_monthly_costs/invoke',
            json={'invoice_month': month}
        )
        response.raise_for_status()
        result = response.json()

        import json
        if isinstance(result, dict) and 'result' in result:
            data = json.loads(result['result'])
        else:
            data = result

        return {
            "status": "success", 
            "data": data,
            "raw_data": data
        }

    except Exception as e:
        logger.error(f"Error getting monthly cost: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Проверка здоровья сервиса"""
    try:
        # Проверяем, что у нас есть API ключ
        if not os.getenv('GOOGLE_API_KEY'):
            return {"status": "unhealthy", "error": "GOOGLE_API_KEY not set"}
        
        # Проверяем Toolbox (но не блокируемся на этом)
        try:
            toolbox_url = os.getenv('TOOLBOX_URL', 'http://127.0.0.1:5001')
            response = requests.get(f'{toolbox_url}/health', timeout=5)
            toolbox_status = "connected" if response.status_code == 200 else "disconnected"
        except:
            toolbox_status = "disconnected"
        
        return {
            "status": "healthy",
            "toolbox_status": toolbox_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI application...")
    print(f"App: {app}")
    print(f"App title: {app.title}")
    port = int(os.getenv("PORT", 8080))
    print(f"Port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
