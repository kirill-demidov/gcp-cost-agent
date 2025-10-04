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
# Simple version endpoint
@app.get("/version")
async def version():
    return {
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "build": os.getenv("GIT_SHA", "unknown"),
        "env": os.getenv("APP_ENV", "dev")
    }

        # Получаем или создаем session_id
        session_id = request.session_id or "default"

        # Получаем историю для этой сессии
        history = conversation_history.get(session_id, [])

        # Проверяем, связан ли запрос с GCP/облачными расходами
        question_lower = request.question.lower()
        gcp_keywords = ['gcp', 'google cloud', 'облако', 'расход', 'затрат', 'стоимост', 'биллинг', 'инвойс', 'счет', 'трат', 'потрач', 'потребил', 'использовал', 'cloud', 'compute', 'storage', 'bigquery', 'vertex', 'artifact', 'run', 'functions', 'kubernetes', 'sql', 'logging', 'monitoring', 'analytics', 'машинное обучение', 'база данных', 'хранение', 'вычисления', 'сервис', 'проект', 'ресурс', 'оптимизац', 'экономи', 'сэконом', 'анализ', 'динамик', 'тренд', 'сравн', 'месяц', 'год', 'день', 'период', 'время', 'дата', 'график', 'диаграмм', 'статистик', 'метрик', 'показател']
        
        # Если в запросе нет ключевых слов, связанных с GCP/расходами
        if not any(keyword in question_lower for keyword in gcp_keywords):
            # Проверяем, не является ли это общим приветствием или вопросом о GCP
            general_greetings = ['привет', 'hello', 'hi', 'как дела', 'как ты', 'что ты умеешь', 'помощь', 'help', 'что такое gcp', 'что такое google cloud', 'расскажи о gcp', 'что ты можешь', 'функции', 'возможности']
            if any(greeting in question_lower for greeting in general_greetings):
                # Это общий вопрос - передаем LLM
                pass
            else:
                # Не связано с GCP - сразу возвращаем unknown
                parsed = {"intent": "unknown", "month": None, "year": None, "date": None, "date_range": None, "service": None, "analysis_type": None, "top_n": None}
        else:
            # Используем LLM для понимания запроса с учетом контекста
            parsed = understand_query_with_llm(request.question, history)

        # Определяем какой инструмент вызвать
        toolbox_url = os.getenv('TOOLBOX_URL', 'http://127.0.0.1:5001')
        import requests
        import json

        # Преобразуем человекочитаемый месяц в YYYYMM для BigQuery
        if parsed['month']:
            current_month = parse_month_to_yyyymm(parsed['month'])
            month_display = parsed['month']  # Сохраняем для отображения
        else:
            # Если месяц не указан, используется логика "за все время" в обработчиках
            current_month = None
            month_display = None

        question_lower = request.question.lower()

        try:
            # Обрабатываем запрос на основе намерения из LLM
            if parsed['intent'] == 'trends':
                # Если указан конкретный месяц и нет сервиса - показываем разбивку по сервисам за этот месяц
                month_param = parsed.get('month')
                service_param = parsed.get('service')

                if month_param and not service_param:
                    # "тренды в мае по всем сервисам" -> разбивка по сервисам за май
                    target_month = parse_month_to_yyyymm(month_param)
                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_cost_by_service/invoke',
                        json={'invoice_month': target_month}
                    )
                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    # Формируем ответ со списком сервисов
                    month_display = format_month_human(target_month)
                    answer = f"Расходы по сервисам за {month_display}:\n\n"
                    for row in data[:15]:  # Топ-15
                        service = row['description'][:50]
                        cost = row['final_cost']
                        currency = row['currency']
                        answer += f"• {service}: {cost:.2f} {currency}\n"
                else:
                    # Обычный trends анализ (динамика по месяцам)
                    year_param = parsed.get('year')
                    if year_param == 'all':
                        # Запрос за весь доступный период (2024-2025)
                        start_month_yyyymm = '202401'
                        end_month_yyyymm = '202512'
                    elif year_param:
                        # Запрос за конкретный год
                        start_month_yyyymm = f"{year_param}01"
                        end_month_yyyymm = f"{year_param}12"
                    else:
                        # По умолчанию - текущий год (2025)
                        start_month_yyyymm = '202501'
                        end_month_yyyymm = '202509'

                    # Проверяем, указан ли сервис для фильтрации
                    if service_param:
                        # Используем get_cost_trends_by_service с маппингом имени
                        service_name = map_service_name(service_param)
                        response = requests.post(
                            f'{toolbox_url}/api/tool/get_cost_trends_by_service/invoke',
                            json={
                                'start_month': start_month_yyyymm,
                                'end_month': end_month_yyyymm,
                                'service_name': service_name
                            }
                        )
                    else:
                        # Используем обычный get_cost_trends
                        response = requests.post(
                            f'{toolbox_url}/api/tool/get_cost_trends/invoke',
                            json={'start_month': start_month_yyyymm, 'end_month': end_month_yyyymm}
                        )

                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    # Проверка на пустые данные
                    if not data or len(data) == 0:
                        service_name = map_service_name(service_param) if service_param else None
                        if service_name:
                            answer = f"Нет данных по сервису {service_name} за указанный период."
                        else:
                            answer = "Нет данных за указанный период."
                    else:
                        # Определяем тип анализа
                        start_display = format_month_human(start_month_yyyymm)
                        end_display = format_month_human(end_month_yyyymm)
                        analysis_type = parsed.get('analysis_type') or 'list'

                        # Используем соответствующий анализатор
                        if analysis_type == 'seasonal':
                            # Анализ сезонности
                            answer = analyze_seasonality(data, start_display, end_display)
                        elif analysis_type in TREND_ANALYZERS:
                            answer = TREND_ANALYZERS[analysis_type](data, start_display, end_display)
                        else:
                            # Дефолтный вывод - список
                            answer = analyze_trends_list(data, start_display, end_display)

            elif parsed['intent'] == 'comparison':
                # Вызываем get_service_comparison
                # Берем месяцы из контекста и текущего запроса
                if history and len(history) >= 2:
                    # Извлекаем месяц из предыдущего запроса
                    last_parsed = history[-1].get('parsed', {})
                    if last_parsed and last_parsed.get('month'):
                        month1_yyyymm = parse_month_to_yyyymm(last_parsed['month'])
                    else:
                        month1_yyyymm = '202507'  # июль по умолчанию
                else:
                    month1_yyyymm = '202507'  # июль по умолчанию
                
                # Месяц из текущего запроса
                if parsed.get('month'):
                    month2_yyyymm = parse_month_to_yyyymm(parsed['month'])
                else:
                    month2_yyyymm = '202506'  # июнь по умолчанию
                response = requests.post(
                    f'{toolbox_url}/api/tool/get_service_comparison/invoke',
                    json={'month1': month1_yyyymm, 'month2': month2_yyyymm}
                )
                response.raise_for_status()
                result = response.json()
                data = json.loads(result['result']) if 'result' in result else result

                # Топ-5 по изменению
                top_changes = sorted(data, key=lambda x: abs(x['cost_difference']), reverse=True)[:5]
                month1_display = format_month_human(month1_yyyymm)
                month2_display = format_month_human(month2_yyyymm)
                answer = f"Топ-5 сервисов по изменению затрат ({month1_display} vs {month2_display}):\n\n"
                for row in top_changes:
                    service = row['service_name'][:40]
                    diff = row['cost_difference']
                    sign = "📈" if diff > 0 else "📉"
                    answer += f"{sign} {service}: {diff:+.2f} ILS\n"

            elif parsed['intent'] == 'daily':
                # Запросы о расходах по дням
                if not parsed['month']:
                    answer = (
                        "За какой месяц вас интересуют расходы по дням?\n\n"
                        "Например:\n"
                        "• 'Расходы по дням за Сентябрь 2025'\n"
                        "• 'Daily costs for May 2025'"
                    )
                else:
                    # Вызываем get_daily_costs
                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_daily_costs/invoke',
                        json={'invoice_month': current_month}
                    )
                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    if not data:
                        answer = f"Данные по дням за {month_display} не найдены."
                    else:
                        # Находим день с максимальными затратами
                        max_day = max(data, key=lambda x: x['daily_cost'])
                        total_cost = sum(row['daily_cost'] for row in data)

                        answer = f"Расходы по дням за {month_display}:\n\n"
                        answer += f"📊 Всего дней с расходами: {len(data)}\n"
                        answer += f"💰 Общая сумма: {total_cost:.2f} {data[0]['currency']}\n"
                        answer += f"📈 Максимум: {max_day['daily_cost']:.2f} {max_day['currency']} ({max_day['date']})\n\n"

                        # Показываем последние 7 дней
                        answer += "Последние дни:\n"
                        for row in data[-7:]:
                            answer += f"• {row['date']}: {row['daily_cost']:.2f} {row['currency']}\n"

            elif parsed['intent'] == 'date_breakdown':
                # Разбивка расходов по сервисам за конкретную дату
                if not parsed.get('date'):
                    answer = (
                        "Укажите конкретную дату для анализа.\n\n"
                        "Например:\n"
                        "• 'Расходы за 13 сентября 2025'\n"
                        "• 'Разбивка по сервисам за 2025-09-13'"
                    )
                else:
                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_costs_for_specific_date/invoke',
                        json={'date': parsed['date']}
                    )
                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    if not data:
                        answer = f"Данные за {parsed['date']} не найдены."
                    else:
                        total_cost = sum(row['cost'] for row in data)
                        answer = f"Расходы за {parsed['date']}:\n\n"

                        for i, row in enumerate(data[:10], 1):  # Топ-10 сервисов
                            service = row['description'][:40]
                            cost = row['cost']
                            answer += f"{i}. {service}: {cost:.2f} {row['currency']}\n"

                        answer += f"\n💰 Общая сумма за день: {total_cost:.2f} {data[0]['currency']}"

            elif parsed['intent'] == 'service_daily':
                # Расходы по дням для конкретного сервиса
                if not parsed.get('service'):
                    answer = (
                        "Укажите сервис для анализа.\n\n"
                        "Например:\n"
                        "• 'Расходы по дням для Cloud Storage'\n"
                        "• 'В какой день больше всего потратили на BigQuery'"
                    )
                elif not parsed['month']:
                    answer = (
                        f"За какой месяц вас интересуют расходы по {parsed['service']}?\n\n"
                        "Например:\n"
                        "• 'За Сентябрь 2025'\n"
                        "• 'За Май 2025'"
                    )
                else:
                    # Преобразуем короткое название сервиса в полное
                    service_full_name = map_service_name(parsed['service'])

                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_daily_costs_for_service/invoke',
                        json={'invoice_month': current_month, 'service_name': service_full_name}
                    )
                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    if not data:
                        answer = f"Данные по {service_full_name} за {month_display} не найдены."
                    else:
                        # Находим день с максимальными затратами
                        max_day = max(data, key=lambda x: x['daily_cost'])
                        total_cost = sum(row['daily_cost'] for row in data)

                        answer = f"Расходы по {service_full_name} за {month_display}:\n\n"
                        answer += f"📊 Всего дней с расходами: {len(data)}\n"
                        answer += f"💰 Общая сумма: {total_cost:.2f} {data[0]['currency']}\n"
                        answer += f"📈 Максимум: {max_day['daily_cost']:.2f} {max_day['currency']} ({max_day['date']})\n\n"

                        # Показываем топ-5 дней
                        answer += "Топ-5 дней по расходам:\n"
                        top_days = sorted(data, key=lambda x: x['daily_cost'], reverse=True)[:5]
                        for row in top_days:
                            answer += f"• {row['date']}: {row['daily_cost']:.2f} {row['currency']}\n"

            elif parsed['intent'] == 'service_year':
                # Расходы по дням для конкретного сервиса за год или за все время
                if not parsed.get('service'):
                    answer = (
                        "Укажите сервис для анализа.\n\n"
                        "Например:\n"
                        "• 'Расходы по Cloud Storage за 2025'\n"
                        "• 'Когда больше всего потратили на BigQuery за все время'"
                    )
                else:
                    # Преобразуем короткое название сервиса в полное
                    service_full_name = map_service_name(parsed['service'])
                    year_param = parsed.get('year', '')

                    # Если year = null, используем текущий год
                    if not year_param:
                        year_param = '2025'

                    # Если year = 'all', передаем пустую строку в BigQuery
                    year_for_query = '' if year_param == 'all' else year_param

                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_daily_costs_for_service_year/invoke',
                        json={'year': year_for_query, 'service_name': service_full_name}
                    )
                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    if not data:
                        period_str = "за все время" if year_param == 'all' else f"за {year_param} год"
                        answer = f"Данные по {service_full_name} {period_str} не найдены."
                    else:
                        # Находим день с максимальными затратами (первый в списке, т.к. отсортировано по убыванию)
                        max_day = data[0]
                        total_cost = sum(row['daily_cost'] for row in data)

                        period_str = "за все время" if year_param == 'all' else f"в {year_param} году"
                        answer = f"Расходы по {service_full_name} {period_str}:\n\n"
                        answer += f"📊 Всего дней с расходами: {len(data)}\n"
                        answer += f"💰 Общая сумма: {total_cost:.2f} {data[0]['currency']}\n"
                        answer += f"📈 Максимум: {max_day['daily_cost']:.2f} {max_day['currency']} ({max_day['date']})\n\n"

                        # Показываем топ-10 дней
                        answer += "Топ-10 дней по расходам:\n"
                        for row in data[:10]:
                            answer += f"• {row['date']}: {row['daily_cost']:.2f} {row['currency']}\n"

            elif parsed['intent'] == 'costs':
                # Если месяц не указан - проверяем контекст
                if not parsed['month']:
                    # Проверяем, не спрашивает ли пользователь о текущем месяце
                    question_lower = request.question.lower()
                    current_month_indicators = ['в этом месяце', 'в текущем месяце', 'за этот месяц', 'за текущий месяц', 'this month', 'current month']
                    
                    if any(indicator in question_lower for indicator in current_month_indicators):
                        # Запрос за текущий месяц
                        from datetime import datetime
                        current_month = datetime.now().replace(day=1)
                        month_str = current_month.strftime("%Y%m")
                        month_display = format_month_human(month_str)
                        
                        response = requests.post(
                            f'{toolbox_url}/api/tool/get_monthly_cost_summary/invoke',
                            json={"month": month_str}
                        )
                        response.raise_for_status()
                        result = response.json()
                        data = json.loads(result['result']) if 'result' in result else result
                        
                        if data:
                            total_cost = data['total_cost']
                            currency = data['currency']
                            answer = f"💰 **Общие затраты за {month_display}:** {total_cost:.2f} {currency}"
                        else:
                            answer = f"❌ Не удалось получить данные за {month_display}"
                    else:
                        # Запрос за все время
                        response = requests.post(
                            f'{toolbox_url}/api/tool/get_cost_by_service_all_time/invoke',
                            json={}
                        )
                        response.raise_for_status()
                        result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    if not data:
                        answer = "Данные не найдены."
                    else:
                        # Проверяем, есть ли фильтр по сервису
                        service_param = parsed.get('service')
                        if service_param:
                            # Фильтруем по конкретному сервису
                            service_name = map_service_name(service_param)
                            filtered_services = [
                                row for row in data
                                if service_name.lower() in row['description'].lower()
                            ]

                            if filtered_services:
                                total = sum(row['final_cost'] for row in filtered_services)
                                answer = f"Затраты на {service_name} за все время:\n\n"
                                for row in filtered_services[:5]:
                                    answer += f"• {row['description']}: {row['final_cost']:.2f} {row['currency']}\n"
                                if len(filtered_services) > 0:
                                    answer += f"\n💰 Итого: {total:.2f} {filtered_services[0]['currency']}"
                            else:
                                answer = f"Данные по сервису {service_name} не найдены."
                        else:
                            # Применяем top_n
                            top_n = parsed.get('top_n', 5)
                            top_n = int(top_n) if top_n else 5
                            top_services = data[:top_n]
                            total_cost = sum(row['final_cost'] for row in data)

                            answer = f"Топ-{top_n} сервисов за все время:\n\n"
                            for i, row in enumerate(top_services, 1):
                                answer += f"{i}. {row['description']}: {row['final_cost']:.2f} {row['currency']}\n"
                            answer += f"\n💰 Общие затраты: {total_cost:.2f} {data[0]['currency']}"
                else:
                    # Общий запрос о расходах за месяц - показываем топ-5 сервисов
                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_cost_by_service/invoke',
                        json={'invoice_month': current_month}
                    )
                    response.raise_for_status()
                    result = response.json()

                    # Парсим результат с обработкой ошибок
                    if 'result' in result:
                        result_str = result['result']
                        if result_str.startswith('[') or result_str.startswith('{'):
                            data = json.loads(result_str)
                        else:
                            data = []
                    else:
                        data = result

                    if not data:
                        answer = f"Данные за {month_display} не найдены. Возможно, этот месяц еще не наступил или данные еще не загружены."
                    else:
                        # Проверяем, есть ли фильтр по сервису из LLM
                        service_mapping = {
                            'storage': 'Storage',
                            'compute': 'Compute Engine',
                            'bigquery': 'BigQuery',
                            'kubernetes': 'Kubernetes',
                            'cloudrun': 'Cloud Run',
                            'functions': 'Cloud Functions',
                            'sql': 'Cloud SQL',
                        }

                        if parsed['service'] and parsed['service'] in service_mapping:
                            # Фильтруем по конкретному сервису
                            service_part = service_mapping[parsed['service']]
                            filtered_services = [
                                row for row in data
                                if service_part.lower() in row['description'].lower()
                            ]
                        else:
                            filtered_services = None

                        if filtered_services:
                            # Показываем конкретный сервис
                            total = sum(row['final_cost'] for row in filtered_services)
                            answer = f"Затраты на {parsed['service'].upper()} за {month_display}:\n\n"
                            for row in filtered_services[:3]:
                                answer += f"• {row['description']}: {row['final_cost']:.2f} {row['currency']}\n"
                            if len(filtered_services) > 0:
                                answer += f"\n💰 Итого: {total:.2f} {filtered_services[0]['currency']}"
                        else:
                            # Топ-N всех сервисов (учитываем top_n из LLM)
                            top_n = parsed.get('top_n', 5)
                            top_n = int(top_n) if top_n else 5
                            top_services = data[:top_n]
                            total_cost = sum(row['final_cost'] for row in data)
                            answer = f"Топ-{top_n} сервисов за {month_display}:\n\n"
                            for i, row in enumerate(top_services, 1):
                                answer += f"{i}. {row['description']}: {row['final_cost']:.2f} {row['currency']}\n"
                            answer += f"\n💰 Общие затраты: {total_cost:.2f} {data[0]['currency']}"

            elif parsed['intent'] == 'projects':
                # Расходы по проектам
                if not parsed['month']:
                    # Запрос за все время
                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_cost_by_project_all_time/invoke',
                        json={}
                    )
                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    if not data:
                        answer = "Данные по проектам не найдены."
                    else:
                        top_n = parsed.get('top_n')
                        if top_n:
                            data = data[:int(top_n)]
                            title = f"Топ-{top_n} проектов по расходам за все время:"
                        else:
                            title = "Расходы по проектам за все время:"

                        total_cost = sum(row['final_cost'] for row in data)
                        answer = f"{title}\n\n"
                        for i, row in enumerate(data, 1):
                            project_name = row['name'] or row['id']
                            answer += f"{i}. {project_name}: {row['final_cost']:.2f} {row['currency']}\n"
                        answer += f"\n💰 Общая сумма: {total_cost:.2f} {data[0]['currency']}"
                else:
                    # Запрос за конкретный месяц
                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_cost_by_project/invoke',
                        json={'invoice_month': current_month}
                    )
                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    if not data:
                        answer = f"Данные по проектам за {month_display} не найдены."
                    else:
                        # Применяем top_n если есть
                        top_n = parsed.get('top_n')
                        if top_n:
                            data = data[:int(top_n)]
                            title = f"Топ-{top_n} проектов по расходам за {month_display}:"
                        else:
                            title = f"Расходы по проектам за {month_display}:"

                        total_cost = sum(row['final_cost'] for row in data)
                        answer = f"{title}\n\n"
                        for i, row in enumerate(data, 1):
                            project_name = row['name'] or row['id']
                            answer += f"{i}. {project_name}: {row['final_cost']:.2f} {row['currency']}\n"
                        answer += f"\n💰 Общая сумма: {total_cost:.2f} {data[0]['currency']}"

            elif parsed['intent'] == 'date_range':
                # Обработка запросов за диапазон дат
                date_range = parsed.get('date_range')
                if not date_range or not date_range.get('start') or not date_range.get('end'):
                    answer = "Укажите диапазон дат в формате 'с 1 по 5 сентября' или 'первые три дня августа'."
                else:
                    start_date = date_range['start']
                    end_date = date_range['end']

                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_costs_for_date_range/invoke',
                        json={'start_date': start_date, 'end_date': end_date}
                    )
                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result

                    if not data or len(data) == 0:
                        answer = f"Нет данных за период с {start_date} по {end_date}."
                    else:
                        # Группируем по датам и сервисам
                        from datetime import datetime
                        start_display = datetime.strptime(start_date, '%Y-%m-%d').strftime('%d.%m.%Y')
                        end_display = datetime.strptime(end_date, '%Y-%m-%d').strftime('%d.%m.%Y')

                        # Считаем общую сумму
                        total_cost = sum(row['daily_cost'] for row in data)
                        currency = data[0]['currency']

                        # Группируем по датам
                        dates = {}
                        for row in data:
                            date = row['date']
                            if date not in dates:
                                dates[date] = []
                            dates[date].append(row)

                        answer = f"Расходы с {start_display} по {end_display}:\n\n"

                        for date in sorted(dates.keys()):
                            date_display = datetime.strptime(date, '%Y-%m-%d').strftime('%d.%m.%Y')
                            day_total = sum(row['daily_cost'] for row in dates[date])
                            answer += f"📅 {date_display}: {day_total:.2f} {currency}\n"

                            # Топ-3 сервиса за день
                            top_services = sorted(dates[date], key=lambda x: x['daily_cost'], reverse=True)[:3]
                            for service_row in top_services:
                                if service_row['daily_cost'] > 0:
                                    answer += f"   • {service_row['description']}: {service_row['daily_cost']:.2f} {currency}\n"
                            answer += "\n"

                        answer += f"💰 Итого за период: {total_cost:.2f} {currency}"

            elif parsed['intent'] == 'forecast':
                # Прогнозирование расходов
                answer = (
                    "📊 **Прогноз расходов**\n\n"
                    "Для точного прогнозирования нужны исторические данные за несколько месяцев. "
                    "Основываясь на текущих трендах, могу предложить:\n\n"
                    "• **Анализ трендов** за последние месяцы\n"
                    "• **Средние расходы** и их динамику\n"
                    "• **Сезонные паттерны** в использовании\n\n"
                    "Попробуйте спросить:\n"
                    "• 'Покажи динамику за 2025 год'\n"
                    "• 'Средние расходы в месяц'\n"
                    "• 'Есть ли сезонность в расходах?'"
                )
                
            elif parsed['intent'] == 'optimization':
                # Анализ оптимизации
                if not parsed.get('year') or parsed.get('year') == 'all':
                    # Запрос за все время для анализа оптимизации
                    response = requests.post(
                        f'{toolbox_url}/api/tool/get_cost_by_service_all_time/invoke',
                        json={}
                    )
                    response.raise_for_status()
                    result = response.json()
                    data = json.loads(result['result']) if 'result' in result else result
                    
                    if data:
                        # Анализируем данные для рекомендаций
                        total_cost = sum(row['final_cost'] for row in data)
                        top_services = sorted(data, key=lambda x: x['final_cost'], reverse=True)[:5]
                        
                        answer = "🔍 **Анализ оптимизации расходов GCP**\n\n"
                        answer += f"💰 **Общие расходы за все время:** {total_cost:.2f} {data[0]['currency']}\n\n"
                        answer += "🎯 **Топ-5 сервисов по расходам:**\n"
                        for i, row in enumerate(top_services, 1):
                            percentage = (row['final_cost'] / total_cost) * 100
                            answer += f"{i}. **{row['description']}**: {row['final_cost']:.2f} {row['currency']} ({percentage:.1f}%)\n"
                        
                        answer += "\n💡 **Рекомендации по оптимизации:**\n"
                        answer += "• Проанализируйте топ-сервисы на предмет неиспользуемых ресурсов\n"
                        answer += "• Рассмотрите возможность перехода на более дешевые альтернативы\n"
                        answer += "• Настройте автоматическое масштабирование для переменных нагрузок\n"
                        answer += "• Используйте коммитменты для стабильных рабочих нагрузок\n"
                    else:
                        answer = "Нет данных для анализа оптимизации."
                else:
                    answer = "Для анализа оптимизации лучше использовать данные за весь период. Попробуйте: 'Где можно сэкономить?'"
                    
            elif parsed['intent'] == 'benchmark':
                # Бенчмаркинг и средние значения
                year_param = parsed.get('year', '2025')
                if year_param == 'all':
                    # Запрос за весь доступный период
                    start_month_yyyymm = '202401'
                    end_month_yyyymm = '202512'
                    period_display = "все время"
                else:
                    # Запрос за конкретный год
                    start_month_yyyymm = f"{year_param}01"
                    end_month_yyyymm = f"{year_param}12"
                    period_display = f"{year_param} год"
                
                response = requests.post(
                    f'{toolbox_url}/api/tool/get_cost_trends/invoke',
                    json={'start_month': start_month_yyyymm, 'end_month': end_month_yyyymm}
                )
                response.raise_for_status()
                result = response.json()
                data = json.loads(result['result']) if 'result' in result else result
                
                if data:
                    costs = [row['total_cost'] for row in data]
                    avg_cost = sum(costs) / len(costs)
                    min_cost = min(costs)
                    max_cost = max(costs)
                    median_cost = sorted(costs)[len(costs)//2]
                    
                    answer = f"📊 **Бенчмарк расходов за {period_display}**\n\n"
                    answer += f"💰 **Средние расходы в месяц:** {avg_cost:.2f} {data[0]['currency']}\n"
                    answer += f"📈 **Максимальные расходы:** {max_cost:.2f} {data[0]['currency']}\n"
                    answer += f"📉 **Минимальные расходы:** {min_cost:.2f} {data[0]['currency']}\n"
                    answer += f"📊 **Медианные расходы:** {median_cost:.2f} {data[0]['currency']}\n\n"
                    
                    # Анализ стабильности
                    variance = sum((x - avg_cost) ** 2 for x in costs) / len(costs)
                    std_dev = variance ** 0.5
                    cv = (std_dev / avg_cost) * 100
                    
                    if cv < 20:
                        stability = "стабильные"
                    elif cv < 50:
                        stability = "умеренно изменчивые"
                    else:
                        stability = "сильно изменчивые"
                    
                    answer += f"📈 **Стабильность расходов:** {stability} (коэффициент вариации: {cv:.1f}%)\n"
                else:
                    answer = f"Нет данных за {period_display} для бенчмаркинга."
                    
            elif parsed['intent'] == 'anomaly':
                # Поиск аномалий и пиков
                year_param = parsed.get('year', '2025')
                if year_param == 'all':
                    start_month_yyyymm = '202401'
                    end_month_yyyymm = '202512'
                    period_display = "все время"
                else:
                    start_month_yyyymm = f"{year_param}01"
                    end_month_yyyymm = f"{year_param}12"
                    period_display = f"{year_param} год"
                
                response = requests.post(
                    f'{toolbox_url}/api/tool/get_cost_trends/invoke',
                    json={'start_month': start_month_yyyymm, 'end_month': end_month_yyyymm}
                )
                response.raise_for_status()
                result = response.json()
                data = json.loads(result['result']) if 'result' in result else result
                
                if data:
                    # Сортируем по расходам
                    sorted_data = sorted(data, key=lambda x: x['total_cost'], reverse=True)
                    top_n = parsed.get('top_n', 3)
                    
                    answer = f"🔍 **Аномалии в расходах за {period_display}**\n\n"
                    answer += f"📈 **Топ-{top_n} самых дорогих месяцев:**\n"
                    
                    for i, row in enumerate(sorted_data[:top_n], 1):
                        month_display = format_month_human(row['invoice.month'])
                        answer += f"{i}. **{month_display}**: {row['total_cost']:.2f} {row['currency']}\n"
                    
                    # Анализ выбросов
                    costs = [row['total_cost'] for row in data]
                    avg_cost = sum(costs) / len(costs)
                    std_dev = (sum((x - avg_cost) ** 2 for x in costs) / len(costs)) ** 0.5
                    
                    outliers = [row for row in data if abs(row['total_cost'] - avg_cost) > 2 * std_dev]
                    
                    if outliers:
                        answer += f"\n🚨 **Выявлены аномалии (выбросы):**\n"
                        for outlier in outliers:
                            month_display = format_month_human(outlier['invoice.month'])
                            deviation = ((outlier['total_cost'] - avg_cost) / avg_cost) * 100
                            answer += f"• **{month_display}**: {outlier['total_cost']:.2f} {outlier['currency']} ({deviation:+.1f}% от среднего)\n"
                    else:
                        answer += f"\n✅ **Аномалий не обнаружено** - расходы в пределах нормы"
                else:
                    answer = f"Нет данных за {period_display} для поиска аномалий."
                    
            elif parsed['intent'] == 'efficiency':
                # Анализ эффективности
                answer = (
                    "⚡ **Анализ эффективности использования GCP**\n\n"
                    "Для полного анализа эффективности нужны дополнительные метрики:\n\n"
                    "📊 **Доступные метрики:**\n"
                    "• Стоимость на пользователя\n"
                    "• ROI облачной инфраструктуры\n"
                    "• Эффективность использования ресурсов\n"
                    "• Сравнение с industry benchmarks\n\n"
                    "🔧 **Рекомендации для улучшения:**\n"
                    "• Используйте автоматическое масштабирование\n"
                    "• Оптимизируйте размеры инстансов\n"
                    "• Применяйте коммитменты для стабильных нагрузок\n"
                    "• Мониторьте неиспользуемые ресурсы\n\n"
                    "Попробуйте запросить:\n"
                    "• 'Где можно сэкономить?'\n"
                    "• 'Анализ по сервисам'\n"
                    "• 'Сравнение расходов по месяцам'"
                )

            else:
                # Проверяем, является ли это unknown интентом
                if parsed['intent'] == 'unknown':
                    # Специальное сообщение для нераспознанных запросов
                    answer = (
                        "❓ **Запрос не распознан**\n\n"
                        "Извините, я не смог понять ваш вопрос. Попробуйте переформулировать запрос или выберите один из примеров:\n\n"
                        "📊 **Анализ расходов:**\n"
                        "• 'Какие были общие затраты за июль?'\n"
                        "• 'Покажи динамику затрат за 2025 год'\n"
                        "• 'Сколько потрачено на Cloud Storage?'\n\n"
                        "🔍 **Детальный анализ:**\n"
                        "• 'Топ-5 самых дорогих сервисов'\n"
                        "• 'Сравни затраты июль vs август'\n"
                        "• 'Покажи расходы по дням за сентябрь'\n\n"
                        "💡 **Рекомендации:**\n"
                        "• 'Где можно сэкономить?'\n"
                        "• 'Найди аномалии в расходах'\n"
                        "• 'Средние расходы в месяц'\n\n"
                        "🚀 **Дополнительные возможности:**\n"
                        "• 'Покажи расходы по проектам'\n"
                        "• 'Детализация за конкретную дату'\n"
                        "• 'Анализ эффективности ресурсов'\n\n"
                        "Или используйте вкладки для просмотра графиков!"
                    )
                else:
                    # Используем LLM для ответа на общие вопросы с учетом контекста
                    try:
                        # Формируем контекст из истории
                        context_summary = ""
                        if history and len(history) > 0:
                            recent = history[-4:]  # последние 2 обмена
                            for msg in recent:
                                if msg['role'] == 'user':
                                    context_summary += f"Q: {msg['content']}\n"
                                else:
                                    context_summary += f"A: {msg.get('answer', '')[:200]}...\n"

                        llm_prompt = f"""Ты - помощник по анализу расходов Google Cloud Platform.

Контекст разговора:
{context_summary}

Текущий вопрос пользователя: "{request.question}"

Ответь на вопрос пользователя кратко и по делу, используя контекст разговора. Если вопрос о сервисе GCP - дай краткое объяснение что это такое и для чего используется (1-2 предложения). Если не можешь ответить - предложи варианты запросов о расходах."""

                        model = genai.GenerativeModel('gemini-2.0-flash-exp')
                        response_llm = model.generate_content(llm_prompt)
                        answer = response_llm.text.strip()
                    except Exception as llm_error:
                        logger.error(f"LLM generation error: {llm_error}")
                        answer = (
                            "❓ **Запрос не распознан**\n\n"
                            "Извините, я не смог понять ваш вопрос. Попробуйте переформулировать запрос или выберите один из примеров:\n\n"
                            "📊 **Анализ расходов:**\n"
                            "• 'Какие были общие затраты за июль?'\n"
                            "• 'Покажи динамику затрат за 2025 год'\n"
                            "• 'Сколько потрачено на Cloud Storage?'\n\n"
                            "🔍 **Детальный анализ:**\n"
                            "• 'Топ-5 самых дорогих сервисов'\n"
                            "• 'Сравни затраты июль vs август'\n"
                            "• 'Покажи расходы по дням за сентябрь'\n\n"
                            "💡 **Рекомендации:**\n"
                            "• 'Где можно сэкономить?'\n"
                            "• 'Найди аномалии в расходах'\n"
                            "• 'Средние расходы в месяц'\n\n"
                            "Или используйте вкладки для просмотра графиков!"
                        )

        except Exception as tool_error:
            logger.error(f"Tool error: {tool_error}")
            answer = f"Ошибка при получении данных: {str(tool_error)}"

        # Сохраняем историю разговора
        if session_id not in conversation_history:
            conversation_history[session_id] = []

        # Добавляем вопрос пользователя
        conversation_history[session_id].append({
            "role": "user",
            "content": request.question
        })

        # Добавляем ответ ассистента
        conversation_history[session_id].append({
            "role": "assistant",
            "answer": answer,
            "parsed": parsed
        })

        # Ограничиваем историю последними 10 сообщениями (5 пар вопрос-ответ)
        if len(conversation_history[session_id]) > 10:
            conversation_history[session_id] = conversation_history[session_id][-10:]

        return ChatResponse(
            answer=answer,
            data=None
        )
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
        elif isinstance(result, str):
            data = json.loads(result)
        else:
            data = result

        # Форматируем для графика
        chart_data = {
            "labels": [str(row['month']) for row in data],
            "datasets": [{
                "label": "Затраты (ILS)",
                "data": [float(row['total_cost']) for row in data],
                "borderColor": "rgb(75, 192, 192)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "tension": 0.1
            }]
        }

        return {
            "success": True,
            "chart_data": chart_data,
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
        elif isinstance(result, str):
            data = json.loads(result)
        else:
            data = result

        # Топ-10 сервисов по изменению
        top_services = sorted(
            data,
            key=lambda x: abs(x['cost_difference']),
            reverse=True
        )[:10]

        chart_data = {
            "labels": [row['service_name'][:30] for row in top_services],
            "datasets": [
                {
                    "label": f"Месяц {month1}",
                    "data": [float(row['month1_cost']) for row in top_services],
                    "backgroundColor": "rgba(54, 162, 235, 0.5)",
                },
                {
                    "label": f"Месяц {month2}",
                    "data": [float(row['month2_cost']) for row in top_services],
                    "backgroundColor": "rgba(255, 99, 132, 0.5)",
                }
            ]
        }

        return {
            "success": True,
            "chart_data": chart_data,
            "raw_data": top_services
        }

    except Exception as e:
        logger.error(f"Error getting service comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/monthly-cost")
async def get_monthly_cost(invoice_month: str):
    """
    Получить общую сумму затрат за месяц
    """
    try:
        toolbox_url = os.getenv('TOOLBOX_URL', 'http://127.0.0.1:5001')

        # Вызываем инструмент напрямую через HTTP
        import requests
        response = requests.post(
            f'{toolbox_url}/api/tool/get_monthly_cost_summary/invoke',
            json={'invoice_month': invoice_month}
        )
        response.raise_for_status()
        result = response.json()

        import json
        if isinstance(result, dict) and 'result' in result:
            data = json.loads(result['result'])
        elif isinstance(result, str):
            data = json.loads(result)
        else:
            data = result

        return {
            "success": True,
            "data": data[0] if data else None
        }

    except Exception as e:
        logger.error(f"Error getting monthly cost: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/data/cost-by-service")
async def get_cost_by_service(invoice_month: str):
    """
    Получить разбивку затрат по сервисам за месяц
    """
    try:
        toolbox_url = os.getenv('TOOLBOX_URL', 'http://127.0.0.1:5001')

        # Вызываем инструмент напрямую через HTTP
        import requests
        response = requests.post(
            f'{toolbox_url}/api/tool/get_cost_by_service/invoke',
            json={'invoice_month': invoice_month}
        )
        response.raise_for_status()
        result = response.json()

        import json
        if isinstance(result, dict) and 'result' in result:
            data = json.loads(result['result'])
        elif isinstance(result, str):
            data = json.loads(result)
        else:
            data = result

        # Топ-10 сервисов
        top_services = data[:10]

        chart_data = {
            "labels": [row['description'] for row in top_services],
            "datasets": [{
                "label": "Затраты (ILS)",
                "data": [float(row['final_cost']) for row in top_services],
                "backgroundColor": [
                    'rgba(255, 99, 132, 0.5)',
                    'rgba(54, 162, 235, 0.5)',
                    'rgba(255, 206, 86, 0.5)',
                    'rgba(75, 192, 192, 0.5)',
                    'rgba(153, 102, 255, 0.5)',
                    'rgba(255, 159, 64, 0.5)',
                    'rgba(199, 199, 199, 0.5)',
                    'rgba(83, 102, 255, 0.5)',
                    'rgba(255, 99, 255, 0.5)',
                    'rgba(99, 255, 132, 0.5)',
                ],
            }]
        }

        return {
            "success": True,
            "chart_data": chart_data,
            "raw_data": data
        }

    except Exception as e:
        logger.error(f"Error getting cost by service: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI application...")
    print(f"App: {app}")
    print(f"App title: {app.title}")
    port = int(os.getenv("PORT", 8081))
    print(f"Port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
