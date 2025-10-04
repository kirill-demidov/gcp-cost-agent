"""
Cloud Run version of the GCP Cost Agent API
Optimized for production deployment
"""

import os
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import google.generativeai as genai
import aiohttp
import json
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI(
    title="GCP Cost Agent",
    description="Agent for analyzing Google Cloud Platform costs",
    version="1.0.0"
)

# Конфигурация
TOOLBOX_URL = os.getenv("TOOLBOX_URL", "http://127.0.0.1:5001")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "${GCP_PROJECT_ID}")
BILLING_TABLE = os.getenv("BILLING_TABLE", "${BILLING_TABLE}")

# Инициализация Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Модели данных
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    data: Optional[Dict] = None

# Хранение истории разговоров (в продакшене лучше использовать Redis или БД)
conversation_history: Dict[str, List[Dict]] = {}

# Статические файлы
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    """Serve the frontend"""
    return FileResponse('frontend/index.html')

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "gcp-cost-agent"}

def format_month_human(yyyymm: str) -> str:
    """Format YYYYMM to human readable format"""
    try:
        year = yyyymm[:4]
        month = yyyymm[4:6]
        month_names = {
            "01": "Январь", "02": "Февраль", "03": "Март", "04": "Апрель",
            "05": "Май", "06": "Июнь", "07": "Июль", "08": "Август",
            "09": "Сентябрь", "10": "Октябрь", "11": "Ноябрь", "12": "Декабрь"
        }
        return f"{month_names.get(month, month)} {year}"
    except:
        return yyyymm

def understand_query_with_llm(question: str, history: List[Dict]) -> Dict:
    """Use LLM to understand user query and extract parameters"""
    
    # Формируем контекст из истории
    context_summary = ""
    if history and len(history) > 0:
        recent = history[-4:]  # последние 2 обмена
        for msg in recent:
            if msg['role'] == 'user':
                context_summary += f"Q: {msg['content']}\n"
            else:
                context_summary += f"A: {msg.get('answer', '')[:200]}...\n"

    prompt = f"""Ты - AI помощник для анализа расходов Google Cloud Platform. 
Твоя задача - понять запрос пользователя и извлечь параметры для анализа.

КОНТЕКСТ РАЗГОВОРА:
{context_summary}

ТЕКУЩИЙ ВОПРОС: "{question}"

ВНИМАНИЕ: Используй контекст разговора! Если пользователь задает неполный вопрос (например, "по сервисам"), 
то используй информацию из предыдущих сообщений для понимания.

Доступные типы запросов:
- costs: общие затраты за период
- trends: динамика и тренды расходов  
- comparison: сравнение между периодами
- services: разбивка по сервисам
- forecast: прогнозирование расходов
- optimization: рекомендации по оптимизации
- benchmark: бенчмаркинг и метрики
- anomaly: поиск аномалий
- efficiency: анализ эффективности
- unknown: если запрос не связан с GCP/расходами

ВНИМАНИЕ: Если вопрос НЕ СВЯЗАН с GCP, облачными расходами, анализом затрат - ВСЕГДА возвращай "unknown"!

Верни JSON с полями:
{{
  "intent": "costs|trends|comparison|services|forecast|optimization|benchmark|anomaly|efficiency|unknown",
  "month": "YYYYMM или null",
  "year": "YYYY или null", 
  "date": "YYYY-MM-DD или null",
  "date_range": "start_date,end_date или null",
  "service": "название сервиса или null",
  "analysis_type": "list|peak|average|volatility|seasonal|trend|anomaly или null",
  "top_n": число или null
}}

Примеры:
"какие были затраты в июле 2025" -> {{"intent": "costs", "month": "202507", "year": "2025", "date": null, "date_range": null, "service": null, "analysis_type": "list", "top_n": null}}
"покажи динамику за 2025 год" -> {{"intent": "trends", "month": null, "year": "2025", "date": null, "date_range": null, "service": null, "analysis_type": "list", "top_n": null}}
"сравни июль и август" -> {{"intent": "comparison", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": null, "top_n": null}}
"топ-5 сервисов" -> {{"intent": "services", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": "list", "top_n": 5}}
"по сервисам" (после запроса о динамике) -> {{"intent": "services", "month": null, "year": "2025", "date": null, "date_range": null, "service": null, "analysis_type": "list", "top_n": null}}
"прогноз расходов на следующий месяц" -> {{"intent": "forecast", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": null, "top_n": null}}
"где можно сэкономить?" -> {{"intent": "optimization", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": null, "top_n": null}}
"средние расходы в месяц" -> {{"intent": "benchmark", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": "average", "top_n": null}}
"найди аномалии в расходах" -> {{"intent": "anomaly", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": "peak", "top_n": null}}
"есть ли сезонность в расходах?" -> {{"intent": "trends", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": "seasonal", "top_n": null}}
"покажи неиспользуемые ресурсы" -> {{"intent": "efficiency", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": null, "top_n": null}}
"анализ эффективности использования ресурсов" -> {{"intent": "efficiency", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": null, "top_n": null}}
"как настроить wifi" -> {{"intent": "unknown", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": null, "top_n": null}}
"какой сегодня день" -> {{"intent": "unknown", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": null, "top_n": null}}
"хочу купить машину" -> {{"intent": "unknown", "month": null, "year": null, "date": null, "date_range": null, "service": null, "analysis_type": null, "top_n": null}}

Отвечай ТОЛЬКО JSON, без дополнительных комментариев."""

    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content(prompt)
        
        # Извлекаем JSON из ответа
        response_text = response.text.strip()
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text
        
        logger.info(f"Raw LLM response: {response_text}")
        
        parsed = json.loads(json_text)
        logger.info(f"LLM parsed: {parsed}")
        
        return parsed
    except Exception as e:
        logger.error(f"LLM parsing error: {e}")
        return {
            "intent": "unknown",
            "month": None,
            "year": None,
            "date": None,
            "date_range": None,
            "service": None,
            "analysis_type": None,
            "top_n": None
        }

async def call_toolbox_tool(tool_name: str, params: Dict) -> Dict:
    """Call toolbox tool"""
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{TOOLBOX_URL}/api/tool/{tool_name}/invoke"
            payload = {"input": params}
            
            logger.info(f"Calling toolbox: {url} with params: {params}")
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Toolbox error {response.status}: {error_text}")
                    return {"error": f"Toolbox error: {response.status}"}
    except Exception as e:
        logger.error(f"Toolbox connection error: {e}")
        return {"error": str(e)}

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

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint"""
    try:
        logger.info(f"Received question: {request.question}")
        
        # Получаем историю для сессии
        session_id = request.session_id or "default"
        history = conversation_history.get(session_id, [])
        
        # Проверяем, связан ли запрос с GCP/облачными расходами
        question_lower = request.question.lower()
        gcp_keywords = ['gcp', 'google cloud', 'облако', 'расход', 'затрат', 'стоимост', 'биллинг', 'инвойс', 'счет', 'трат', 'потрач', 'потребил', 'использовал', 'cloud', 'compute', 'storage', 'bigquery', 'vertex', 'artifact', 'run', 'functions', 'kubernetes', 'sql', 'logging', 'monitoring', 'analytics', 'машинное обучение', 'база данных', 'хранение', 'вычисления', 'сервис', 'проект', 'ресурс', 'оптимизац', 'экономи', 'сэконом', 'анализ', 'динамик', 'тренд', 'сравн', 'месяц', 'год', 'день', 'период', 'время', 'дата', 'график', 'диаграмм', 'статистик', 'метрик', 'показател']

        # Инициализируем parsed по умолчанию
        parsed = {"intent": "unknown", "month": None, "year": None, "date": None, "date_range": None, "service": None, "analysis_type": None, "top_n": None}
        
        # Если в запросе нет ключевых слов, связанных с GCP/расходами
        if not any(keyword in question_lower for keyword in gcp_keywords):
            # Проверяем, не является ли это общим приветствием или вопросом о GCP
            general_greetings = ['привет', 'hello', 'hi', 'как дела', 'как ты', 'что ты умеешь', 'помощь', 'help', 'что такое gcp', 'что такое google cloud', 'расскажи о gcp', 'что ты можешь', 'функции', 'возможности']
            if any(greeting in question_lower for greeting in general_greetings):
                # Это общий вопрос - передаем LLM
                parsed = understand_query_with_llm(request.question, history)
            else:
                # Не связано с GCP - сразу возвращаем unknown
                parsed = {"intent": "unknown", "month": None, "year": None, "date": None, "date_range": None, "service": None, "analysis_type": None, "top_n": None}
        else:
            # Используем LLM для понимания запроса с учетом контекста
            parsed = understand_query_with_llm(request.question, history)

        answer = ""
        data = None

        # Обработка различных интентов
        if parsed['intent'] == 'costs':
            # Общие затраты за период
            if parsed['month']:
                # За конкретный месяц
                params = {"month": parsed['month']}
                result = await call_toolbox_tool("get_monthly_costs", params)
                
                if 'data' in result and result['data']:
                    month_display = format_month_human(parsed['month'])
                    total_cost = result['data'][0]['total_cost']
                    currency = result['data'][0]['currency']
                    answer = f"💰 **Общие затраты за {month_display}:** {total_cost:.2f} {currency}"
                    data = result['data']
                else:
                    answer = f"❌ Не удалось получить данные за {format_month_human(parsed['month'])}"
            
            elif parsed['year']:
                # За год
                params = {"start_date": f"{parsed['year']}-01-01", "end_date": f"{parsed['year']}-12-31"}
                result = await call_toolbox_tool("get_cost_trends", params)
                
                if 'data' in result and result['data']:
                    total_cost = sum(row['total_cost'] for row in result['data'])
                    currency = result['data'][0]['currency'] if result['data'] else 'ILS'
                    answer = f"💰 **Общие затраты за {parsed['year']} год:** {total_cost:.2f} {currency}\n\n"
                    answer += "📅 **По месяцам:**\n"
                    for row in result['data']:
                        month_display = format_month_human(row['invoice.month'])
                        answer += f"• {month_display}: {row['total_cost']:.2f} {row['currency']}\n"
                    data = result['data']
                else:
                    answer = f"❌ Не удалось получить данные за {parsed['year']} год"
            
            else:
                # За последний месяц
                last_month = datetime.now().replace(day=1) - timedelta(days=1)
                month_str = last_month.strftime("%Y%m")
                params = {"month": month_str}
                result = await call_toolbox_tool("get_monthly_costs", params)
                
                if 'data' in result and result['data']:
                    month_display = format_month_human(month_str)
                    total_cost = result['data'][0]['total_cost']
                    currency = result['data'][0]['currency']
                    answer = f"💰 **Общие затраты за {month_display}:** {total_cost:.2f} {currency}"
                    data = result['data']
                else:
                    answer = f"❌ Не удалось получить данные за {format_month_human(month_str)}"

        elif parsed['intent'] == 'trends':
            # Динамика и тренды
            if parsed['year']:
                params = {"start_date": f"{parsed['year']}-01-01", "end_date": f"{parsed['year']}-12-31"}
            else:
                # За последние 12 месяцев
                end_date = datetime.now()
                start_date = end_date.replace(year=end_date.year-1)
                params = {"start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d")}
            
            result = await call_toolbox_tool("get_cost_trends", params)
            
            if 'data' in result and result['data']:
                start_display = format_month_human(result['data'][0]['invoice.month'])
                end_display = format_month_human(result['data'][-1]['invoice.month'])
                
                if parsed.get('analysis_type') == 'seasonal':
                    answer = analyze_seasonality(result['data'], start_display, end_display)
                else:
                    answer = f"📊 **Динамика затрат с {start_display} по {end_display}:**\n\n"
                    for row in result['data']:
                        month_display = format_month_human(row['invoice.month'])
                        answer += f"📅 {month_display}: {row['total_cost']:.2f} {row['currency']}\n"
                
                data = result['data']
            else:
                answer = "❌ Не удалось получить данные о динамике"

        elif parsed['intent'] == 'services':
            # Разбивка по сервисам
            if parsed['month']:
                params = {"month": parsed['month']}
            elif parsed['year']:
                params = {"year": parsed['year']}
            else:
                # За последний месяц
                last_month = datetime.now().replace(day=1) - timedelta(days=1)
                month_str = last_month.strftime("%Y%m")
                params = {"month": month_str}
            
            if parsed.get('top_n'):
                params['top_n'] = parsed['top_n']
            
            result = await call_toolbox_tool("get_cost_by_service_all_time", params)
            
            if 'data' in result and result['data']:
                answer = f"🔧 **Разбивка по сервисам:**\n\n"
                for i, row in enumerate(result['data'][:parsed.get('top_n', 10)], 1):
                    answer += f"{i}. **{row['service.description']}**: {row['total_cost']:.2f} {row['currency']}\n"
                data = result['data']
            else:
                answer = "❌ Не удалось получить данные по сервисам"

        elif parsed['intent'] == 'comparison':
            # Сравнение периодов
            # Извлекаем месяцы из контекста или используем значения по умолчанию
            month1_yyyymm = '202507'  # По умолчанию июль
            month2_yyyymm = '202506'  # По умолчанию июнь
            
            # Пытаемся извлечь из истории
            if history and len(history) > 0:
                last_parsed = history[-1].get('parsed', {})
                if last_parsed.get('month'):
                    month1_yyyymm = last_parsed['month']
            
            # Пытаемся извлечь из текущего запроса
            if parsed.get('month'):
                month2_yyyymm = parsed['month']
            
            # Получаем данные для обоих месяцев
            params1 = {"month": month1_yyyymm}
            params2 = {"month": month2_yyyymm}
            
            result1 = await call_toolbox_tool("get_monthly_costs", params1)
            result2 = await call_toolbox_tool("get_monthly_costs", params2)
            
            if 'data' in result1 and result1['data'] and 'data' in result2 and result2['data']:
                cost1 = result1['data'][0]['total_cost']
                cost2 = result2['data'][0]['total_cost']
                currency = result1['data'][0]['currency']
                
                month1_display = format_month_human(month1_yyyymm)
                month2_display = format_month_human(month2_yyyymm)
                
                diff = cost1 - cost2
                diff_percent = (diff / cost2 * 100) if cost2 > 0 else 0
                
                answer = f"📊 **Сравнение затрат:**\n\n"
                answer += f"📅 {month1_display}: {cost1:.2f} {currency}\n"
                answer += f"📅 {month2_display}: {cost2:.2f} {currency}\n\n"
                
                if diff > 0:
                    answer += f"📈 **Рост:** +{diff:.2f} {currency} (+{diff_percent:.1f}%)\n"
                elif diff < 0:
                    answer += f"📉 **Снижение:** {diff:.2f} {currency} ({diff_percent:.1f}%)\n"
                else:
                    answer += f"➡️ **Без изменений**\n"
                
                data = {
                    "month1": result1['data'][0],
                    "month2": result2['data'][0],
                    "difference": diff,
                    "difference_percent": diff_percent
                }
            else:
                answer = f"❌ Не удалось получить данные для сравнения {format_month_human(month1_yyyymm)} и {format_month_human(month2_yyyymm)}"

        elif parsed['intent'] == 'forecast':
            # Прогнозирование
            answer = "🔮 **Прогноз расходов:**\n\n"
            answer += "Для точного прогнозирования нужны исторические данные за несколько месяцев. "
            answer += "Рекомендую проанализировать тренды и сезонность в расходах.\n\n"
            answer += "💡 **Советы для прогнозирования:**\n"
            answer += "• Анализируйте динамику за последние 12 месяцев\n"
            answer += "• Учитывайте сезонные паттерны\n"
            answer += "• Мониторьте изменения в использовании сервисов\n"
            answer += "• Планируйте бюджет с учетом роста нагрузки"

        elif parsed['intent'] == 'optimization':
            # Рекомендации по оптимизации
            answer = "💡 **Рекомендации по оптимизации расходов GCP:**\n\n"
            answer += "🔍 **Анализ использования:**\n"
            answer += "• Проверьте неиспользуемые ресурсы\n"
            answer += "• Оптимизируйте размеры виртуальных машин\n"
            answer += "• Используйте коммитированные скидки\n\n"
            answer += "💰 **Управление затратами:**\n"
            answer += "• Настройте алерты на превышение бюджета\n"
            answer += "• Используйте бюджеты и квоты\n"
            answer += "• Регулярно анализируйте отчеты по расходам\n\n"
            answer += "🚀 **Лучшие практики:**\n"
            answer += "• Автоматизируйте масштабирование\n"
            answer += "• Выбирайте правильные типы хранилища\n"
            answer += "• Оптимизируйте запросы к BigQuery"

        elif parsed['intent'] == 'benchmark':
            # Бенчмаркинг
            if parsed.get('analysis_type') == 'average':
                # Средние расходы
                end_date = datetime.now()
                start_date = end_date.replace(year=end_date.year-1)
                params = {"start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d")}
                
                result = await call_toolbox_tool("get_cost_trends", params)
                
                if 'data' in result and result['data']:
                    costs = [row['total_cost'] for row in result['data']]
                    avg_cost = sum(costs) / len(costs)
                    min_cost = min(costs)
                    max_cost = max(costs)
                    currency = result['data'][0]['currency']
                    
                    answer = f"📊 **Бенчмарк расходов:**\n\n"
                    answer += f"💰 **Средние расходы в месяц:** {avg_cost:.2f} {currency}\n"
                    answer += f"📉 **Минимальные расходы:** {min_cost:.2f} {currency}\n"
                    answer += f"📈 **Максимальные расходы:** {max_cost:.2f} {currency}\n"
                    answer += f"📊 **Период анализа:** {len(costs)} месяцев\n"
                    
                    data = {
                        "average": avg_cost,
                        "minimum": min_cost,
                        "maximum": max_cost,
                        "period_months": len(costs),
                        "currency": currency
                    }
                else:
                    answer = "❌ Не удалось получить данные для бенчмаркинга"
            else:
                answer = "📊 **Бенчмаркинг расходов GCP:**\n\n"
                answer += "Для получения точных метрик используйте конкретные запросы:\n"
                answer += "• 'Средние расходы в месяц'\n"
                answer += "• 'Сравни с предыдущим годом'\n"
                answer += "• 'Топ-5 самых дорогих сервисов'"

        elif parsed['intent'] == 'anomaly':
            # Поиск аномалий
            end_date = datetime.now()
            start_date = end_date.replace(year=end_date.year-1)
            params = {"start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d")}
            
            result = await call_toolbox_tool("get_cost_trends", params)
            
            if 'data' in result and result['data']:
                costs = [row['total_cost'] for row in result['data']]
                avg_cost = sum(costs) / len(costs)
                
                # Находим аномалии (отклонение > 50% от среднего)
                anomalies = []
                for row in result['data']:
                    deviation = abs(row['total_cost'] - avg_cost) / avg_cost * 100
                    if deviation > 50:
                        anomalies.append({
                            'month': row['invoice.month'],
                            'cost': row['total_cost'],
                            'deviation': deviation
                        })
                
                if anomalies:
                    answer = f"🚨 **Найдены аномалии в расходах:**\n\n"
                    for anomaly in anomalies:
                        month_display = format_month_human(anomaly['month'])
                        answer += f"📅 {month_display}: {anomaly['cost']:.2f} ILS "
                        answer += f"(отклонение: {anomaly['deviation']:.1f}%)\n"
                    
                    answer += f"\n📊 **Средние расходы:** {avg_cost:.2f} ILS"
                    data = {"anomalies": anomalies, "average": avg_cost}
                else:
                    answer = f"✅ **Аномалий не обнаружено**\n\n"
                    answer += f"📊 **Средние расходы:** {avg_cost:.2f} ILS\n"
                    answer += f"📈 **Период анализа:** {len(costs)} месяцев"
                    data = {"anomalies": [], "average": avg_cost}
            else:
                answer = "❌ Не удалось получить данные для поиска аномалий"

        elif parsed['intent'] == 'efficiency':
            # Анализ эффективности
            answer = "⚡ **Анализ эффективности использования ресурсов:**\n\n"
            answer += "🔍 **Метрики эффективности:**\n"
            answer += "• Соотношение затрат к использованию\n"
            answer += "• Процент неиспользуемых ресурсов\n"
            answer += "• Оптимальность выбора сервисов\n\n"
            answer += "💡 **Рекомендации:**\n"
            answer += "• Регулярно проверяйте использование ресурсов\n"
            answer += "• Настройте автоматическое масштабирование\n"
            answer += "• Используйте мониторинг для оптимизации"

        elif parsed['intent'] == 'unknown':
            # Неизвестный запрос - используем LLM для ответа
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

        # Сохраняем в историю
        history.append({
            "role": "user",
            "content": request.question,
            "timestamp": datetime.now().isoformat()
        })
        history.append({
            "role": "assistant", 
            "answer": answer,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "parsed": parsed
        })
        
        # Ограничиваем историю последними 20 сообщениями
        if len(history) > 20:
            history = history[-20:]
        
        conversation_history[session_id] = history

        return ChatResponse(answer=answer, data=data)

    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            answer=f"❌ Произошла ошибка при обработке запроса: {str(e)}",
            data=None
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
