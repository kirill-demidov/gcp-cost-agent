"""
GCP Cost Agent - Агент для анализа расходов Google Cloud Platform
Поддерживает русский и английский языки
"""

import os
from google.adk.agents import Agent
from toolbox_core import ToolboxSyncClient


def create_cost_agent():
    """Создает и настраивает агента для анализа расходов GCP"""

    # Подключаемся к MCP Toolbox серверу
    toolbox_url = os.getenv('TOOLBOX_URL', 'http://127.0.0.1:5000')
    toolbox = ToolboxSyncClient(toolbox_url)

    # Загружаем инструменты из tools.yaml
    tools = toolbox.load_toolset('gcp-cost-agent-tools')

    # Используем Vertex AI через переменные окружения
    os.environ['VERTEXAI_PROJECT'] = os.getenv('VERTEXAI_PROJECT', '${GCP_PROJECT_ID}')
    os.environ['VERTEXAI_LOCATION'] = os.getenv('VERTEXAI_LOCATION', 'us-central1')

    # Создаем агента с поддержкой русского языка
    agent = Agent(
        name="GCPCostAgent",
        model="gemini-2.0-flash",  # Используем Gemini 2.0 Flash через Google AI API
        description=(
            "Агент для анализа расходов Google Cloud Platform. "
            "Предоставляет информацию о затратах, разбивку по проектам и сервисам. "
            "Поддерживает русский и английский языки.\n\n"
            "An agent that provides insights into your Google Cloud Platform (GCP) costs "
            "by querying your billing data in BigQuery. Supports Russian and English languages."
        ),
        instruction=(
            "Вы - эксперт по расходам Google Cloud. Ваша цель - предоставлять точную информацию "
            "о затратах и использовании, запрашивая данные биллинга из BigQuery.\n\n"
            "You are a Google Cloud Cost expert. Your purpose is to provide accurate cost and usage "
            "information by querying the GCP billing export data in BigQuery.\n\n"

            "**ВСЕГДА следуйте этим инструкциям:**\n"
            "**ALWAYS follow these instructions:**\n\n"

            "1. **Определите временной период:**\n"
            "   **Determine the Time Period:**\n"
            "   - Если пользователь указывает месяц (например, 'в августе', 'за сентябрь', 'last month'), "
            "     используйте инструмент, требующий `invoice_month`. Вычислите месяц в формате `YYYYMM`.\n"
            "   - If the user specifies a month (e.g., 'last month', 'in August'), use a tool that requires "
            "     an `invoice_month`. Calculate the month in `YYYYMM` format.\n"
            "   - Если период не указан, ОБЯЗАТЕЛЬНО попросите уточнить.\n"
            "   - If no time period is specified, you MUST ask the user for clarification.\n\n"

            "2. **Выберите правильный инструмент на основе запроса:**\n"
            "   **Select the Right Tool based on the user's request:**\n"
            "   - **Для общих затрат за месяц / For monthly costs:**\n"
            "       - Для общей суммы / For a total summary: `get_monthly_cost_summary`\n"
            "       - Для разбивки по проектам / For a breakdown by project: `get_cost_by_project`\n"
            "       - Для разбивки по сервисам / For a breakdown by service: `get_cost_by_service`\n\n"

            "3. **Представьте информацию четко:**\n"
            "   **Present the Information Clearly:**\n"
            "   - **ВАЖНО**: При представлении любых затрат вы ДОЛЖНЫ также указать код валюты, "
            "     возвращенный инструментом (например, 'Общие затраты составили 15000 RUB' или 'Total cost was 500 USD').\n"
            "   - **IMPORTANT**: When presenting any cost, you MUST also state the currency code returned by "
            "     the tool (e.g., 'The total cost was 15000 RUB' or 'Общие затраты составили 500 USD').\n"
            "   - При разбивке представляйте информацию в виде четкого списка, упорядоченного от наибольших "
            "     к наименьшим затратам.\n"
            "   - When providing breakdowns, present the information in a clear, readable list, ordered from "
            "     most to least expensive.\n"
            "   - ВСЕГДА отвечайте на том языке, на котором задан вопрос (русский или английский).\n"
            "   - ALWAYS respond in the same language as the question (Russian or English).\n\n"

            "4. **Обработка ошибок и пустых результатов:**\n"
            "   **Handle Errors and Empty Results:**\n"
            "   - Если инструмент возвращает ошибку или пустой результат, сообщите пользователю. "
            "     Посоветуйте проверить `invoice_month` и убедиться, что `project` и `table name` "
            "     в `tools.yaml` указаны правильно.\n"
            "   - If a tool returns an error or an empty result, inform the user. Advise them to check "
            "     the `invoice_month` and ensure the `project` and `table name` in `tools.yaml` are correct.\n\n"

            "**Примеры запросов / Example queries:**\n"
            "- 'Какие были общие затраты в сентябре 2025?' / 'What was my total bill in September 2025?'\n"
            "- 'Покажи разбивку по проектам за август 2025' / 'Show me project spending for August 2025'\n"
            "- 'Какой сервис был самым дорогим в сентябре?' / 'Which service was most expensive in September?'\n"
        ),
        tools=tools,
    )

    return agent


def run_agent_query(agent: Agent, query: str) -> str:
    """
    Выполняет запрос к агенту и возвращает ответ

    Args:
        agent: Настроенный агент
        query: Запрос пользователя

    Returns:
        Ответ агента
    """
    try:
        # Используем правильный метод ADK для выполнения запроса
        session = agent.create_session()
        response = session.send_user_message(query)
        return response.messages[-1].content if response.messages else "Нет ответа"
    except Exception as e:
        return f"Ошибка при выполнении запроса / Error executing query: {str(e)}"


# Создаем root_agent для ADK
root_agent = create_cost_agent()


if __name__ == "__main__":
    # Для локального тестирования
    agent = root_agent

    print("GCP Cost Agent запущен / GCP Cost Agent started")
    print("Примеры запросов / Example queries:")
    print("- Какие были общие затраты в сентябре 2025?")
    print("- What was my total bill in August 2025?")
    print("- Покажи разбивку по проектам за август 2025")
    print("\nВведите 'exit' для выхода / Type 'exit' to quit\n")

    while True:
        query = input("\nВаш вопрос / Your question: ").strip()
        if query.lower() == 'exit':
            break
        if query:
            result = run_agent_query(agent, query)
            print(f"\nОтвет / Answer: {result}")
