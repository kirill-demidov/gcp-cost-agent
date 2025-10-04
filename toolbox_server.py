from fastapi import FastAPI
import uvicorn
import subprocess
import os

app = FastAPI(title="GCP Cost Toolbox")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/api/tool/{tool_name}/invoke")
def invoke_tool(tool_name: str, request: dict):
    # Simple proxy to local toolbox
    try:
        # For now, return mock data
        if "get_cost_by_service" in tool_name:
            return {"result": '[{"service": "BigQuery", "cost": 568.17, "currency": "ILS"}, {"service": "Cloud Storage", "cost": 259.37, "currency": "ILS"}]'}
        elif "get_monthly_cost_summary" in tool_name:
            # Accept both 'month' and 'invoice_month' parameters
            month_param = request.get('invoice_month') or request.get('month', '202510')
            return {"result": f'{{"total_cost": 827.54, "currency": "ILS", "month": "{month_param}"}}'}
        else:
            return {"result": "Tool not implemented yet"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)
