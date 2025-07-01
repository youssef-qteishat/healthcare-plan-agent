import base64
from fastmcp import FastMCP, Client
from fastmcp.tools import Tool
from langchain_core.tools import tool
from agents.policy_analyzer import analyze_policy
from utils.pdf_parser import extract_text_from_pdf

mcp = FastMCP(name="MyServer")
app = mcp.http_app()

@mcp.tool
def extract_inputs(policy_pdf_bytes: str, user_query: str) -> dict:
    """
    policy_pdf_bytes: base64-encoded string received from client
    Decode it to bytes, then extract text.
    """
    # Decode base64 string to bytes
    pdf_bytes = base64.b64decode(policy_pdf_bytes)
    
    policy_text = extract_text_from_pdf(pdf_bytes)
    return {"surgery": user_query, "policy_text": policy_text}

@mcp.tool
def format_result(coverage: str) -> str:
    return f"Result: {coverage}"

analyze_policy_tool = mcp.tool(analyze_policy)

client = Client(mcp)

if __name__ == "__main__":
    mcp.run(transport="http")