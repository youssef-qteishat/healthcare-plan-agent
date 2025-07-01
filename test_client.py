import asyncio
import base64
import json
from fastmcp import Client

client = Client(transport="http://127.0.0.1:8000/mcp/")

async def test_policy_analyzer(pdf_path: str, surgery_query: str):
    async with client:
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        # encode to base64 string
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        raw_inputs = await client.call_tool("extract_inputs", {
            "policy_pdf_bytes": pdf_b64,
            "user_query": surgery_query
        })
        print("Raw Extracted Inputs:", raw_inputs)

        # raw_inputs is likely a list of TextContent objects, extract the text and parse JSON
        inputs_json_str = raw_inputs[0].text  # get the text from first TextContent
        inputs = json.loads(inputs_json_str)  # parse string into dict

        print("Parsed Inputs:", inputs)

        coverage = await client.call_tool("analyze_policy", {
            "surgery": inputs["surgery"],
            "policy_text": inputs["policy_text"]
        })

        formatted = await client.call_tool("format_result", {
            "coverage": coverage
        })
        print("Formatted result:", formatted)

if __name__ == "__main__":
    asyncio.run(test_policy_analyzer("cigna_policy.pdf", "ACL Reconstruction"))
