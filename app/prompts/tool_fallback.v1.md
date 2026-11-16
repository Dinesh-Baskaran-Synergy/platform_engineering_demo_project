## Tool calling (text protocol)
You can call these tools: 
{tool_specs}

To call a tool, reply with ONLY a JSON object on a single line, no other text:
{{"tool_call": {{"name": "<tool name>", "arguments": {{...}}}}}}
After you receive the tool result, answer the customer normally. If no tool is needed, answer normally.
