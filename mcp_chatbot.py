
from dotenv import load_dotenv
import google.generativeai as genai
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typing import List, Dict, TypedDict
from contextlib import AsyncExitStack
import json
import asyncio
import nest_asyncio

nest_asyncio.apply()
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class ToolDefinition(TypedDict):
    name: str
    description: str
    input_schema: dict

def convert_mcp_tools_to_gemini_declarations(mcp_tools):
    """
    Convert MCP tool list to Gemini-compatible function declarations.
    Recursively handle nested schemas and array items.
    """
    function_declarations = []
    for tool in mcp_tools:
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        input_schema = tool.get("input_schema", {})

        for param_name, param_info in input_schema.get("properties", {}).items():
            param_config = {"type": param_info.get("type", "string"), "description": param_info.get("description", "")}
            
            # Handle nested objects or arrays recursively
            if param_info.get("type") == "object" and "properties" in param_info:
                param_config["properties"] = {}
                for nested_name, nested_info in param_info["properties"].items():
                    param_config["properties"][nested_name] = {
                        "type": nested_info.get("type", "string"),
                        "description": nested_info.get("description", "")
                    }
            elif param_info.get("type") == "array" and "items" in param_info:
                param_config["items"] = {
                    "type": param_info["items"].get("type", "string"),
                    "description": param_info["items"].get("description", "")
                }

            parameters["properties"][param_name] = param_config

        if "required" in input_schema:
            parameters["required"] = input_schema["required"]

        declaration = {
            "name": tool["name"],
            "description": tool.get("description", f"Tool function: {tool['name']}"),
            "parameters": parameters
        }

        function_declarations.append(declaration)

    return function_declarations

def format_tool_result(result):
    """
    Format MCP tool result for Gemini function response.
    """
    try:
        if hasattr(result, 'content'):
            # Handle MCP response content
            if isinstance(result.content, list):
                # Multiple content items - join them
                content_parts = []
                for item in result.content:
                    if hasattr(item, 'text'):
                        content_parts.append(item.text)
                    else:
                        content_parts.append(str(item))
                return '\n'.join(content_parts)
            elif hasattr(result.content, 'text'):
                # Single text content
                return result.content.text
            else:
                # Direct content
                return str(result.content)
        else:
            # Direct result
            return str(result)
    except Exception as e:
        return f"Error formatting result: {str(e)}"

class MCP_ChatBot:

    def __init__(self):
        # Initialize session and client objects for multiple servers
        self.sessions: List[ClientSession] = []  # List of client sessions
        self.exit_stack = AsyncExitStack()  # Context manager for cleanup
        self.available_tools: List[ToolDefinition] = []  # All tools from all servers
        self.tool_to_session: Dict[str, ClientSession] = {}  # Map tool name to session
        self.chat = None

    async def connect_to_server(self, server_name: str, server_config: dict) -> None:
        """Connect to a single MCP server."""
        try:
            server_params = StdioServerParameters(**server_config)
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.sessions.append(session)
            
            # List available tools for this session
            response = await session.list_tools()
            tools = response.tools
            print(f"\nConnected to {server_name} with tools:", [t.name for t in tools])
            
            # Map each tool to its session and add to available tools
            for tool in tools:
                self.tool_to_session[tool.name] = session
                self.available_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                })
        except Exception as e:
            print(f"Failed to connect to {server_name}: {e}")

    async def connect_to_servers(self):
        """Connect to all configured MCP servers."""
        try:
            with open("server_config.json", "r") as file:
                data = json.load(file)
            
            servers = data.get("mcpServers", {})
            
            for server_name, server_config in servers.items():
                await self.connect_to_server(server_name, server_config)
                
            # After connecting to all servers, initialize Gemini with all tools
            if self.available_tools:
                gemini_function_declarations = convert_mcp_tools_to_gemini_declarations(self.available_tools)
                
                print(f"\nTotal tools available: {len(self.available_tools)}")
                #print("Tool declarations for Gemini:")
                #for decl in gemini_function_declarations:
                #    print(f"- {decl['name']}: {decl['description']}")

                # Initialize Gemini model with all tools
                model = genai.GenerativeModel(
                    model_name="gemini-2.5-flash",
                    tools=[{
                        "function_declarations": gemini_function_declarations
                    }]
                )

                self.chat = model.start_chat()
                print("\nGemini model initialized with all MCP tools!")
            else:
                print("No tools available. Initializing Gemini without tools.")
                model = genai.GenerativeModel(model_name="gemini-2.5-flash")
                self.chat = model.start_chat()
                
        except Exception as e:
            print(f"Error loading server configuration: {e}")
            raise

    async def process_query(self, query):
        try:
            response = self.chat.send_message(query)
            process_query = True

            while process_query:
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    
                    if hasattr(candidate.content, 'parts'):
                        function_calls = []  # Store all function calls in this turn
                        has_function_call = False
                        
                        # Collect all function calls
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call') and part.function_call:
                                has_function_call = True
                                function_calls.append(part.function_call)
                        
                        if has_function_call:
                            # Prepare responses for all function calls
                            function_responses = []
                            for function_call in function_calls:
                                tool_name = function_call.name
                                tool_args = dict(function_call.args)
                                
                                print(f"Calling tool {tool_name} with args {tool_args}")

                                try:
                                    # Get the correct session for this tool
                                    session = self.tool_to_session.get(tool_name)
                                    if not session:
                                        raise Exception(f"No session found for tool {tool_name}")
                                    
                                    # Call the MCP tool using the correct session
                                    result = await session.call_tool(tool_name, arguments=tool_args)
                                    
                                    # Format the result for Gemini
                                    formatted_result = format_tool_result(result)
                                    print(f"Tool result: {formatted_result}")
                                    
                                    # Create function response
                                    function_response = genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=tool_name,
                                            response={"result": formatted_result}
                                        )
                                    )
                                    function_responses.append(function_response)
                                
                                except Exception as e:
                                    print(f"Error executing tool {tool_name}: {str(e)}")
                                    # Create error response
                                    error_response = genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name=tool_name,
                                            response={"error": f"Tool execution failed: {str(e)}"}
                                        )
                                    )
                                    function_responses.append(error_response)

                            # Send all function responses back to Gemini
                            if function_responses:
                                response = self.chat.send_message(function_responses)
                        else:
                            # Handle regular text responses
                            for part in candidate.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    print(part.text)
                            process_query = False
                    else:
                        # No parts in content, print the response text if available
                        if hasattr(response, 'text'):
                            print(response.text)
                        process_query = False
                else:
                    # No candidates, something went wrong
                    print("No response generated")
                    process_query = False

        except Exception as e:
            print(f"Error processing query: {str(e)}")

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMulti-Server MCP Gemini Chatbot Started!")
        #print("Available tools from all connected servers:")
        #for tool in self.available_tools:
        #    print(f"  - {tool['name']}: {tool['description']}")
        print("\nType your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == 'quit':
                    break
                await self.process_query(query)
                print("\n" + "-"*50)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Cleanly close all resources using AsyncExitStack."""
        await self.exit_stack.aclose()

async def main():
    print("Starting Multi-Server MCP Gemini Chatbot...")
    chatbot = MCP_ChatBot()
    try:
        # Connect to all servers defined in server_config.json
        await chatbot.connect_to_servers()
        await chatbot.chat_loop()
    finally:
        # Ensure proper cleanup of all connections
        await chatbot.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
