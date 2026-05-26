import json
import re
import torch

class Tool:
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func
        
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class Agent:
    def __init__(self, model, tokenizer, device, tools=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.system_prompt = self._build_system_prompt()
        
    def _build_system_prompt(self):
        prompt = "You are a helpful AI assistant that can use tools.\n"
        if self.tools:
            prompt += "You have access to the following tools:\n"
            for name, tool in self.tools.items():
                prompt += f"- {name}: {tool.description}\n"
            prompt += "To use a tool, respond with:\nAction: tool_name\nAction Input: arguments\n"
        prompt += "If you can answer without tools, just answer the user.\n"
        return prompt

    def generate(self, prompt, max_tokens=150):
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        output_ids = self.model.generate(
            input_tensor, 
            max_new_tokens=max_tokens, 
            temperature=0.2, # Low temperature for agent reasoning
            top_p=0.9
        )[0].tolist()
        return self.tokenizer.decode(output_ids[len(input_ids):])
        
    def run(self, user_input, max_steps=5):
        prompt = self.system_prompt + f"\nUser: {user_input}\nAssistant:"
        
        for step in range(max_steps):
            response = self.generate(prompt)
            print(f"\n[Agent Thought/Response]:\n{response.strip()}")
            
            action_match = re.search(r"Action:\s*(\w+)", response)
            input_match = re.search(r"Action Input:\s*(.*)", response)
            
            if action_match and input_match:
                tool_name = action_match.group(1).strip()
                tool_input = input_match.group(1).strip()
                
                if tool_name in self.tools:
                    print(f"\n[System]: Running tool '{tool_name}' with input '{tool_input}'...")
                    try:
                        tool_result = str(self.tools[tool_name](tool_input))
                    except Exception as e:
                        tool_result = f"Error: {e}"
                    print(f"[System Observation]: {tool_result}")
                    prompt += response + f"\nObservation: {tool_result}\nAssistant:"
                else:
                    error_msg = f"Tool {tool_name} not found."
                    print(f"[System Observation]: {error_msg}")
                    prompt += response + f"\nObservation: {error_msg}\nAssistant:"
            else:
                return response.strip()
        return "Agent stopped: Max steps reached."
