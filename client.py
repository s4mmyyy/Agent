import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict
from promot import REACT_PROMPT_TEMPLATE
from tools import ToolExecutor, Search
import re


load_dotenv()

class HelloAgentsLLM:
    '''
    为本书"Hello Agents"定制的LLM客户端
    它用于调用任何兼容OpenAI接口的服务，并默认使用流式响应
    '''
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout:int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载
        """
        self.model =model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT",60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务器地址必须被提供或者在.env文件夹内被定义")
    
        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float= 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🧠正在调用{self.model}模型...")
        try:
            print(f"准备请求，模型：{self.model}，温度：{temperature}，消息条数：{len(messages)}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )

            #处理流式响应
            print("✅大语言模型响应成功:")
            collected_content = []
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                print(content, end="",flush=True)
                collected_content.append(content)
            print() #在流式输出结束后换行
            return "".join(collected_content)
        
        except Exception as e:
            print(f"❌调用LLM API时发生错误:{e}")
            return None


class ReactLLM:
    def __init__(self, llm_client: HelloAgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str) -> str:
        """
        运行ReAct智能体来回答一个问题
        """
        self.history = [] #每次运行时重置历史记录
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"--- 第{current_step}步 ---")

            #1.格式化提示词
            print("格式化提示词")
            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            #tool_name = "XXXXX"
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools= tools_desc,
                question= question,
                history= history_str,
            )
            #print(prompt)
            #2.调用LLM进行思考
            messages = [{"role":"user", "content":prompt}]
            
            response_text = self.llm_client.think(messages=messages)

            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break

            #3.解析LLM的输出
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"思考: {thought}")
            
            if not action:
                print("错误:未能解析出有效的Action，流程终止")
                break

            #4. 执行 Action
            if action.strip().startswith("Finish"):
                # 如果是 Finish，提取最终答案并结束（兼容多种括号/格式）
                match = re.match(r"Finish\s*\[(.*)\]", action.strip(), re.DOTALL)
                if match:
                    final_answer = match.group(1).strip()
                else:
                    # 模型没用 Finish[答案] 格式时，去掉 "Finish" 前缀作为答案
                    final_answer = re.sub(r"^Finish\s*[\[:：\s]+", "", action.strip()).strip() or action.strip()
                print(f"🎉最终答案: {final_answer}")
                return final_answer
            
            #5. 执行工具
            tool_name, tool_input = self._parse_action(action)
            if not tool_name or not tool_input:
                print("错误:未能解析出有效的工具名称，流程终止")
                break

            tool_function = self.tool_executor.getTool(tool_name)
            if not tool_function:
                observation = f"错误:未找到名为 '{tool_name}' 的工具。"
            else:
                observation = tool_function(tool_input) #调用真实工具
                # (这段逻辑紧随工具调用之后，在 while 循环的末尾)
            print(f"👀 观察: {observation}")
            
            # 将本轮的Action和Observation添加到历史记录中
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达最大步数,流程终止")
        return None



    def _parse_output(self,text:str):
        """
        解析LLM的响应，提取Action和Observation
        """
        #Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        #Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text:str):
        """解析Action，提取工具名称和输入参数"""
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return None, None



if __name__ == '__main__' :
    try:
        llm = HelloAgentsLLM()
        # 2️⃣ 初始化工具执行器
        tool_executor = ToolExecutor()

        # 3️⃣ 初始化 ReAct Agent
        llmClient = ReactLLM(llm, tool_executor)

        search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
        tool_executor.registerTool("Search", search_description, Search.search)
        print("\n--- 可用的工具 ---")
        print(tool_executor.getAvailableTools())
        print("--- 调用LLM ---")
        responseText = llmClient.run("帮我查询2025124期中国体育彩票双色球开奖号码")
        if responseText:
            print("\n\n--- 完整模型响应 ---")
            print(responseText)
    except ValueError as e:
        print(e)

