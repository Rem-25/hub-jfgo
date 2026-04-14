"""
保险公司Agent示例 - 演示大语言模型的function call能力
设计场景、设计function，完成function_call
展示从用户输入 -> 工具调用 -> 最终回复的完整流程
"""
import json
from openai import OpenAI
from config import Api_key

def get_products():
    """"获取产品列表"""
    products = [
        {
            "id": "Magnetic_001",
            "name": "磁悬浮冷水机组",
            "type": "主机",
            "description":"b2b推荐机型,适合预算充足、洁净程度要求非常高条件下购买，噪音程度低",
            "min_amount":1,
            "max_amount":10,
            "min_cost":15,
            "max_cost":120
        },
        {
            "id": "Centrifugal_001",
            "name":"水冷式离心机组",
            "type":"主机",
            "description":"b2b推荐机型，适合预算一般、洁净程度要求中等条件下购买，噪音程度中等",
            "min_amount":1,
            "max_amount":20,
            "min_cost":10,
            "max_cost":95
        },
        {
            "id": "Screw_001",
            "name":"水冷螺杆机组",
            "type":"主机",
            "description":"b2b推荐机型，适合预算低、洁净程度低条件下购买，噪音程度高",
            "min_amount":1,
            "max_amount":20,
            "min_cost":5,
            "max_cost":35
        }
    ]
    return json.dumps(products, ensure_ascii=False)

def calculate_cost(product_id: str,cost:int, amount: int):
    """Args:
        product_id:产品ID
        cost:单台最低金额
        amount:购买台数
        dealer_coefficient:经销商系数
    """
    dealer_coefficient =  {
        "Magnetic_001":0.85,
        "Centrifugal_001":0.77,
        "Screw_001":0.75

    }
    if product_id not in dealer_coefficient:
        return json.dumps({"error": "产品不存在"}, ensure_ascii=False)

    d_c = dealer_coefficient[product_id]
    total = amount * d_c * cost
    result = {
        "product_id": product_id,
        "amount": amount,
        "cost": cost,
        "total": round(total, 2),
        "calculation_note":f"购买{amount}台,编号{product_id},预计需要{total}万元"

    }
    return json.dumps(result, ensure_ascii=False)

#================================工具函数Json Schema定义===============================
tools = [
    {
        "type":"function",
        "function":{
            "name":"get_products",
            "description":"获取产品列表，包括了产品名称、单台最低价格（万元）、单台最高价格（万元）、单次最低预定量，单次最高预定量",
            "parameters":{
            "type":"object",
            "properties":{},
            "required":[]
            }
        }
    },
    {
        "type":"function",
        "function":{
            "name":"calculate_cost",
            "description":"计算指定产品、指定台数、指定价格的的经销商费用",
            "parameters":{
                "type":"object",
                "properties":{
                    "product_id":{
                        "type":"string",
                        "description":"产品ID"
                },
                    "cost:":{
                        "type":"integer",
                        "description":"产品单价"
                    },
                    "amount":{
                        "type":"integer",
                        "description":"产品数量"
                }
            },
                "required":["product_id","cost","amount"]
        }
    }
    }
]

#====================核心逻辑====================
available_functions = {
    "get_products":get_products,
    "calculate_cost":calculate_cost
}

def run_agent(user_query: str, api_key = None, model:str = "qwen-plus"):
    client = OpenAI(api_key=Api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    messages = [
        {"role":"system",
         "content":"""你是一位专业的暖通空调销售工程师。你可以：
         1.介绍各种机型的空调
         2.根据客户要求计算经销商价格
         
         请根据用户的问题，使用合适的工具来获取信息并给出专业建议。"""
        },
        {
            "role":"user",
            "content":user_query
         }
    ]

    print("\n" + "=" * 60)
    print("【用户问题】：")
    print(user_query)
    print("=" * 60)

    #Agent循环，最多5轮
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        print(f"\n--第{iteration}轮Agent思考 ---")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice= "auto"
        )

        response_message = response.choices[0].message

        messages.append(response_message)
        tool_calls = response_message.tool_calls

        if not tool_calls:
            print("\n【Agent最终回复】")
            print(response_message.content)
            print("=" * 60)
            return response_message.content

        print(f"\n【Agent调用了{len(tool_calls)}个工具】")

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            print(f"\n工具名称：{function_name}")
            print(f"\n工具参数：{json.dumps(function_args, ensure_ascii=False)}")

            if function_name in available_functions:
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)

                print(f"工具返回:{function_response[:200]}..." if len(function_response) > 200 else f"工具返回:{function_response}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": function_name,
                    "content": function_response
                })
            else:
                print(f"错误：未找到工具 {function_name}")

    print("\n【警告】达到最大迭代次数，Agent循环结束")
    return "抱歉，处理您的请求时遇到了问题。"

def demo_scenarios():
    """
    演示几个典型场景
    """
    print("\n" + "#"*60)
    print("# 暖通公司Agent演示 - Function Call能力展示")
    print("#"*60)

    scenarios = [
        "你们有哪些空调产品？",
        "我想了解一下磁悬浮系列的详细信息",
        "我想买几台离心机，预算大概50万元，大概能怎么分配？"
        ]
    print("\n以下是几个示例场景，您可以选择其中一个运行：\n")
    for i, scenario in enumerate(scenarios, 1):
        print(f"{i}. {scenario}")

    print("\n" + "-" * 60)
    print("要运行示例，请取消注释main函数中的相应代码")
    print("并确保设置了环境变量：DASHSCOPE_API_KEY")
    print("-" * 60)

if __name__ == "__main__":
    run_agent("买一台磁悬浮机组最少需要多少钱？")

client = OpenAI(api_key=Api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
response = client.models.list()
print(response)


