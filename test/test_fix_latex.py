import json
import sys
sys.path.insert(0, '../server')
from llm.llm_service import LLMClient

# 测试修复LaTeX转义序列的功能
client = LLMClient()

# 测试用例：包含LaTeX转义序列的JSON字符串
test_json = '''{
    "keypoints": [
        {
            "topic": "测试",
            "source_snippet": "We consider a simple overlapping generations (OLG) economy in discrete time $t \\in \\{ \\dots, -2, -1, 0, 1, 2, \\ldots \\}$ ."
        }
    ]
}'''

print("测试1: 包含LaTeX转义序列的JSON")
try:
    result = client._parse_json_response(test_json)
    print("✓ 解析成功!")
    print(json.dumps(result, ensure_ascii=False, indent=2))
except Exception as e:
    print(f"✗ 解析失败: {e}")

# 测试用例2: 读取实际的response1.json文件
print("\n测试2: 解析response1.json文件")
try:
    with open('response1.json', 'r', encoding='utf-8') as f:
        content = f.read()
    result = client._parse_json_response(content)
    print("✓ 解析成功!")
    print(f"找到 {len(result.get('keypoints', []))} 个关键点")
except Exception as e:
    print(f"✗ 解析失败: {e}")

