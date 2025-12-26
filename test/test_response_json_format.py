import json

with open('response1.json', 'r') as f:
    data = json.load(f)

print(data)