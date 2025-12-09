import json
import mistune


with open('benchmarks/gpt-4o.json', 'r') as f:
    data = json.load(f)

data = data[0]['problem']

renderer = mistune.create_markdown(plugins=['strikethrough', 'table', 'math'])
print(renderer(data))
