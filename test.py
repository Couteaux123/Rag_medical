from embed import connect_elasticresearch
from vector_index import extra_subsections
es = connect_elasticresearch()
response = es.get(index = "zhbd", id = "人工牛黄") 
# print(response)
content = response['_source']['content']
#提取所有小标题及其内容
subsections = extra_subsections(content)
print(subsections)
sub_title = '性状'
found_content = []
output = []
if sub_title in subsections and subsections[sub_title]:
    found_content.append(f"小标题 '{sub_title}' 的内容:\n{subsections[sub_title]}")
if found_content:
    output.append("\n".join(found_content))            #返回单个药品中的在数据库中存在字段的内容
else:
    #    没有找到任何小标题，输出完整内容
    output.append(f"未找到任何小标题，输出完整内容:\n{content}")
print(output)