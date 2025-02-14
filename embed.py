from sentence_transformers import SentenceTransformer
from openai import OpenAI
from elasticsearch import Elasticsearch, exceptions
from vector_index import retrival_data_from_es,extra_subsections
import os
# model="gpt-4o-mini"
# prompt = "Describe what a/an fruit apple looks like in an image, list 8 phrases which contain one key feature with a short description?"
# max_tokens = 300
# temperature=0.5
# # output = dict()


# response = client.chat.completions.create(model=model, max_tokens=max_tokens, n=1, stop=None,
#                                           temperature=temperature,
#                                           messages=[
#                                             {
#                                                 "role": "user", 
#                                                 "content": prompt
#                                             }
#                                         ]
#                                     )
    # 提取生成的文本
# generated_text = response.choices[0].message.content
# generated_text = [item.strip() for item in generated_text.split('\n')]
# print(generated_text)
import re

# def extract_drug_info(text):  # 将标准化后的信息切分
#     # 匹配多个药品名和标准化输出
#     drug_pattern = re.compile(r'提到的药品名：(.+?)\s*标准化输出：\s*(.*?)(?=提到的药品名：|$)', re.DOTALL)
#     matches = drug_pattern.findall(text)
#     print(matches)
#     drugs = []
#     standard_outputs = []
    
#     for match in matches:
#         # 提取药品名，可能包含多个药品，以逗号或其他标点分隔
#         drug_names = [name.strip() for name in re.split(r'[、,，]', match[0]) if name.strip()]
#         # 提取并分割标准化输出的每一行
#         outputs = [line.strip() for line in match[1].strip().split('\n') if line.strip()]
        
#         for drug_name in drug_names:
#             drugs.append(drug_name)
#             standard_outputs.append(outputs)
    
#     print(drugs, standard_outputs)
#     # return drugs, standard_outputs

# 正确的输入格式
# text = """提到的药品名：板蓝根、金银花
# 标准化输出：
# 板蓝根：清热解毒
# 金银花：疏风散热

# 提到的药品名：黄连
# 标准化输出：
# 黄连：泻火解毒
# """

# extract_drug_info(text)

client = OpenAI(
    api_key="sk-KbwhEmvv5gIqt6W6FPGBgmco4Rd9QZsCj1yZwTkYQnGJhHIj",
    base_url="https://api.chatanywhere.tech/v1"
)



class MedicineStandardizer:
    global field_list
    field_list=["药物名","类别", "鉴别", "贮藏", "指纹图谱", "功能主治", "规格", #文档中会出现的小标题
                             "含量测定",  "性味与归经", "浸岀物", 
                             "规定", "制法",  "检査", "用法与用量",
                             "用途",  "触藏", "正丁醇提取物", "特征图谱","禁忌", 
                              "效价测定", "正丁醇浸出物", 
                             "注意事项", "功能与主治", "制剂",
                             "性状","挥发油","处方", 
                             "适应症"]
    def __init__(self, llm) -> None:
        self.field_list = field_list
        self.client = client
        self.llm = llm   #此处仍使用openAI
        
    def bzh(self, input_data): #提取信息，标准化输出
        '''
        从问题中提取字段列表中的信息，若没有则为空
        :param question: 输入的问题。
        :param field_list: 字段列表。
        :return: 一个字典，键为字段名，值为从问题中提取出的对应信息或空字符串。

        '''
        text = f'''问题一般是药品相关的问题，所以字段可能会有近义语句，
        比如制法即是制作方法,有重量的是处方的一部分，“用*制作而成”*也一般是处方，处方中通常只有药材名例如“板蓝根，罂粟壳”，无其他说明
        只有提到的字段才可以出现'''
        all_field_list = ", ".join(self.field_list)
        extract_template = f'''你是个语意理解大师，你需要充分理解问题中的内容含义，问题提到了哪些信息，问题通常答案指向一种药物的名称，所以问题中提到的药物有克数的一般为处方中的内容。你需要把问题中的信息分类到字段中提到的内容中
        问题：{input_data}
        字段：{all_field_list}
        用中文回复以及中文字符，回复时参考以下格式，比如 处方：板蓝根1500g,大青叶2250g。将涉及的字段与信息全部输出，顺序为字段名，提到的字段，同时，未提到的字段不需要输出字段名：信息
        额外信息：{text}
        未涉及的字段一定不要提到。一定不要出现“字段：None”的类似句子通常来说问题中只有2-3个字段内容，确保你不会输出超过3个字段，字段间换行输出
        '''
        response = self.llm.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{
                'role':'user',
                'content':extract_template
            }]
        )
        answer = response.choices[0].message.content.strip() if response.choices else "无法提供答案"
        print(f"bzh标准化的答案: {answer}")
        return answer
    
    def standardize_information(self, input_data):
        '''
        利用大模型标准化输入

        return 标准化后的信息 包含药品名
        '''
        standar_template = f'''从以下药物信息中总结字段，回答这句话需要使用到哪些字段，以及该药品的药品名，不用赘述其他：
        {input_data}
        字段列表：{', '.join(self.field_list)}
        
        请返回以下格式的结果：
        提到的药品名：药品名
        标准化输出：
        字段名
        字段名
        例如：
        提到的药品名：八角茴香
        标准化输出：
        功能主治
        性状
        '''
        response = self.llm.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{'role':'user', 'content':standar_template}]
        )

        standardized_output = response.choices[0].message.content.strip() if response.choices else "无输出"

        return standardized_output

    
def connect_elasticresearch(): #连接es服务器
    hosts = [
        {'host':'127.0.0.1', 'port': 9200, 'scheme': 'http'
        }
    ]
    es = None
    for host in hosts:
        try:
            es = Elasticsearch(
                [host],
                basic_auth=('elastic', '7ztvwEMjr0H+_R4Vec*R'),
                verify_certs=False,  # 在开发时禁用 SSL 验证，生产环境中请谨慎使用
                # timeout=60
            )
            if es.ping():
                print(f'成功连接到 Elasticsearch: {host["host"]}')
                return es
            else:
                print(f'无法连接到 Elasticsearch: {host["host"]}')
        except exceptions.ConnectionError as e:
            print(f"连接错误：{e} - 尝试下一个主机")
    print('所有主机连接失败')
    return None

def classify_query_type(input_data):
    '''
    判断是否与药品或者医药学相关。
    return good or bad
    '''
    classify_template = f'''你是一名药学专家，请判断以下问题是否与药品或药学相关：
    {input_data}
    
    如果这是一个与药品或药学相关的问题（如提问药品的功能、用途、副作用等），返回 "good"；
    如果不是药学相关的问题，返回 "bad"。
    只能返回“good”或者“bad”
    '''

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [{
            "role": "user",
            "content" : classify_template
        }
        ]
    )

    query_type = response.choices[0].message.content.strip() if response.choices else "未知类型"

    print(query_type)
        
    # 确保只返回 "good" 或 "bad"
    if query_type not in ["good", "bad"]:
        return "未知类型"
    
    return query_type

def extract_drag_info(input_text):   #切分标准化输入
    
    #匹配标准化信息
    pattern = re.compile(r'提到的药品名：(.+?)\s+标准化输出：\s*(.+?)(?=(提到的药品名：|$))', re.DOTALL)
    matches = pattern.findall(input_text)

    drugs = []
    standard_outputs = []

    for match in matches:
        #提取多个药品名
        drugs_names = [name.strip() for name in re.split(r'[、，,]', match[0]) if name.strip()]
        #提取标准化输出的每一行
        outputs = [line.strip() for line in match[1].strip().split('\n') if line.strip()]
        
        for drug in drugs_names:
            drugs.append(drug)
            standard_outputs.append(outputs)

    return drugs, standard_outputs

def verify_data_in_es(es, es_index, doc_id, sub_title):

    output = []

    try:
        # 检查药品名（doc_id）
        response = es.get(index = es_index, id = doc_id) #查询是否存在该药品
        content = response['_source']['content']
        #提取所有小标题及其内容
        subsections = extra_subsections(content)

        found_content = []

        #验证问题中的字段是否存在指定药品的小标题中
        # for sub_title in sub_titles:  #[性状]
        if sub_title in subsections and subsections[sub_title]:
            found_content.append(f"小标题 '{sub_title}' 的内容:\n{subsections[sub_title]}")

        if found_content:
            output.append("\n".join(found_content))            #返回单个药品中的在数据库中存在字段的内容
        else:
            #    没有找到任何小标题，输出完整内容
            output.append(f"未找到任何小标题，输出完整内容:\n{content}")

        

    except exceptions.NotFoundError:
        output.append(f"文档 ID '{doc_id}' 或索引 '{es_index}' 不存在。")
    except exceptions.TransportError as e:
        output.append(f"查询错误：{e}")

    return "\n".join(output)


# classify_query_type("你是谁")
# es = connect_elasticresearch()
# content = retrival_data_from_es("板蓝根", es)
# print(content)
# standardized = MedicineStandardizer(client)
# answer = standardized.bzh("如何鉴别牛胆粉？")