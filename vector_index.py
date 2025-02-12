
from elasticsearch import Elasticsearch, exceptions
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re


model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("加载sentence模型完成!")

def connect_elasticsearch():# 连接Elasticsearch
    hosts = [
        # {'host': '192.168.110.28', 'port': 9200, 'scheme': 'https'},
        # {'host': '192.168.1.230', 'port': 9200, 'scheme': 'https'},
        {'host': '127.0.0.1', 'port': 9200, 'scheme': 'http'}
    ]
    es = None
    for host in hosts:
        try:
            es = Elasticsearch(
                [host],
                basic_auth=('elastic', '7ztvwEMjr0H+_R4Vec*R'),
                verify_certs=False  # 在开发时禁用 SSL 验证，生产环境中请谨慎使用
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

def load_faiss_index(vector_index_path):
    data = np.load(vector_index_path)
    embeddings = data["embeddings"]
    ids = data['ids']
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

def retrival_data_from_es(index_name, es):
    res = es.search(index = index_name, body = {"query": {"match_all": {}}, "size":10000})
    return res['hits']['hits']   #返回一个列表，每个元素为一个字典，代表着一个小标题

def extra_subsections(content):  #提取大标题中的小标题内容（一个大段）
    
    pattern = re.compile(r'(?:【|t)(.+?)(?:】)')  #[ ]或者t ]里面的内容
    matches = pattern.finditer(content)
    
    last_title = None
    last_position = 0
    subsections = {}

    for match in matches:
        title = match.group().strip('【】t]')  #去除这些字符
        if last_title:
            subsections[last_title] = content[last_position:match.start()].strip()   #形状:xxx, 功能:xxx

        last_title = title
        last_position = match.end()

    if last_title:
        subsections[last_title] = content[last_position:match.start()].strip()

    return subsections

def process_vector_index(index_name, vector_index_path, es):
    if os.path.exists(vector_index_path):
        print(f"从磁盘载入faiss向量库")
        load_faiss_index(vector_index_path)
        return
    
    #从es处理向量化数据到faiss
    entries = retrival_data_from_es(index_name, es)
    subsection_list = []
    id_to_content = {}
    ids = []
    texts = []
    #遍历内容
    for entry in entries:
        doc_id = entry['_id']  #文档的ID，其实就是title    ```板蓝根```
        content = entry['_source']['content']  #文档大段的内容 
        subsection = extra_subsections(content) 

        #id与全文转移到id_to_content中
        id_to_content[doc_id] = content
        print(f"Document ID: {doc_id} - Content Length: {len(content)}")  # 打印文档ID和内容长度

        for title, text in subsection.items():
            subsection_list.append((doc_id, title, text)) #药品名，小标题，内容
            ids.append(doc_id)
            texts.append((title, text)) #存储小标题，内容

    #向量化内容
    embeddings = model.encode([text for _,_,text in subsection_list], convert_to_numpy=True)

    #通过faiss创建索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    #保存faiss向量到指定文件夹
    np.savez_compressed(vector_index_path, embeddings=embeddings, ids=ids, texts=texts)   #保存为字典形式了
    print("Data processed and saved to disk.")
    

def retrieve_vetcor(input_data, embedding_file_path, top_k = 1):
    if not os.path.exists(embedding_file_path):
        raise FileNotFoundError(f"Embedding file not found at: {embedding_file_path}")
    
    query_embedding = model.encode([input_data], convert_to_numpy=True)
    #加载faiss向量
    data = np.load(embedding_file_path, allow_pickle=True)   #为字典 allow_pickle允许访问字典
    embeddings = data['embeddings']
    ids = data['ids']
    texts = data['texts']
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))

    #检索相关向量
    Distance, I = index.search(query_embedding.astype(np.float32), top_k)

    #获取检索的向量与ID
    retrieve_ids = ids[I[0]].tolist()   #I[0]指的是第一个查询的检索到的内容
    print(f"检索到的ID: {retrieve_ids}")

    retrieve_texts = [texts[i] for i in I[0]]

    #组合
    results = [(retrieve_ids[i], retrieve_texts[i][0], retrieve_texts[i][1]) for i in range(top_k)]

    return results