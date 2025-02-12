from openai import OpenAI
import gradio as gr
from embed import MedicineStandardizer,classify_query_type,connect_elasticresearch,extract_drag_info,verify_data_in_es
from vector_index import retrieve_vetcor

client = OpenAI(
    api_key="sk-KbwhEmvv5gIqt6W6FPGBgmco4Rd9QZsCj1yZwTkYQnGJhHIj",
    base_url="https://api.chatanywhere.tech/v1"
)


es = connect_elasticresearch()

def ask_question(message, history):
    input_data = message
    standardizer = MedicineStandardizer(client)
    history.append((message, ""))

    #判断是否与药学相关
    query_type = classify_query_type(input_data)

    if query_type == "good":

        standard_input = standardizer.standardize_information(input_data)# 标准化用户输入的问题 (药名：板蓝根，标准化输出：字段\n字段)

        if es: #首先利用es检索
            input_text = standard_input
            print("------------es检索产物------------")
            print(f"标准化输入: {input_text}")
            doc_id, sub_title = extract_drag_info(input_text)
            print(f"提取到的药品名：{doc_id}, 提取到的小标题: {sub_title}")
            
            combined_results = []

            for id, outputs in zip(doc_id, sub_title):   #一个药品名-多个标准化输出
                print(f"当前ID: {id} ，当前输出: {outputs}")
                for sub in outputs:
                    print(f"检索的子标题{sub}")
                    result = verify_data_in_es(es, es_index, id, sub)
                    print(f"检索结果：{result}")
                    combined_results.append(result)
            
            unique_result = list(set(combined_results))
            final_result = "\n\n".join(unique_result) + "\n"
            print(f"es的检索结果: {final_result}")      #似乎不带药品名字，但不确定

        else:
            final_result = ""

        #使用faiss中的内容，除去药物和小标题进行标准化输入 
        llm_q = standardizer.bzh(input_data)
        output_lines = []
        for line in llm_q.splitlines():
            results = retrieve_vetcor(line.strip(), vector_db_path, top_k = 1)
            for doc_id, title, text in results:
                output_lines.append(f"Document ID: {doc_id}, Titile: {title}, Text:{text}")
        
        output_result = "\n".join(output_lines) #faiss向量化检索的结果
        final_result += output_result + "\n" #es和faiss向量库同时检索
        
    # pass