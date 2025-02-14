from openai import OpenAI
import gradio as gr
from embed import MedicineStandardizer,classify_query_type,connect_elasticresearch,extract_drag_info,verify_data_in_es
from vector_index import retrieve_vetcor
import time
from upload import UploadDoc
client = OpenAI(
    api_key="sk-KbwhEmvv5gIqt6W6FPGBgmco4Rd9QZsCj1yZwTkYQnGJhHIj",
    base_url="https://api.chatanywhere.tech/v1"
)


es = connect_elasticresearch()

# history = []
def update_config(es_host, es_port, es_user, es_pass, es_index, vector_db):# 配置页面的更新函数, 用于更新配置信息应用到全局
    """
    更新配置信息

    参数:
    es_host (str): Elasticsearch 主机地址
    es_port (str): Elasticsearch 端口号
    es_user (str): Elasticsearch 用户名
    es_pass (str): Elasticsearch 密码
    es_index (str): Elasticsearch 索引名称
    vector_db (str): 向量数据库路径

    返回:
    str: 配置更新成功的消息
    """
    global config, vector_db_path,es_indx
    config = {
        'es_host': es_host,
        'es_port': es_port,
        'es_user': es_user,
        'es_pass': es_pass,
        'es_index': es_index
    }
    es_indx=es_index
    vector_db_path = vector_db  # 更新全局变量
    return "配置已更新!"

def LLM_q(question):
    template = f'''你是一个药典问答机器人，回答以下的问题：
    {question}
    
    请给出具体回答。
    '''
    try:
        response = client.chat.completions.create(
            model = "gpt-4o-mini",
            messages = [{"role":"user",
            "content":template}],
            stream = False
        )
        answer = response.choices[0].message.content.strip() if response.choices else "无法提供答案"
        return answer
    except Exception as e:
        print(f"API调用失败:{e}")
        return "无法提供答案"


def ask_question(message, history):
    input_data = message
    standardizer = MedicineStandardizer(client)
    # history.append((message, ""))

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
                    result = verify_data_in_es(es, es_indx, id, sub)
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

        context = "\n".join([f"用户: {msg}\n助手: {resp}" for msg, resp in history])
        final_query = f'''这里是用户与助手的历史对话记录:{context}
        这里是你要回答的问题：{input_data}
        可以借助检索到的数据回答：'{final_result}'.不要瞎编，涉及的敏感词请替换成同义词。可以润色内容，使其贴切或易懂,只回答我提的问题，问题没问的哪怕给了信息，也不用回答,可以结合用户和助手的对话内容，不要重复回答。
        '''
        print(f"LLM接受到的prompt:{final_query}")
        statement = "小助手正在借助数据库思考，请稍后。。。"
        for i in range(len(statement)):
            time.sleep(0.01)
            yield statement[:i + 1]# 逐步返回小助手的思考过程

        final_answer = LLM_q(final_query)
    elif query_type == "bad":
        #直接将用户内容输入LLM
        context = "\n".join([f"用户: {msg}\n助手: {resp}" for msg, resp in history])
        final_query = f"这里是上下文：{context}\n\n这是你要回答的问题{input_data}.不要瞎编，可以润色内容，使其贴切或易懂,只回答我提的问题，问题没问的哪怕给了信息，也不用回答,可以结合用户和助手的对话内容，不要重复回答。"
        final_answer = LLM_q(final_query)

    else:
        final_answer = "无法判断输入问题类型"

    yield "\n"
    for i in range(len(final_answer)):
        time.sleep(0.03)  # Adjust the delay as needed
        yield final_answer[:i + 1]
    # history[-1] = (message, final_answer.strip())

def import_new_doc(upload_file, index_name, vector_index_path):
    global es
    if upload_file is not None:
        file_input = upload_file.name #获取上传文件的路径
        # 检查配置是否已设置
        if not config or 'es_host' not in config:
            return "请先更新配置!"
        
        uploader = UploadDoc(file_input=file_input)
        uploader.upload_doc(index_name, vector_index_path, es)
        
        print("上传文件成功！")

    else:
        print("没有上传文件")

#利用gr开启前端
if __name__ == "__main__":
    with gr.Blocks() as demo:  # 创建一个 Gradio 界面
        with gr.Tab("聊天机器人"):
            #创建chatbot实例
            gr.Markdown("**第一次使用请先配置信息并保存")
            chatbot = gr.Chatbot(height=600)
            qa_interface = gr.ChatInterface(
                fn = ask_question,
                chatbot = chatbot
            )
        with gr.Tab("导入数据库文档"):
            folder_input = gr.Textbox(label="保存向量数据库的文件夹路径", value=r"/home/fengw/rag/data")
            upload_interface = gr.Interface(
                fn = import_new_doc,
                inputs=[
                    gr.File(label = "上传文档"),
                    gr.Textbox(label="数据库与es的索引名称",placeholder="请输入"),
                    folder_input
                ],
                outputs=gr.Textbox(label = "上传结果"),
                title="上传文档",
                description="上传新的文档以更新数据库"
            )
        with gr.Tab("配置界面"):
            es_host_input = gr.Textbox(label="ES主机地址", value='localhost')
            es_port_input = gr.Textbox(label="ES服务端口", value='9200')
            es_user_input = gr.Textbox(label="ES用户名", value='elastic')
            es_pass_input = gr.Textbox(label="ES密码", type="password", value='7ztvwEMjr0H+_R4Vec*R')
            
            es_index_input = gr.Textbox(label="ES索引名（上传的会和向量数据库同名）", value='zhbd')  # 新增索引名输入框
            vector_db_path_input = gr.Textbox(label="向量数据库位置", value='/home/fengw/rag/data/zhbd.npz')  # 新增向量数据库位置输入框

            config_submit = gr.Button("保存配置")
            config_message = gr.Textbox(label="状态", interactive=False)

            config_submit.click(
                fn=update_config,
                inputs=[es_host_input, es_port_input, es_user_input, es_pass_input, es_index_input, vector_db_path_input],
                outputs=config_message
            )

            
    demo.launch()  # 运行 Web UI