import re

import os
import docx
from elasticsearch import exceptions
from vector_index import process_vector_index


global config

class UploadDoc:
    def __init__(self, file_input) -> None:
        self.file_input = file_input

    def split_index_doc(self, index_name, es):
        if not os.path.exists(self.file_input):
            print(f"文件{self.file_input}不存在!")
            return
    
        try:
            doc_obj = docx.Document(self.file_input)
            try:
                content_dict = self.extract_titles(doc_obj)
            except Exception as e:
                print(f"提取标题时出错:{e}")

            self.store_in_elasticresarch(content_dict, index_name, es)

            print(f"已将{len(content_dict)}个篇章存入Elasticresearch!")

        except Exception as e:
            print(f"处理文件时出错:{e}")

        
    def store_in_elasticresarch(self, content_dict, index_name, es):
        # print(f"文件内容")
        for title, content in content_dict.items():
            try:
                es.index(index = index_name, id = title, body = {"content": '\n'.join(content)})
                print("存储成功")
            except exceptions.ConnectionError as e:
                print(f"连接错误：{e}")
            except exceptions.TransportError as e:
                print(f"存储错误：{e}")
        

    def clean_title(self, title):
        '''
        清除中文内容以外的字符串
        '''
        return "".join(re.findall(r'[\u4e00-\u9fff]+', title)).rstrip() #去除末尾空格

    def extract_titles(self, doc_obj):
        content_dict = {}
        temp_content = []
        for paragraph in doc_obj.paragraphs:
            if not paragraph.runs:
                continue
            font_size = paragraph.runs[0].font.size
            if font_size is not None:
                print(f"Font size: {font_size.pt}, Type: {type(font_size.pt)}")

                #保证大小为数字
                if isinstance(font_size.pt, (int, float)):
                    if font_size.pt == 12:
                        if temp_content:
                            title = self.clean_title(temp_content[0])
                            if title:
                                content_dict[title] = temp_content
                            temp_content = []
                else:
                    print(f"Unexpected font size type: {type(font_size.pt)}")
            print(f"加入文本: {paragraph.text}")
            temp_content.append(paragraph.text)             

        #最后一段
        if temp_content:
            title = self.clean_title(temp_content[0])
            if title:
                content_dict[title] = temp_content
            temp_content = []
        return content_dict

    def upload_doc(self, index_name, vector_index_path, es):
        self.split_index_doc(index_name, es)
        vector_index_path = os.path.join(vector_index_path, f"{index_name}.npz")
        process_vector_index(index_name, vector_index_path, es)


# from embed import connect_elasticresearch
# from elasticsearch import Elasticsearch
# es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
# if es.ping:
#     print("连接成功")
# import_new_doc("data/2020年药典一部.docx", "zhbd", "data", es)
