import sqlite3
from typing import Union, Literal, Union, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from catenaconf import Catenaconf

from ..catena_core.paths import llmconf as LF
from ..catena_core.utils.format_builtin import delete_list_format
from ..catena_core.node.base import Node
from ..settings import RTConfig


import logging
#from src import loginit
logger = logging.getLogger(__name__)

class Preset(BaseModel):
    db: Union[str, List[str]] = Field(default="default_database", description="将要检索的数据库或数据库列表")
    limit: int = Field(default=3, description="检索结果的数量限制")
    seperator: str = Field(default="\n\n", description="检索结果的分隔符")

class InputRetrieve(BaseModel):
    setter: Literal["Vector", "Keyword"] = Field(default="Vector", description="检索器类型")
    args: Optional[dict] = Field(default=Preset, description="检索器参数")

class VectorRetriever:
    
    def __init__(self):
        from chromadb import PersistentClient
        from sentence_transformers import SentenceTransformer
        import torchvision
        torchvision.disable_beta_transforms_warning()

        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
        self.client = PersistentClient(path=LF.DB_PATH)
        self.preset = Preset()

    def __enter__(self):
        # 返回自身以便在 with 语句中使用
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 在退出时清理资源
        self.close()

    def query(self, query: str, **kwargs):
        query_embedding = [self.model.encode(query).tolist()]
        all_results = []

        db = kwargs.get('db', None) or self.preset.db
        limit = kwargs.get('limit', None) or self.preset.limit
        where = kwargs.get('where', None) or self.preset.where
        seperator = kwargs.get("seperator", None) or self.preset.seperator

        # 如果 db 是一个列表，则遍历每个 collection，进行查询
        dbs = db if isinstance(db, list) else [db]

        for db in dbs:
            collection = self.client.get_collection(db)
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=limit,
                where=where
            )

            # 如果有结果，将其加入总结果列表
            if results and results['documents'][0]:  # 检查是否有匹配到的结果
                for i in range(len(results['documents'][0])):
                    all_results.append({
                        'node_id': results['ids'][0][i],
                        'distance': results['distances'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'document': results['documents'][0][i]
                    })

        # 对所有匹配的结果按距离重新排序
        sorted_results = sorted(all_results, key=lambda x: x['distance'])

        # 取出匹配度最高的前 limit 个结果
        top_results = sorted_results[:self.filter.limit]

        # 提取文档并解析
        documents = [result['document'] for result in top_results]
        return self.parse(documents, seperator)

    def parse(self, results: List[str], seperator: str = None):
        if seperator:
            return delete_list_format(results, self.filter.seperator)
        else:
            return results

    def create(self, db: str):
        self.collection = self.client.create_collection(db)
   
    def close(self):
        import torch
        torch.cuda.empty_cache()
        del self.model

class KeywordRetriever:
    
    def __init__(self) -> None:
        self.conn = sqlite3.connect(LF.DB_PATH_.val("wiki.db"))
        self.preset = Preset()
        logger.info("数据库连接成功。当前数据库类型：WikiRetriever")

    def __enter__(self):
        # 返回自身以便在 with 语句中使用
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 在退出时清理资源
        self.close()

    def select(self, condition: Dict=None, **kwargs) -> List:
        """
        根据给定的查询条件，从数据库中查询数据。
        
        Args:
            condition (Dict, optional): 查询条件字典，键为字段名，值为字段值。默认值为None。
            **kwargs: 其他查询条件，键为字段名，值为字段值。
        
        Returns:
            List[sqlite3.Row]: 查询结果列表，其中每个元素为一个sqlite3.Row对象，代表一条记录。
        
        Raises:
            AssertionError: 当未提供任何查询条件时触发。
        
        """
        logger.info("正在搜索数据...")

        query = "SELECT * FROM main WHERE 1=1"
        params = []

        if condition:
            for key, value in condition.items():
                query += f" AND {key} = ? "
                params.append(value)
        else:
            assert kwargs, "No query or kwargs provided"
            for key, value in kwargs.items():
                query += f" AND {key} = ? "
                params.append(value)

        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        result = cursor.fetchall()

        return result

    def parse(
        self, fetched: List, outformat: str="s", selection: Union[List, str]=None
    ):
        """
        解析获取的数据，返回特定格式的数据。
        
        Args:
            fetched (List): 从数据库或其他数据源获取的数据列表，每个元素为字典类型。
            outformat (str, optional): 输出数据的格式。默认为 "s"。可选值包括：
                - "ld": 返回一个字典列表，列表中的每个元素为一个字典，包含从 fetched 中选取的字段。
                - "ls": 返回一个字符串列表，列表中的每个元素为从 fetched 中选取的字段 "content" 的值。
                - "s": 返回一个字符串，其中包含了从 fetched 中选取的字段 "content" 的值，并且每两个值之间用两个换行符分隔。
            selection (Union[List, str], optional): 要从 fetched 中选取的字段列表或单个字段名。默认为 None，表示选取 fetched 中的所有字段。
        
        Returns:
            Union[List[Dict], List[str], str]: 返回一个特定格式的数据，具体类型取决于 outformat 参数的值。
        
        Raises:
            ValueError: 如果 outformat 参数的值不是 "ld"、"ls" 或 "s" 之一，则引发此异常。
        """
        if selection:
            if isinstance(selection, list):
                selected = [{k: row[k] for k in selection} for row in fetched]
            else:
                selected = [row for row in fetched]
        else:
            selected = fetched

        if outformat == 'ld':   # list of dict
            result = selected
        elif outformat == "ls":  #list of string
            result = [row["content"] for row in selected]
        elif outformat == "s":  #string
            result = "\n\n".join([row["content"] for row in selected])
        else:
            raise ValueError(f"Unsupported output format: {outformat}")

        return result

    def query(self, query, **kwargs):
        where = kwargs.get("where", None) or self.preset.where
        selected = self.select(query)
        #print("SELECTED:", selected)
        selection = self.parse(selected, selection=where)

        return selection
    
    def close(self):
        self.conn.close()
        logger.info("数据库连接已关闭")

class Retriever:
    """  """
    def __new__(cls, setter: Literal["Vector", "Keyword"]):
        class_name = setter + "Retriever"
        if class_name in globals() and isinstance(globals()[class_name], type):
            return super().__new__(globals()[class_name])
        raise ValueError(f"Class {class_name} does not exist.")

class SmartRetriever(Node):
    
    def __init__(self):
        pass


    def retrieve(self, query: Union[str, dict, List], **kwargs):
        pass

    

    def operate(self, input, config: RTConfig = None, *args, **kwargs):
        return self.retrieve(input, **kwargs)
    


        



   

if __name__ == "__main__":
    pass