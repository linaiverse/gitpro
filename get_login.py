import pandas as pd
from pymongo import MongoClient
from typing import List, Optional, Dict, Any
import time
import argparse
import json

start = time.time()

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='Retrieve data from the database')

# 修改参数定义 - 使用字符串类型，稍后解析为列表
parser.add_argument('collections', type=str, help='collections of your want (as JSON list)')
parser.add_argument('fields', type=str, help='fields of your want (as JSON list)')
parser.add_argument('csvname', type=str, help='name of your dumped file')
parser.add_argument('--filter', type=str, default='{}', help='filter condition (as JSON object)')

# 解析参数
args = parser.parse_args()

def batch_merge_collections(
    db_name: str,
    collection_names: List[str],
    batch_size: int = 5000,
    fields: Optional[List[str]] = None,
    filter_query: Optional[Dict[str, Any]] = None  
) -> List[dict]:
    # 连接 MongoDB
    client = MongoClient(
        host='10.128.11.182',
        port=27017,
        username='senbatest',
        password='mdGBp7ck',
        authSource='admin',
        authMechanism='SCRAM-SHA-1'
    )
    
    db = client[db_name]
    merged_docs = []
    # 获取当前数据库中所有存在的集合名称（用于后续检查）
    existing_collections = db.list_collection_names()
    
    for col_name in collection_names:
        # 核心修改：检查集合是否存在，不存在则跳过
        if col_name not in existing_collections:
            print(f"⚠️  Collection '{col_name}' does NOT exist, skipping...")
            continue
        
        print(f"✅  Processing collection: {col_name}")
        
        # 构建投影（字段选择规则）
        projection = {"_id": 0}  # 默认不包含_id（避免MongoDB默认的ObjectId干扰）
        if fields is not None and len(fields) > 0:
            for field in fields:
                projection[field] = 1  # 仅保留指定字段

        # 使用投影+过滤条件查询
        cursor = db[col_name].find(
            filter=filter_query if (filter_query and filter_query != {}) else {},
            projection=projection,
            batch_size=batch_size
        )
        
        # 流式处理文档（避免一次性加载大量数据到内存）
        count = 0
        for doc in cursor:
            merged_docs.append(doc)
            count += 1
        
        print(f"📊  Retrieved {count} documents from '{col_name}'")

    # 关闭MongoDB连接（避免资源泄漏）
    client.close()
    return merged_docs

# 解析参数为实际列表/字典（处理单引号转双引号，适配JSON格式）
try:
    # 将命令行传入的单引号替换为双引号，确保JSON解析成功
    collections = json.loads(args.collections.replace("'", '"'))
    fields = json.loads(args.fields.replace("'", '"'))
    filter_query = json.loads(args.filter.replace("'", '"'))
    # 校验collections是否为列表（避免参数格式错误）
    if not isinstance(collections, list):
        raise ValueError("'collections' must be a JSON list (e.g., [\"col1\",\"col2\"])")
    if not isinstance(fields, list):
        raise ValueError("'fields' must be a JSON list (e.g., [\"field1\",\"field2\"])")
except json.JSONDecodeError as e:
    print(f"❌  Error parsing arguments: Invalid JSON format - {e}")
    print("📝  Example usage: python3 getMongo4.0.py '[\"col1\",\"col2\"]' '[\"field1\",\"field2\"]' 'output.csv'")
    exit(1)
except ValueError as e:
    print(f"❌  Argument format error: {e}")
    exit(1)

# 打印参数确认信息
print(f"\n🔧  Configuration Summary:")
print(f"    Collections to process: {collections}")
print(f"    Fields to retrieve: {fields}")
print(f"    Filter condition: {filter_query if filter_query != {} else 'No filter'}")
print(f"    Output CSV path: {args.csvname}\n")

# 执行MongoDB查询
result = batch_merge_collections(
    db_name="senba_gps",
    collection_names=collections,
    batch_size=50000,  # 大批次减少I/O次数，提升效率
    fields=fields,
    filter_query=filter_query
)

# 处理查询结果并保存到CSV
if result and len(result) > 0:
    df = pd.DataFrame(result)
    print(f"\n📈  Total retrieved records: {len(df)}")
    print(f"🔍  Sample data (first 5 rows):")
    print(df.head(5))
    
    # 保存CSV（index=False避免多余的索引列）
    df.to_csv(args.csvname, index=False, encoding='utf-8-sig')  # utf-8-sig支持中文显示
    print(f"\n💾  Data successfully saved to: {args.csvname}")
else:
    print(f"\n⚠️  No valid data retrieved from MongoDB (all collections may be empty or non-existent)")

# 计算并打印总执行时间
end = time.time()
print(f"\n⏱️  Total execution time: {round(end - start, 2)} seconds")