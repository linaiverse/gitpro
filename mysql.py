import pymysql
import pandas as pd
from typing import List, Optional
import time
import argparse
import json
from sqlalchemy import create_engine

start = time.time()

# 创建ArgumentParser对象
parser = argparse.ArgumentParser(description='Retrieve data from source database and write to fixed remote target')

# 关键修改：将fields参数设为可选（nargs='?'），默认值设为None
parser.add_argument('table', type=str, help='Source and target MySQL table name (must be the same)')
parser.add_argument('fields', type=str, nargs='?', default=None,
                    help='Fields you want (as JSON list, e.g. \'["col1","col2"]\'); leave blank to query all fields')
parser.add_argument('--where', type=str, help='WHERE condition for SQL query', default='')

# 解析参数
args = parser.parse_args()

# 固定的目标数据库配置 - 在这里修改为实际的目标数据库信息
TARGET_CONFIG = {
    "host": "127.0.0.1",  # 目标数据库主机地址
    "port": 3306,          # 目标数据库端口
    "user": "root",        # 目标数据库用户名
    "password": "root",    # 目标数据库密码
    "database": "CaiKuData"# 目标数据库名称
}


# 连接源数据库
def get_source_db_connection():
    return pymysql.connect(
        host="10.128.11.183",
        user="senbatest",
        password="mdGBp7ck",
        database="senba_basedata",
        charset="utf8mb4"
    )


# 创建目标数据库的 SQLAlchemy 引擎（适配 pandas.to_sql）
def get_target_db_engine():
    # 构建 MySQL 连接字符串：mysql+pymysql://用户名:密码@主机:端口/数据库名
    conn_str = f"mysql+pymysql://{TARGET_CONFIG['user']}:{TARGET_CONFIG['password']}@{TARGET_CONFIG['host']}:{TARGET_CONFIG['port']}/{TARGET_CONFIG['database']}?charset=utf8mb4"
    # 创建并返回引擎
    return create_engine(conn_str)


# 获取目标数据库的 pymysql 连接（仅用于创建表、预览数据，不用于 to_sql）
def get_target_db_connection():
    return pymysql.connect(
        host=TARGET_CONFIG["host"],
        port=TARGET_CONFIG["port"],
        user=TARGET_CONFIG["user"],
        password=TARGET_CONFIG["password"],
        database=TARGET_CONFIG["database"],
        charset="utf8mb4"
    )


# 获取源表字段数据类型映射（适配“全部字段”逻辑）
def get_field_types(conn, source_table, fields: Optional[List[str]]):
    field_types = {}
    cursor = conn.cursor()

    # 查询字段类型（无论是否指定字段，先查全表结构）
    query = f"DESCRIBE {source_table}"
    cursor.execute(query)
    results = cursor.fetchall()

    for result in results:
        field_name = result[0]
        # 逻辑：如果指定了字段列表，只保留指定字段；否则保留全部字段
        if not fields or field_name in fields:
            field_type = result[1]
            field_types[field_name] = field_type

    cursor.close()
    return field_types


# 在目标数据库创建表（与源表同名同结构，支持全部字段）
def create_target_table(conn, target_table, field_types):
    cursor = conn.cursor()

    # 先检查表是否存在，如果存在则删除
    cursor.execute(f"DROP TABLE IF EXISTS {target_table}")

    # 构建创建表的SQL语句（字段与源表一致，指定字段则创建指定字段，否则创建全部字段）
    fields_definition = []
    for field, type_str in field_types.items():
        fields_definition.append(f"`{field}` {type_str}")

    create_sql = f"CREATE TABLE {target_table} ({', '.join(fields_definition)})"
    print(f"Creating target table with SQL: {create_sql}")

    cursor.execute(create_sql)
    conn.commit()
    cursor.close()


# 分批读取源表并写入目标表（支持“全部字段”查询）
def read_and_write_to_remote_table(table: str, fields: Optional[List[str]], where_condition: str = ''):
    # 连接源数据库、目标数据库、创建SQLAlchemy引擎
    source_conn = get_source_db_connection()
    target_conn = get_target_db_connection()
    target_engine = get_target_db_engine()

    try:
        # 获取字段类型（支持全部字段）
        field_types = get_field_types(source_conn, table, fields)
        if not field_types:
            print("No fields found, cannot create target table")
            return

        # 在目标数据库创建表（指定字段则创建指定字段，否则创建全部字段）
        create_target_table(target_conn, table, field_types)

        # 构建源表查询SQL（核心逻辑：指定字段则查指定字段，否则查全部字段）
        if fields:
            fields_str = ", ".join(fields)
        else:
            fields_str = "*"  # 不指定字段时，查询全部字段

        if where_condition:
            sql = f"SELECT {fields_str} FROM {table} WHERE {where_condition}"
        else:
            sql = f"SELECT {fields_str} FROM {table}"

        print(f"Executing source SQL: {sql}")

        # 用于记录总写入行数
        total_rows = 0

        # 分批读取并写入目标数据库
        for chunk in pd.read_sql(sql, source_conn, chunksize=50000):  # 每次读50000行
            chunk.to_sql(
                name=table,
                con=target_engine,
                if_exists='append',
                index=False,
                chunksize=10000
            )

            chunk_rows = len(chunk)
            total_rows += chunk_rows
            print(f"Inserted {chunk_rows} rows, total so far: {total_rows}")

        print(f"Successfully inserted {total_rows} records into remote table {table}")

        # 显示目标表前5行数据预览
        cursor = target_conn.cursor()
        cursor.execute(f"SELECT * FROM {table} LIMIT 5")
        preview = cursor.fetchall()
        print("Remote table data preview:")
        for row in preview:
            print(row)
        cursor.close()

    except Exception as e:
        target_conn.rollback()
        print(f"Error during process: {e}")
    finally:
        # 关闭所有连接
        source_conn.close()
        target_conn.close()
        target_engine.dispose()


# 解析参数：fields为None时设为空列表（表示查询全部字段）
try:
    if args.fields is None:
        fields = None  # 无输入时，标记为“查询全部字段”
    else:
        # 有输入时，解析为JSON列表
        fields = json.loads(args.fields.replace("'", '"'))
        # 额外校验：确保解析后是列表类型
        if not isinstance(fields, list):
            raise ValueError("Fields must be a JSON list")
except (json.JSONDecodeError, ValueError) as e:
    print(f"Error parsing fields argument: {e}")
    print("Usage 1 (query all fields): python3 mysql.py Sys_Accident")
    print("Usage 2 (query specific fields): python3 mysql.py Sys_Accident '[\"AccidentTime\",\"VehicleID\"]'")
    exit(1)

# 打印参数信息（区分“全部字段”和“指定字段”）
print(f"Table name (source and target): {args.table}")
print(f"Target database: {TARGET_CONFIG['host']}:{TARGET_CONFIG['port']}/{TARGET_CONFIG['database']}")
print(f"Fields: {'All fields' if fields is None else fields}")
print(f"Where condition: {args.where if args.where else 'None'}")

# 执行查询并写入远程表
read_and_write_to_remote_table(args.table, fields, args.where)

end = time.time()
print(f"Execution time: {end - start} seconds")