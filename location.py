import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Tuple
import time
import gc
import argparse
from datetime import datetime, timedelta

def generate_collection_names(days=8):
    """生成最近N天的集合名称，格式为 Locationyyyymmdd"""
    yesterday = datetime.now() - timedelta(days=1)
    return [
        f"Location{(yesterday - timedelta(days=i)):%Y%m%d}" 
        for i in range(days)
    ]
# 生成最近8天的集合名称
COLLECTION_NAMES = generate_collection_names(8)

# 打印结果
print("生成的集合名称列表:")
for name in COLLECTION_NAMES:
    print(name)

# 数据库连接配置 - 可根据实际环境修改
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "CaiKuData"
}

def create_db_engine():
    """创建数据库引擎"""
    connection_str = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
        f"{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}?charset=utf8mb4"
    )
    return create_engine(
        connection_str,
        pool_size=5,
        max_overflow=10,
        pool_recycle=3600
    )

def write_to_database(df: pd.DataFrame, table_name: str, engine) -> bool:
    """
    将DataFrame写入数据库
    
    参数:
        df: 要写入的数据
        table_name: 目标表名
        engine: 数据库引擎
        
    返回:
        写入成功返回True，否则返回False
    """
    if df.empty:
        print(f"警告：{table_name}数据为空，跳过写入")
        return False
    
    try:
        # 使用replace模式，存在表则替换，不存在则创建
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists='replace',
            index=False,
            chunksize=10000
        )
        print(f"✅ 成功写入{len(df)}条记录到表{table_name}")
        return True
    except SQLAlchemyError as e:
        print(f"❌ 写入表{table_name}失败：{str(e)}")
        return False

def get_location_data_simplified(
    db_name: str,
    collection_names: List[str],
    batch_size: int = 5000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    基于MongoDB聚合管道的位置数据处理（简化版）：
    1. 总里程：每日里程最大值；
    2. 速度统计：速度>5的日指标（sum/count/max/avg/std）；
    """
    # 1. MongoDB连接配置
    client = MongoClient(
        host='10.128.11.182',
        port=27017,
        username='senbatest',
        password='mdGBp7ck',
        authSource='admin',
        authMechanism='SCRAM-SHA-1',
        serverSelectionTimeoutMS=5000  # 连接超时控制
    )
    db = client[db_name]
    
    # 筛选存在的集合（避免操作无效集合）
    existing_cols = [col for col in collection_names if col in db.list_collection_names()]
    if not existing_cols:
        print("警告：未找到任何有效集合！")
        client.close()
        return pd.DataFrame(), pd.DataFrame()


    # -------------------------- 1. 总里程管道（每日里程最大值）--------------------------
    def build_total_mileage_pipeline(cols: List[str]) -> List[dict]:
        # 单个集合的核心逻辑：提取日里程最大值
        single_total_pipeline = [
            # 阶段1：转换秒级时间戳为“天”级日期（忽略时分秒）
            {"$addFields": {
                "gpsDate": {
                    "$dateTrunc": {
                        "date": {"$toDate": {"$multiply": ["$gpsTime", 1000]}},  # 秒→毫秒→Mongo日期
                        "unit": "day"
                    }
                }
            }},
            # 阶段2：按“日期+设备号”分组，取当日里程最大值（核心）
            {"$group": {
                "_id": {"gpsDate": "$gpsDate", "deviceNo": "$deviceNo"},
                "daily_max_mileage": {"$max": "$mileage"},  # 当日所有里程的最大值
                "daily_record_count": {"$sum": 1}  # 当日记录数（用于异常核对）
            }},
            # 阶段3：整理字段格式（提取分组键到顶层，删除临时_id）
            {"$project": {
                "gpsDate": "$_id.gpsDate",
                "deviceNo": "$_id.deviceNo",
                "daily_max_mileage": 1,
                "daily_record_count": 1,
                "_id": 0
            }},
            # 阶段4：排序（按设备号→日期升序，便于后续查看）
            {"$sort": {"deviceNo": 1, "gpsDate": 1}}
        ]
        
        # 多集合合并：用$unionWith拼接所有集合结果
        main_pipeline = []
        main_pipeline.extend(single_total_pipeline)  # 先处理第一个集合
        for col in cols[1:]:  # 合并剩余集合
            main_pipeline.append({
                "$unionWith": {"coll": col, "pipeline": single_total_pipeline}
            })
        
        return main_pipeline

    # 执行总里程聚合查询
    total_mileage_df = pd.DataFrame()
    try:
        total_pipeline = build_total_mileage_pipeline(existing_cols)
        total_cursor = db[existing_cols[0]].aggregate(
            pipeline=total_pipeline,
            batchSize=batch_size,
            allowDiskUse=True  # 大数据量时用磁盘临时存储，避免内存溢出
        )
        total_mileage_df = pd.DataFrame(list(total_cursor))
    except Exception as e:
        print(f"总里程管道执行失败：{str(e)}")


    # -------------------------- 2. 速度统计管道（速度>5的日指标）--------------------------
    def build_speed_stats_pipeline(cols: List[str]) -> List[dict]:
        # 单个集合的核心逻辑：筛选有效速度并计算日指标
        single_speed_pipeline = [
            # 阶段1：筛选有效数据（速度>5，且存在jobNumber，排除异常记录）
            {"$match": {
                "speed": {"$gt": 5},
                "jobNumber": {"$exists": True},  # 确保司机ID存在
                "jobNumber": {"$ne": ""},
                "jobNumber": {"$ne": None}       # 排除空值
            }},
            # 阶段2：转换时间戳为“天”级日期
            {"$addFields": {
                "gpsDate": {
                    "$dateTrunc": {
                        "date": {"$toDate": {"$multiply": ["$gpsTime", 1000]}},
                        "unit": "day"
                    }
                }
            }},
            # 阶段3：按“日期+司机ID”分组，计算速度指标
            {"$group": {
                "_id": {"gpsDate": "$gpsDate", "jobNumber": "$jobNumber"},
                "speed_sum": {"$sum": "$speed"},    # 当日速度总和
                "speed_count": {"$sum": 1},         # 当日有效速度记录数
                "speed_max": {"$max": "$speed"},    # 当日最大速度
                "speed_avg": {"$avg": "$speed"},    # 当日平均速度
                "speed_std": {"$stdDevPop": "$speed"}  # 当日速度标准差（总体）
            }},
            # 阶段4：整理字段格式
            {"$project": {
                "gpsDate": "$_id.gpsDate",
                "jobNumber": "$_id.jobNumber",
                "speed_sum": 1,
                "speed_count": 1,
                "speed_max": 1,
                "speed_avg": 1,
                "speed_std": 1,
                "_id": 0
            }},
            # 阶段5：排序（按司机ID→日期升序）
            {"$sort": {"jobNumber": 1, "gpsDate": 1}}
        ]
        
        # 多集合合并
        main_pipeline = []
        main_pipeline.extend(single_speed_pipeline)
        for col in cols[1:]:
            main_pipeline.append({
                "$unionWith": {"coll": col, "pipeline": single_speed_pipeline}
            })
        
        return main_pipeline

    # 执行速度统计聚合查询
    speed_stats_df = pd.DataFrame()
    try:
        speed_pipeline = build_speed_stats_pipeline(existing_cols)
        speed_cursor = db[existing_cols[0]].aggregate(
            pipeline=speed_pipeline,
            batchSize=batch_size,
            allowDiskUse=True
        )
        speed_stats_df = pd.DataFrame(list(speed_cursor))
    except Exception as e:
        print(f"速度统计管道执行失败：{str(e)}")


    # -------------------------- 数据后处理（统一格式+释放资源）--------------------------
    # 日期格式统一为"YYYY-MM-DD"字符串
    for df in [total_mileage_df, speed_stats_df]:
        if not df.empty and "gpsDate" in df.columns:
            df["gpsDate"] = pd.to_datetime(df["gpsDate"]).dt.strftime("%Y-%m-%d")
    
    # 释放内存与关闭连接
    gc.collect()
    client.close()


    return total_mileage_df, speed_stats_df


# -------------------------- 调用示例 --------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    # 1. 配置参数（按实际需求修改）
    DB_NAME = "senba_gps"
    # 集合列表：Location20250801 ~ Location20250822（共22天的位置数据）
    # COLLECTION_NAMES = [f"Location202508{i:02d}" for i in range(1, 23)]
    BATCH_SIZE = 10000  # 批次大小（内存充足时可增大，减少I/O次数）
    
    # 2. 执行查询（仅返回总里程和速度统计）
    total_mileage, speed_stats = get_location_data_simplified(
        db_name=DB_NAME,
        collection_names=COLLECTION_NAMES,
        batch_size=BATCH_SIZE
    )
    
    # 3. 创建数据库引擎
    engine = create_db_engine()
    
    # 4. 写入数据库
    if not total_mileage.empty:
        write_to_database(total_mileage, "daily_total_mileage", engine)
        print("总里程数据前5条预览：")
        print(total_mileage.head())
    else:
        print("❌ 未获取到总里程数据")
    
    if not speed_stats.empty:
        write_to_database(speed_stats, "daily_speed_stats", engine)
        print("\n速度统计数据前5条预览：")
        print(speed_stats.head())
    else:
        print("\n❌ 未获取到速度统计数据")
    
    # 5. 关闭数据库连接
    engine.dispose()
    
    # 6. 打印执行时间
    end_time = time.time()
    print(f"\n⏱️  总执行时间：{round(end_time - start_time, 2)} 秒")