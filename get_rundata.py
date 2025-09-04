import pandas as pd
import numpy as np
from pymongo import MongoClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional, Dict, Any
import time

# 数据库连接配置
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
    """将DataFrame写入数据库，清空表后写入"""
    if df.empty:
        print(f"警告：{table_name}数据为空，跳过写入")
        return False
    
    try:
        # 检查并清空表
        with engine.connect() as conn:
            table_exist_sql = text(f"""
                SELECT COUNT(*) 
                FROM INFORMATION_SCHEMA.TABLES 
                WHERE TABLE_SCHEMA = '{DB_CONFIG['database']}' 
                AND TABLE_NAME = '{table_name}'
            """)
            table_exist = conn.execute(table_exist_sql).scalar()
            
            if table_exist > 0:
                conn.execute(text(f"TRUNCATE TABLE {table_name}"))
                conn.commit()
                print(f"表{table_name}已存在，已清空原有数据")
            else:
                print(f"表{table_name}不存在，将创建新表")
        
        # 写入数据
        df.to_sql(
            name=table_name,
            con=engine,
            if_exists='append',
            index=False,
            chunksize=10000
        )
        print(f"成功写入{len(df)}条记录到表{table_name}")
        return True
    
    except SQLAlchemyError as e:
        print(f"写入表{table_name}失败：{str(e)}")
        return False

def get_VehicleRunDataDays(
    db_name: str, 
    collection_names: List[str], 
    batch_size: int = 5000,
    fields: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    使用MongoDB聚合管道获取司机车辆当日主动安全报警类型次数
    并返回处理后的DataFrame
    """
    # MongoDB连接配置
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
    all_results = []
    
    # 构建聚合管道
    def build_aggregation_pipeline():
        pipeline = [
            # 1. 转换时间戳为日期
            {
                "$addFields": {
                    "AddDateTime": {
                        "$toDate": {"$multiply": ["$Addtime", 1000]}  # 假设Addtime是秒级时间戳
                    },
                    # 计算衍生字段
                    "NegativeReview": {
                        "$cond": {
                            "if": {"$in": ["$judge", [3, 4]]},
                            "then": 1,
                            "else": 0
                        }
                    },
                    "CarryPassengerNum": {
                        "$cond": {
                            "if": {"$gt": ["$passengerCount", 0]},
                            "then": 1,
                            "else": 0
                        }
                    },
                    # 确保passengerTime为整数
                    "passengerTimeInt": {"$toInt": "$passengerTime"}
                }
            },
            # 2. 提取日期部分
            {
                "$addFields": {
                    "AddDate": {"$dateTrunc": {"date": "$AddDateTime", "unit": "day"}}
                }
            },
            # 3. 按分组字段聚合
            {
                "$group": {
                    "_id": {
                        "certificateId": "$certificateId",
                        "AddDate": "$AddDate",
                        "carNo": "$carNo"
                    },
                    "freeMileage": {"$sum": "$freeMileage"},
                    "passengerTime": {"$sum": "$passengerTimeInt"},
                    "waitTime": {"$sum": "$waitTime"},
                    "mileage": {"$sum": "$mileage"},
                    "money": {"$sum": "$money"},
                    "NegativeReview": {"$sum": "$NegativeReview"},
                    "CarryPassengerNum": {"$sum": "$CarryPassengerNum"}
                }
            },
            # 4. 重塑输出格式
            {
                "$project": {
                    "certificateId": "$_id.certificateId",
                    "AddDate": "$_id.AddDate",
                    "carNo": "$_id.carNo",
                    "freeMileage": 1,
                    "passengerTime": 1,
                    "waitTime": 1,
                    "mileage": 1,
                    "money": 1,
                    "NegativeReview": 1,
                    "CarryPassengerNum": 1,
                    "_id": 0
                }
            },
            # 5. 排序
            {
                "$sort": {"certificateId": 1, "AddDate": 1}
            }
        ]
        return pipeline
    
    # 获取聚合管道
    pipeline = build_aggregation_pipeline()
    
    # 处理每个集合
    for col_name in collection_names:
        # 检查集合是否存在
        if col_name not in db.list_collection_names():
            print(f"集合 {col_name} 不存在，跳过处理")
            continue
            
        print(f"正在处理集合: {col_name}")
        try:
            # 执行聚合查询
            cursor = db[col_name].aggregate(
                pipeline,
                batchSize=batch_size,
                allowDiskUse=True  # 大数据量时允许使用磁盘
            )
            
            # 将结果转换为DataFrame并添加到列表
            df = pd.DataFrame(list(cursor))
            if not df.empty:
                all_results.append(df)
                print(f"处理完成 {col_name}，获取 {len(df)} 条记录")
                
        except Exception as e:
            print(f"处理集合 {col_name} 时出错: {str(e)}")
            continue
    
    # 合并所有结果
    if all_results:
        result = pd.concat(all_results, ignore_index=True)
        # 转换日期格式为字符串
        if 'AddDate' in result.columns:
            result['AddDate'] = pd.to_datetime(result['AddDate']).dt.date
        print(f"所有集合处理完成，共获取 {len(result)} 条记录")
    else:
        result = pd.DataFrame()
        print("没有获取到任何数据")
    
    # 关闭连接
    client.close()
    return result

# 执行主流程
if __name__ == "__main__":
    start_time = time.time()
    
    # 配置参数 - 根据实际情况修改
    DB_NAME = "senba_gps"
    # 集合列表 - 根据实际需求修改
    VehicleRunData_li = ["VehicleRunData202508","VehicleRunData202509","VehicleRunData202510","VehicleRunData202511","VehicleRunData202512","VehicleRunData202601","VehicleRunData202602"]  # 示例：8月1日至22日
    BATCH_SIZE = 10000
    FIELDS = ["Addtime", "certificateId", "carNo", "freeMileage", 
              "passengerTime", "waitTime", "mileage", "money", "judge", "passengerCount"]
    
    # 获取处理后的数据
    VehicleRunDataDays = get_VehicleRunDataDays(
        db_name=DB_NAME,
        collection_names=VehicleRunData_li,
        batch_size=BATCH_SIZE,
        fields=FIELDS
    )
    
    # 写入MySQL数据库
    if not VehicleRunDataDays.empty:
        engine = create_db_engine()
        write_to_database(VehicleRunDataDays, "vehicle_run_data_daily", engine)
        engine.dispose()
        # 打印前5条数据预览
        print("\n数据预览:")
        print(VehicleRunDataDays.head())
    else:
        print("没有数据可写入数据库")
    
    # 输出执行时间
    end_time = time.time()
    print(f"\n总执行时间: {round(end_time - start_time, 2)} 秒")
