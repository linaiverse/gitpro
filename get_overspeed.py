import pandas as pd
import pymongo
from typing import List, Optional
from pymongo import MongoClient
import time
import argparse

def get_adasAlarm_with_pipeline(
    db_name: str, 
    collection_names: List[str], 
    batch_size: int = 5000
) -> pd.DataFrame:
    """
    使用MongoDB聚合管道获取司机车辆当日主动安全报警类型次数（优化版）
    :param db_name: 数据库名
    :param collection_names: 需要查询的集合列表
    :param batch_size: 游标批次大小（控制内存占用）
    :return: 处理后的DataFrame
    """
    # 1. 连接MongoDB
    client = MongoClient(
        host='10.128.11.182',
        port=27017,
        username='senbatest',
        password='mdGBp7ck',
        authSource='admin',
        authMechanism='SCRAM-SHA-1'
    )
    db = client[db_name]
    
    # 2. 筛选存在的集合（避免$unionWith操作不存在的集合报错）
    existing_cols = [col for col in collection_names if col in db.list_collection_names()]
    if not existing_cols:
        print("警告：没有找到任何存在的集合！")
        return pd.DataFrame()
    
    # 3. 定义单个集合的核心聚合管道
    single_col_pipeline = [
        # 阶段1：筛选需要的字段（减少数据量）
        {"$project": {
            "alarmTime": 1,         # 报警时间戳（秒）
            "jobNumber": 1,         # 资格证ID
            "vehicleNumber": 1,     # 车牌号
            "speed": 1,             # 速度
            "handleStatus": 1,      # 处理状态
            "handleTime": 1,        # 处理时间
            "_id": 0                # 排除无用的_id
        }},
        
        # 阶段2：过滤有效报警（匹配原代码condition条件）
        {"$match": {
            "$and": [
                # 原逻辑：筛选有效报警（未处理或已处理）
                {"$or": [
                    {"$and": [{"handleStatus": 0}, {"handleTime": 0}]},  # 未处理（状态0+时间0）
                    {"handleStatus": 1}  # 已处理（状态1）
                ]},
                # 新增：筛选超速记录（速度>阈值，且排除speed为null/0的无效数据）
                {"speed": {"$lt": 250}},  # $gt = greater than（小于）
                {"speed": {"$ne": None}},  # 排除speed为null的记录
                {"speed": {"$gt": 20}}  # 排除speed为0的无效记录
            ]
        }},
        
        # 阶段3：转换时间戳为日期（匹配原代码alarmDate逻辑）
        {"$addFields": {
            "alarmDate": {
                "$dateTrunc": {
                    "date": {"$toDate": {"$multiply": ["$alarmTime", 1000]}},  # 秒转毫秒→转Date
                    "unit": "day"  # 截断到“天”级别（忽略时分秒）
                }
            }
        }},
        
        # 阶段4：按「日期+资格证ID+车牌号」分组，统计两类报警次数
        {"$group": {
            "_id": {
                "alarmDate": "$alarmDate",
                "jobNumber": "$jobNumber",
                "vehicleNumber": "$vehicleNumber"
            },

            "超速次数": {"$sum": 1}
        }},
        
        # 阶段5：整理字段格式（将_id中的关键字段提取到顶层，删除临时_id）
        {"$project": {
            "alarmDate": "$_id.alarmDate",
            "jobNumber": "$_id.jobNumber",
            "vehicleNumber": "$_id.vehicleNumber",
            "超速次数": 1,
            "_id": 0  # 排除临时分组键
        }}
    ]
    
    # 4. 动态生成多集合合并的管道（主集合+其他集合$unionWith）
    main_pipeline = []
    # 4.1 先处理第一个集合（主集合）
    main_pipeline.extend(single_col_pipeline)
    # 4.2 合并剩余集合（每个集合都执行相同的单集合管道）
    for col in existing_cols[1:]:
        main_pipeline.append({
            "$unionWith": {
                "coll": col,                # 要合并的集合名
                "pipeline": single_col_pipeline  # 该集合执行的管道
            }
        })
    
    # 5. 执行聚合查询（批量获取结果，避免内存溢出）
    cursor = db[existing_cols[0]].aggregate(
        pipeline=main_pipeline,
        batchSize=batch_size,
        allowDiskUse=True  # 数据量大时允许使用磁盘临时存储，避免内存不足
    )
    
    # 6. 转换结果为DataFrame（与原代码输出格式完全一致）
    result_df = pd.DataFrame(list(cursor))
    
    # 7. 关闭连接
    client.close()
    
    return result_df

# -------------------------- 调用示例 --------------------------
if __name__ == "__main__":
    start = time.time()
    # 请替换为你的集合列表（示例：AlarmInfo20250801 ~ AlarmInfo20250822）
    # AlarmInfo_li = [f"AlarmInfo202508{i:02d}" for i in range(1, 23)]
    SpeedRoad_li = ["SpeedRoad202508","SpeedRoad202509","SpeedRoad202510","SpeedRoad202511","SpeedRoad202512","SpeedRoad202601","SpeedRoad202602"]
    
    # 调用聚合管道版本的函数
    SpeedRoad = get_adasAlarm_with_pipeline(
        db_name="senba_gps",
        collection_names=SpeedRoad_li,
        batch_size=5000
    )
    
    # 查看结果（与原代码输出一致）
    print(f"总记录数：{len(SpeedRoad)}")
    print(SpeedRoad.head())
    
    # 可选：保存为CSV
    SpeedRoad.to_csv('./SpeedRoad.csv', index=False)
    end = time.time()
    print(f"Execution time: {round(end - start, 2)} seconds")