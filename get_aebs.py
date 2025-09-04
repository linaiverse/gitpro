import pandas as pd
import pymongo
from typing import List, Optional
from pymongo import MongoClient
import time
import argparse

def get_aebswarning_with_pipeline(
    db_name: str, 
    collection_names: List[str], 
    batch_size: int = 5000
) -> pd.DataFrame:
    """
    使用MongoDB聚合管道获取司机车辆当日缓碰撞预警（AEBS）类型次数
    按warningType和AebsStatus分类，按日期+DriverID+DeviceID分组统计
    :param db_name: 数据库名
    :param collection_names: 需要查询的集合列表
    :param batch_size: 游标批次大小（控制内存占用）
    :return: 处理后的DataFrame
    """
    # 1. 连接MongoDB（添加超时控制，避免长期阻塞）
    client = MongoClient(
        host='10.128.11.182',
        port=27017,
        username='senbatest',
        password='mdGBp7ck',
        authSource='admin',
        authMechanism='SCRAM-SHA-1',
        serverSelectionTimeoutMS=5000  # 连接超时时间（5秒）
    )
    db = client[db_name]
    
    # 2. 筛选存在的集合（避免操作不存在的集合报错）
    existing_cols = [col for col in collection_names if col in db.list_collection_names()]
    if not existing_cols:
        print(f"警告：未找到任何有效集合（输入集合：{collection_names}）")
        client.close()
        return pd.DataFrame()
    
    # 3. 单个集合的核心聚合管道（核心：分类+分组统计）
    single_col_pipeline = [
        # 阶段1：筛选需要的字段（仅保留必要字段，减少数据量）
        {"$project": {
            "TimeSatellite": 1,    # 预警时间戳（秒）
            "DriverID": 1,         # 资格证ID（分组维度）
            "DeviceID": 1,         # 设备号（分组维度）
            "warningType": 1,      # 预警类型（分类依据1）
            "AebsStatus": 1,       # AEBS状态（分类依据2）
            "_id": 0               # 排除无用的默认ID
        }},
        # -------------------------- 新增：统一字段类型（关键兼容步骤）--------------------------
        {"$addFields": {
            # 1. 将warningType统一转换为字符串（处理数字/字符串混合场景）
            "warningTypeStr": {
        	"$cond": [
            	{"$eq": [{"$type": "$warningType"}, "string"]},  # 布尔表达式：是否为字符串
            	"$warningType",  # 是字符串：直接使用原始值（如"1"）
            	{"$toString": "$warningType"}  # 不是字符串：转为字符串（如1→"1"）
       	]
         },
            # 2. 将AebsStatus统一转换为字符串（同理）
            "AebsStatusStr": {
                "$cond": [
                    {"$eq": [{"$type": "$AebsStatus"}, "string"]},
                    "$AebsStatus",
                    {"$toString": "$AebsStatus"}
                ]
            }
        }},        



        # 阶段2：过滤有效预警（仅保留warningType为1-4的记录，匹配原需求）
        {"$match": {
            "warningTypeStr": {"$in": ["1", "2", "3", "4"]}  # 修复语法错误：键和值需用引号/大括号包裹
        }},
        
        # 阶段3：转换时间戳为日期（用于按“天”分组）
        {"$addFields": {
            "alarmDate": {
                "$dateTrunc": {
                    "date": {"$toDate": {"$multiply": ["$TimeSatellite", 1000]}},  # 秒→毫秒→MongoDB日期
                    "unit": "day"  # 截断到“天”级别（忽略时分秒，确保按日分组）
                }
            }
        }},
        
        # 阶段4：根据warningType和AebsStatus分类预警类型（替代Python的customer_type函数）
        {"$addFields": {
            "warningTypeName": {
                "$switch": {
                    "branches": [
                        # 所有判断基于统一的字符串类型
                        {"case": {"$eq": ["$warningTypeStr", "1"]}, "then": "FCW"},
                        {"case": {"$eq": ["$warningTypeStr", "2"]}, "then": "HMW"},
                        {"case": {"$eq": ["$warningTypeStr", "3"]}, "then": "LDW"},
                        # warningType=4时，结合AebsStatusStr判断
                        {"case": {"$and": [
                            {"$eq": ["$warningTypeStr", "4"]},
                            {"$eq": ["$AebsStatusStr", "0"]}
                        ]}, "then": "无制动"},
                        {"case": {"$and": [
                            {"$eq": ["$warningTypeStr", "4"]},
                            {"$eq": ["$AebsStatusStr", "1"]}
                        ]}, "then": "双目制动"},
                        {"case": {"$and": [
                            {"$eq": ["$warningTypeStr", "4"]},
                            {"$eq": ["$AebsStatusStr", "2"]}
                        ]}, "then": "毫米波制动"},
                        {"case": {"$and": [
                            {"$eq": ["$warningTypeStr", "4"]},
                            {"$eq": ["$AebsStatusStr", "3"]}
                        ]}, "then": "超声波制动"}
                    ],
                    "default": "Inactive"
                }
            }
        }},
        
        # 阶段5：按「日期+DriverID+DeviceID」分组，统计各类预警次数
        {"$group": {
            "_id": {
                "alarmDate": "$alarmDate",    # 分组维度1：日期
                "DriverID": "$DriverID",      # 分组维度2：资格证ID
                "DeviceID": "$DeviceID"       # 分组维度3：设备号
            },
            # 统计各类预警的次数（用$sum+$cond实现条件计数）
            "FCW": {"$sum": {"$cond": [{"$eq": ["$warningTypeName", "FCW"]}, 1, 0]}},
            "HMW": {"$sum": {"$cond": [{"$eq": ["$warningTypeName", "HMW"]}, 1, 0]}},
            "LDW": {"$sum": {"$cond": [{"$eq": ["$warningTypeName", "LDW"]}, 1, 0]}},
            "无制动": {"$sum": {"$cond": [{"$eq": ["$warningTypeName", "无制动"]}, 1, 0]}},
            "双目制动": {"$sum": {"$cond": [{"$eq": ["$warningTypeName", "双目制动"]}, 1, 0]}},
            "毫米波制动": {"$sum": {"$cond": [{"$eq": ["$warningTypeName", "毫米波制动"]}, 1, 0]}},
            "超声波制动": {"$sum": {"$cond": [{"$eq": ["$warningTypeName", "超声波制动"]}, 1, 0]}},
            "无效预警": {"$sum": {"$cond": [{"$eq": ["$warningTypeName", "Inactive"]}, 1, 0]}}
        }},
        
        # 阶段6：整理字段格式（将_id中的分组维度提取为顶层字段，便于后续分析）
        {"$project": {
            "alarmDate": "$_id.alarmDate",    # 日期（顶层字段）
            "DriverID": "$_id.DriverID",      # 资格证ID（顶层字段）
            "DeviceID": "$_id.DeviceID",      # 设备号（顶层字段）
            # 各类预警次数（保留所有统计项）
            "FCW": 1,
            "HMW": 1,
            "LDW": 1,
            "无制动": 1,
            "双目制动": 1,
            "毫米波制动": 1,
            "超声波制动": 1,
            "无效预警": 1,
            "_id": 0  # 排除临时的分组ID
        }}
    ]
    
    # 4. 构建多集合合并管道（用$unionWith合并所有有效集合）
    main_pipeline = []
    # 先添加第一个集合的处理逻辑
    main_pipeline.extend(single_col_pipeline)
    # 合并剩余集合（每个集合执行相同的管道）
    for col in existing_cols[1:]:
        main_pipeline.append({
            "$unionWith": {
                "coll": col,
                "pipeline": single_col_pipeline
            }
        })
    
    # 5. 执行聚合查询（添加异常捕获，确保连接安全关闭）
    try:
        cursor = db[existing_cols[0]].aggregate(
            pipeline=main_pipeline,
            batchSize=batch_size,
            allowDiskUse=True,           # 数据量大时用磁盘临时存储，避免内存溢出
            maxTimeMS=3600000            # 超时时间（1小时），避免长期阻塞
        )
        # 转换为DataFrame（直接从游标读取，效率高）
        result_df = pd.DataFrame(list(cursor))
    except Exception as e:
        print(f"聚合查询失败：{str(e)}")
        result_df = pd.DataFrame()
    finally:
        client.close()  # 无论成功与否，都关闭连接（避免连接泄漏）
    
    # 6. 优化日期格式（转换为字符串，便于Excel打开和阅读）
    if not result_df.empty:
        result_df["alarmDate"] = pd.to_datetime(result_df["alarmDate"]).dt.strftime("%Y-%m-%d")
    
    return result_df

# -------------------------- 调用示例 --------------------------
if __name__ == "__main__":
    start_time = time.time()
    
    # 配置参数（根据实际需求修改）
    DB_NAME = "senba_gps"
    COLLECTION_NAMES = ["AebsWarningInfo202508","AebsWarningInfo202509","AebsWarningInfo202510","AebsWarningInfo202511","AebsWarningInfo202512","AebsWarningInfo202601","AebsWarningInfo202602"]  # 预警数据集合（单个或多个）
    BATCH_SIZE = 10000  # 批次大小（内存足够时可增大，减少I/O次数）
    
    # 调用函数统计AEBS预警次数
    aebs_warning_df = get_aebswarning_with_pipeline(
        db_name=DB_NAME,
        collection_names=COLLECTION_NAMES,
        batch_size=BATCH_SIZE
    )
    
    # 处理结果
    if not aebs_warning_df.empty:
        # 保存为CSV（支持中文编码）
        aebs_warning_df.to_csv('./aebs_warning_count.csv', index=False, encoding="utf-8-sig")
        # 打印关键信息
        print("AEBS预警次数统计完成！")
        print(f"总分组数：{len(aebs_warning_df)}（按每日-司机-设备分组）")
        print(f"前5条数据：")
        print(aebs_warning_df.head())
    else:
        print("未统计到任何AEBS预警数据（可能集合无数据或筛选后无匹配记录）")
    
    # 打印执行时间
    end_time = time.time()
    print(f"\n执行时间：{round(end_time - start_time, 2)} 秒")