# 基础库
import numpy as np
import pandas as pd
import itertools
import math
import os
from collections import defaultdict
# 连接数据库
import pymysql
from sqlalchemy import create_engine, text
# datetimes
import time
import warnings
import json
from datetime import date, timedelta, datetime
from meteostat import Point, Daily # 根据时空获取天气

# 全局设置
warnings.filterwarnings('ignore')

start = time.time()
# 读取JSON文件
with open('./data_type.json', 'r', encoding='utf-8') as f:
    final_json = json.load(f)  # 返回Python字典或列表
# final_json

#  定义日期判断函数
def get_date_type(date_obj, region="CN"):

    # 结果匹配
    date_str = date_obj.strftime("%Y-%m-%d")
    for entry in final_json["holidays"]:
        if entry["date"] == date_str:
            # 根据类型返回结果
            if entry["type"] == "public_holiday":
                return "法定节假日"
            elif entry["type"] == "transfer_workday":
                return "调休工作日"
    
    # 未在节假日数据中找到时，按星期判断
    if date_obj.weekday() < 5:  # 周一至周五
        return "普通工作日"
    else:
        return "周末"  


# 连接MySQL数据库
conn = pymysql.connect(
    host="127.0.0.1",
    user="root",
    password="root",
    database="CaiKuData",
    charset="utf8mb4"
)

# 分批读取与合并
def read_sql(sql, conn):
    chunks = []
    for chunk in pd.read_sql(sql, conn, chunksize=50000): # 每次读5000行[6](@ref)
        chunks.append(chunk)
    # print(chunks)
    return pd.concat(chunks, ignore_index=True)

# 事故数据
accident = pd.read_sql("SELECT * FROM Sys_Accident", conn)
# accident = read_sql("SELECT AccidentTime, VehicleID, DriverID,LncidentArea,SpecificArea, Damage FROM Sys_Accident", conn)
accident['AccidentDate'] = pd.to_datetime(accident['AccidentTime']).dt.date
accident.drop_duplicates(keep='last',inplace=True)
accident = accident.astype({'VehicleID':str,'DriverID':str})

# 司机维表 ok
driver = pd.read_sql("SELECT * FROM Sys_Driver", conn)
driver.drop_duplicates(subset=['IDNumber','Certified'],keep='last',inplace=True)
driver = driver.astype({'Certified':str, 'DriverID':str})
# DriverTypeID	司机类型（1主班、2副班、3机动）
driver['DriverTypeID'] = pd.to_numeric(driver['DriverTypeID'], errors='coerce')
driver = driver[driver['Certified']!='9111199910'] # 排除资格证号：9111199910  翁任畅
driver = driver[driver["IDNumber"].str.len() == 18]
# driver['IsShenzhen'] = driver['IDNumber'].str[:4].eq("4403")
driver['IsGuangdong'] = driver['IDNumber'].str[:2].apply(lambda x: 1 if x == "44"  else 0 )
driver['sex'] = driver['IDNumber'].str[16].astype(int)
driver['sex'] = [1 if digit % 2 == 1 else 0 for digit in driver['sex']] # 1：男，0：女
driver['DrivingLicenseStartDate'] = pd.to_datetime(driver['DrivingLicenseStartDate'])
driver['EntryDate'] = pd.to_datetime(driver['EntryDate'])
driver['brithday'] = pd.to_datetime(driver['IDNumber'].str.slice(6, 14), format="%Y%m%d",errors="coerce")

# 车辆维表 ok
Vehicle = read_sql("SELECT * FROM Sys_Vehicle", conn)
Vehicle.sort_values(by='AddTime', inplace=True, ascending=True)
Vehicle.drop_duplicates(subset=['VehicleLic'],keep='last',inplace=True)
Vehicle['TEM'] = np.where(Vehicle['TerminalEquipmentManufacturer'].notna(),np.where(Vehicle['TerminalEquipmentManufacturer'].str.contains('CK-R'),1,0),np.nan)

# 违章违规记录 ok
violation = read_sql("SELECT * FROM Sys_Violation", conn)
violation['Date'] = pd.to_datetime(violation['ViolationTime']).dt.date
violation = violation.astype({'DriverID':str})
violation = pd.merge(left=violation,right=driver,left_on='DriverID',right_on='DriverID')
violationType = pd.crosstab(
    index=[violation['Date'], violation['VehicleID'], violation['Certified']],
    columns=violation['Type']
).reset_index().rename_axis(None, axis=1)
violationType.rename(columns={1:'违章',2:'违规'},inplace=True)
required_columns = ['Date','VehicleID','Certified','违章','违规']
violationType = violationType.assign(**{col: 0 for col in required_columns if col not in violationType})
violationType = violationType[required_columns]
# violationType.dropna(how='any',subset=['Certified'],inplace=True)
violationType['Certified'] = violationType['Certified'].astype(str)
violationType['VehicleID'] = violationType['VehicleID'].astype(str)

# 车辆系统状态表 ok
baseinfohis = read_sql("SELECT * FROM VehicleBaseInfoStatusHistory", conn)
baseinfohis.sort_values(by='TimeSatellite', inplace=True, ascending=True)
baseinfohis['TimeSatellite'] = pd.to_datetime(baseinfohis['TimeSatellite']).dt.date
baseinfohis.drop_duplicates(subset=['TimeSatellite','VehicleID'], keep='last',inplace=True)

# 获取车辆总里程数据
totalmileage = read_sql("SELECT * FROM daily_total_mileage", conn)
# totalmileage = pd.read_csv('./daily_max_mileage.csv')

# 获取司机当日速度数据
# GpsDaySpeed = pd.read_csv('./daily_speed_stats.csv')

# 获取司机当日驾驶时长数据


# 获取司机车辆当日主动安全报警类型次数
Alarm= pd.read_csv('./alarm_optimized.csv',dtype={'jobNumber':'str'})
Alarm ['alarmDate'] = pd.to_datetime(Alarm['alarmDate']).dt.date
Alarm.dropna(subset=['jobNumber'],inplace=True,axis=0)
group_cols = ["alarmDate","jobNumber","vehicleNumber"]
# 获取所有数值列（包括整数和浮点数）
numeric_cols = Alarm.select_dtypes(include=['int', 'float', 'int64', 'float64']).columns
# 排除分组列
agg_cols = [col for col in numeric_cols if col not in group_cols]
# 创建聚合字典
agg_dict = {col: 'sum' for col in agg_cols}
Alarm = Alarm.groupby(group_cols).agg(agg_dict).reset_index()

# 根据warningType和AebsStatus分类报警类型
Aebs = pd.read_csv('./aebs_warning_count.csv',dtype={'DriverID':'str','DeviceID':'str'})
Aebs['alarmDate'] = pd.to_datetime(Aebs['alarmDate']).dt.date
Aebs = Aebs.groupby(['alarmDate','DeviceID','DriverID']).agg(
{'FCW':'sum','HMW':'sum','LDW':'sum','无制动':'sum','双目制动':'sum','毫米波制动':'sum','超声波制动':'sum','无效预警':'sum'}
).reset_index()

# 获取司机车辆当日营运数据
# VehicleRunData = pd.read_csv('./VehicleRunData202508.csv')
# VehicleRunData['AddDateTime'] = pd.to_datetime(VehicleRunData['Addtime'],unit='s')
#VehicleRunData['passengerTime'] = VehicleRunData['passengerTime'].astype(int)
#VehicleRunData['NegativeReview'] = np.where(VehicleRunData["judge"].isin([3,4]), 1, 0)
#VehicleRunData['CarryPassengerNum'] = np.where(VehicleRunData["passengerCount"]>0, 1, 0)


VehicleRunData = read_sql("SELECT * FROM vehicle_run_data_daily", conn)
VehicleRunData['AddDate'] = pd.to_datetime(VehicleRunData['AddDate']).dt.date
VehicleRunData['certificateId'] = VehicleRunData['certificateId'].astype(str)
VehicleRunDataDays = VehicleRunData.groupby(['certificateId','AddDate','carNo']).agg(
{'freeMileage':'sum','passengerTime':'sum','waitTime':'sum','mileage':'sum','money':'sum','NegativeReview':'sum','CarryPassengerNum':'sum'}
).reset_index()
# VehicleRunDataDays.to_csv('./FitModel_VehicleRunDataDays.csv',index=None)
# 计算连续签到天数
def calculate_consecutive_days(df, date_col='date'):
    # 1. 确保日期列是datetime类型
    df[date_col] = pd.to_datetime(df[date_col])
    
    # 2. 获取所有日期并排序
    all_dates = df[date_col].sort_values().unique()
    
    # 3. 创建完整日期范围（从最小日期到最大日期）
    min_date = all_dates.min()
    max_date = all_dates.max()
    full_date_range = pd.date_range(start=min_date, end=max_date)
    
    # 4. 创建存在标记（1表示日期存在，0表示缺失）
    date_exists = pd.Series(1, index=all_dates)
    date_exists = date_exists.reindex(full_date_range, fill_value=0)
    
    # 5. 计算连续天数（包括当前日期）
    # 使用分组技巧计算连续天数
    groups = (date_exists != date_exists.shift(1)).cumsum()
    consecutive_counts = date_exists.groupby(groups).cumsum()
    
    # 6. 创建连续天数Series（索引为日期）
    consecutive_series = pd.Series(consecutive_counts.values, index=full_date_range)
    
    # 7. 计算前一天连续天数（不含当前日期）
    # 前一天连续天数 = 当前日期的连续天数 - 1
    prev_day_consecutive = consecutive_series
    
    # 8. 合并结果到原始数据
    result_df = df.copy()
    result_df['prev_day_consecutive'] = result_df[date_col].map(
        lambda d: prev_day_consecutive.get(d - timedelta(days=1), 0)
    )
    
    return result_df

# 获取司机连续工作天数
DriverLogin = pd.read_csv('./DriverLoginInfo.csv')
DriverLogin.sort_values(by='time', inplace=True, ascending=True)
DriverLogin['DateTime'] = pd.to_datetime(DriverLogin['time'],unit='s')
DriverLogin['Date'] = pd.to_datetime(DriverLogin['DateTime']).dt.date
DriverLogin = pd.crosstab(
    index=[DriverLogin['Date'], DriverLogin['certification']],
    columns=DriverLogin['status']
).reset_index().rename_axis(None, axis=1)
DriverLogin.rename(columns={0:'签退',1:'签到'},inplace=True)
required_columns = ['Date','certification','签退','签到']
DriverLogin = DriverLogin.assign(**{col: 0 for col in required_columns if col not in DriverLogin})
DriverLogin = DriverLogin[required_columns]
# 按资格证计算连续签到天数
member_consecutive = DriverLogin.groupby('certification').apply(
    lambda group: calculate_consecutive_days(group, 'Date')
).reset_index(drop=True)
member_consecutive

# 获取司机车辆超速数据
OverSpeed = pd.read_csv('./SpeedRoad.csv',dtype={'jobNumber':'str'})
OverSpeed.rename(columns={'alarmDate':'Date'},inplace=True)
OverSpeed = OverSpeed.groupby(['Date','jobNumber','vehicleNumber']).agg({'超速次数':'sum'}).reset_index()

##############################################################################
## 合并数据集
# 级联司机维度数据 处理同一司机身份证下有多个资格证id的问题
# driver_ID = driver[['Certified','IDNumber']]
# VehicleRunDataDays = pd.merge(left=VehicleRunDataDays,right=driver_ID,how='left',left_on='certificateId',right_on='Certified')
# VehicleRunDataDays = VehicleRunDataDays.loc[:,VehicleRunDataDays.columns != 'Certified']
# 找出每个A值最后出现的B值
# last_values = VehicleRunDataDays.groupby('IDNumber')['certificateId'].last().reset_index()
# 创建映射字典
# mapping = dict(zip(last_values['IDNumber'], last_values['certificateId']))
# 应用映射替换B列值
# VehicleRunDataDays['certificateId'] = VehicleRunDataDays['IDNumber'].map(mapping)
#############################################################################
# 级联司机维度数据
driver = driver.astype({'Certified':str,'IDNumber':str})
data1 = pd.merge(left=VehicleRunDataDays,right=driver.loc[:,driver.columns != 'IDNumber'],how='left',left_on='certificateId',right_on='Certified')
data1['age'] = (pd.to_datetime(data1['AddDate']) - data1['brithday']).dt.days
data1['driveAge'] = np.where(data1['DrivingLicenseStartDate'].notna(),(pd.to_datetime(data1['AddDate']) - data1['DrivingLicenseStartDate']).dt.days,np.nan)
data1['EntryAge'] = np.where(data1['EntryDate'].notna(),(pd.to_datetime(data1['AddDate']) - data1['EntryDate']).dt.days,np.nan)
data1 = data1.drop(columns=['DrivingLicenseStartDate', 'EntryDate', 'brithday'])
# data1.to_csv('./checkdata/FitModelData1.csv',index=None)
print(f" VehicleRunDataDays: {len(VehicleRunDataDays)}")
print(f" data1: {len(data1)}")
# 级联车辆维表
Vehicle_dim = Vehicle[['VehicleID','VehicleLic','TerminalEquipmentID','TEM']]
Vehicle_dim['VehicleID'] = Vehicle_dim['VehicleID'].astype('str')  # 转换为字符串
Vehicle_dim['TerminalEquipmentID'] = Vehicle_dim['TerminalEquipmentID'].astype('str')  # 转换为字符串
data2 = pd.merge(left=data1,right=Vehicle_dim,how='left',left_on='carNo',right_on='VehicleLic')
data2 = data2.loc[:,data2.columns != 'VehicleLic']
print(f" data2: {len(data2)}")
# data2.to_csv('./checkdata/FitModelData2.csv',index=None)
# 级联车辆系统数据
baseinfohis['VehicleID'] = baseinfohis['VehicleID'].astype('str')  # 转换为字符串
data3 = pd.merge(left=data2,right=baseinfohis,how='left',left_on=['AddDate','VehicleID'],right_on=['TimeSatellite','VehicleID'])
data3 = data3.drop(columns=['TimeSatellite', 'VehicleLic'])
print(f" data3: {len(data3)}")
# data3.to_csv('./checkdata/FitModelData3.csv',index=None)
# 级联车辆总里程数据
totalmileage['gpsDate'] = pd.to_datetime(totalmileage['gpsDate']).dt.date
totalmileage['deviceNo'] = totalmileage['deviceNo'].astype(str)
totalmileage = totalmileage.rename(columns={'daily_max_mileage':'totalmileage'})
totalmileage.sort_values(by=['gpsDate','deviceNo','totalmileage'], inplace=True, ascending=True)
totalmileage.drop_duplicates(subset=['gpsDate','deviceNo'],keep='last',inplace=True)
data4 = pd.merge(left=data3,right=totalmileage,how='left',left_on=['AddDate','TerminalEquipmentID'],right_on=['gpsDate','deviceNo'])
data4 = data4.drop(columns=['gpsDate','deviceNo','daily_record_count'])
# print(totalmileage.info())
print(f" data4: {len(data4)}")
# data4.to_csv('./checkdata/FitModelData4.csv',index=None)
# 级联司机当天主动安全报警次数
data8 = pd.merge(left=data4,right=Alarm,how='left',left_on=['AddDate','certificateId','carNo'],right_on=['alarmDate','jobNumber','vehicleNumber'])
data8 = data8.drop(columns=['alarmDate','jobNumber','vehicleNumber'])
# print(Alarm.info())
print(f" data8: {len(data8)}")
# data8.to_csv('./checkdata/FitModelData8.csv',index=None)
# 级联司机当天缓碰撞预警次数

# Aebs.dropna(subset=['DriverID'],inplace=True,axis=0)
Aebs = Aebs.rename(columns={'DeviceID': 'DeviceID_y','DriverID':'DriverID_y'})
data9 = pd.merge(left=data8,right=Aebs,how='left',left_on=['AddDate','TerminalEquipmentID','certificateId'],right_on=['alarmDate','DeviceID_y','DriverID_y'])
data9 = data9.drop(columns=['alarmDate','DeviceID_y','DriverID_y'])
# print(Aebs.info())
print(f" data9: {len(data9)}")
# data9.to_csv('./checkdata/FitModelData9.csv',index=None)
# 级联司机当天违规违章次数
violationType['gpsDate']= pd.to_datetime(violationType['Date']).dt.date
violationType = violationType.astype({'VehicleID':str,'Certified':str})
violationType = violationType.rename(columns={'Certified': 'Certified_y'})
data10 = pd.merge(left=data9,right=violationType,how='left',left_on=['AddDate','VehicleID','certificateId'],right_on=['Date','VehicleID','Certified_y'])
data10 = data10.drop(columns=['Date','Certified_y','gpsDate'])
# print(violationType.info())
print(f" data10: {len(data10)}")
# data10.to_csv('./checkdata/FitModelData10.csv',index=None)
# 级联司机当天（不含）连续签到数据
member_consecutive['certification'] = member_consecutive['certification'].astype(str)
member_consecutive['Date']= pd.to_datetime(member_consecutive['Date']).dt.date
data11 = pd.merge(left=data10,right=member_consecutive,how='left',left_on=['AddDate','certificateId'],right_on=['Date','certification'])
data11 = data11.drop(columns=['Date','certification','签退','签到'])
# print(member_consecutive.info())
print(f" data11: {len(data11)}")
# data11.to_csv('./checkdata/FitModelData11.csv',index=None)
# 级联司机当天超速数据
OverSpeed['Date']= pd.to_datetime(OverSpeed['Date']).dt.date
data12 = pd.merge(left=data11,right=OverSpeed,how='left',left_on=['AddDate','certificateId','carNo'],right_on=['Date','jobNumber','vehicleNumber'])
data12 = data12.drop(columns=['Date','jobNumber','vehicleNumber'])
# print(OverSpeed.info())
print(f" data12: {len(data12)}")
# data12.to_csv('./checkdata/FitModelData12.csv',index=None)

# 根据日期获取日期类型
data12["date_type"] = pd.to_datetime(data12['AddDate']).apply(get_date_type)
data13 = data12
# data13.to_csv('./checkdata/FitModelData13.csv',index=None)
# 级联事故司机车辆日期数据
# print(data13.info())
data14 = pd.merge(left=data13,right=accident,how='left',left_on=['AddDate','DriverID','VehicleID'],right_on=['AccidentDate','DriverID','VehicleID'])
data14['target'] = np.where(data14['AccidentTime'].notna(),1,0)
data14 = data14.drop(columns=['AccidentTime','AccidentDate'])
print(f" data14: {len(data14)}")
print(data14.info())
# 删除测试人员的数据
data14 = data14[~data14['VehicleID'].astype(str).isin(['64069','64071'])] # 排除测试车辆
data14 = data14[~data14['certificateId'].astype(str).isin(['9111199910'])] # 排除测试车辆
data14 = data14[~data14['carNo'].astype(str).isin(['粤B22222'])] # 排除测试车辆

data14.to_csv('./FitModelData.csv',index=None)

# 1. 读取目标数据库配置（已提供）
TARGET_CONFIG = {
    "host": "127.0.0.1",    # 目标数据库主机地址（本地）
    "port": 3306,           # MySQL 默认端口
    "user": "root",         # 数据库用户名
    "password": "root",     # 数据库密码
    "database": "CaiKuData" # 目标数据库名称
}
# 2. 构建 SQLAlchemy 连接引擎（格式：mysql+pymysql://用户:密码@主机:端口/数据库）
engine_url = f"mysql+pymysql://{TARGET_CONFIG['user']}:{TARGET_CONFIG['password']}@{TARGET_CONFIG['host']}:{TARGET_CONFIG['port']}/{TARGET_CONFIG['database']}"
engine = create_engine(engine_url)

# 3. 定义写入参数（根据需求调整）
table_name = "fit_model_data"  # 目标表名（可自定义，如：司机车辆风险模型数据）
if_exists = "replace"        # 表存在时的处理：replace=覆盖，append=追加，fail=报错（推荐首次用replace，后续用append）
chunksize = 10000            # 分批写入（避免数据量过大时内存溢出，建议1-5万行/批）
dtype = None                 # 若需指定字段类型（如日期、字符串），可在这里定义（示例见下方说明）
index = False                # 是否将 pandas 的索引作为表的一列（建议False，避免多余列）

# 4. 执行数据写入
if len(VehicleRunDataDays) >= len(data14):
    try:
        print(f"开始写入数据到 MySQL 表：{TARGET_CONFIG['database']}.{table_name}")
        data14.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            chunksize=chunksize,
            dtype=dtype,
            index=index
        )
        print(f"数据写入成功！共写入 {len(data14)} 行数据")
    except Exception as e:
        print(f"数据写入失败！错误信息：{str(e)}")
    finally:
        # 关闭数据库连接（释放资源）
        engine.dispose()
        print("数据库连接已关闭")
else:
    engine.dispose()
    print("数据产生笛卡尔积不予写入，数据库连接关闭")
end = time.time()
print(f"Execution time: {round(end - start, 2)} seconds")