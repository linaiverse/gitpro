#!/bin/bash

# 切换到工作目录
cd /root/pythonWorkSpace/myenv/

# 激活 Python 虚拟环境
# source /root/pythonWorkSpace/myenv/bin/activate

# 执行所有 Python 脚本

python3 mysql.py 'Sys_Accident' '["AccidentTime","VehicleID","DriverID"]'
python3 mysql.py 'Sys_Driver' '["DriverID","DriverName","IDNumber","Certified","DriverTypeID","DrivingLicenseStartDate","EntryDate","ExamNum","TrainNum","ViolateNum","ViolationNum","AccidentNum"]' 
python3 mysql.py 'Sys_Vehicle' '["VehicleID","VehicleLic","Sim","TerminalEquipmentID","TerminalEquipmentManufacturer","AddTime","UpdateTime","IsEnable"]'
python3 mysql.py 'Sys_Violation' '["ViolationTime","VehicleID","DriverID","Type"]' 
python3 mysql.py 'VehicleBaseInfoStatusHistory' '["TimeSatellite","VehicleID","VehicleLic","DeviceID","IsGps","IsCamera","IsAeb"]' 

# python3 getMongo4.0.py '["DriverLoginInfo202508","DriverLoginInfo202509"]' '["time","certification","status"]' './DriverLoginInfo.csv'
python3 get_login.py '["DriverLoginInfo202508","DriverLoginInfo202509","DriverLoginInfo202510","DriverLoginInfo202511","DriverLoginInfo202512","DriverLoginInfo202601","DriverLoginInfo202602"]' '["time","certification","status"]' './DriverLoginInfo.csv'
# python3 getMongo4.0.py '["VehicleRunData202508","VehicleRunData202509"]' '["Addtime","certificateId","carNo","freeMileage","passengerTime","waitTime","mileage","money","judge","passengerCount"]' './VehicleRunData.csv'
python3 get_rundata.py
python3 get_aebs.py
python3 get_alarm.py
python3 get_overspeed.py
python3 location.py

# 退出虚拟环境
# deactivate
