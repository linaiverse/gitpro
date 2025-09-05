# 基础库
import numpy as np
import pandas as pd
# 机器学习算法
from sklearn.preprocessing import OneHotEncoder  # 独热编码，将分类变量转为二进制
from sklearn.preprocessing import StandardScaler  # 特征缩放，标准化
from sklearn.model_selection import GridSearchCV  # 网格搜素
from sklearn.model_selection import train_test_split  # 按比例划分训练测试集
from sklearn.ensemble import RandomForestClassifier  # 随机森林
from sklearn.linear_model import LogisticRegression  # 逻辑斯蒂回归
from sklearn.neighbors import KNeighborsClassifier  # K近邻
from sklearn.ensemble import GradientBoostingClassifier  # 梯度提升决策树
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE  # 过采样
from sklearn.metrics import auc, roc_auc_score, roc_curve, recall_score, accuracy_score, classification_report
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
# datetimes
from datetime import date, datetime, timedelta
import warnings,time
import joblib  # 保存或加载模型
from sqlalchemy import create_engine # 连接数据库
import pymysql
from pymysql import Error

# 全局设置
warnings.filterwarnings('ignore')
# 目标数据库配置
TARGET_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "root",
    "database": "CaiKuData"
}

# 固定表名
TABLE_NAME = "fit_model_data"

def get_date_range():
    """计算日期范围：昨天到前8天"""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    eight_days_ago = today - timedelta(days=15)
    return eight_days_ago, yesterday

def query_fit_model_data():
    connection = None
    cursor = None
    try:
        connection = pymysql.connect(
            host=TARGET_CONFIG["host"],
            port=TARGET_CONFIG["port"],
            user=TARGET_CONFIG["user"],
            password=TARGET_CONFIG["password"],
            database=TARGET_CONFIG["database"]
        )
        
        # 获取表结构，获取实际列名
        cursor = connection.cursor()
        cursor.execute(f"DESCRIBE {TABLE_NAME}")
        columns = [col[0] for col in cursor.fetchall()]
        print(f"表 {TABLE_NAME} 的实际列名: {columns}")
        
        # 执行数据查询
        start_date, end_date = get_date_range()
        query = f"""
        SELECT * FROM {TABLE_NAME}
        WHERE AddDate BETWEEN %s AND %s
        """
        print(f"查询表: {TABLE_NAME}")
        print(f"查询日期范围: {start_date} 至 {end_date}")
        
        cursor.execute(query, (start_date, end_date))
        records = cursor.fetchall()
        
        print(f"查询到 {len(records)} 条记录")
        
        # 如果返回的是元组列表而不是字典列表，手动指定列名
        if records and isinstance(records[0], tuple):
            df = pd.DataFrame(records, columns=columns)
        else:
            df = pd.DataFrame(records)
            
        return df if not df.empty else None
        
    except Error as e:
        print(f"数据库错误: {e}")
        return None
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception as e:
                print(f"关闭游标时出错: {e}")
        
        if connection:
            try:
                connection.close()
                print("数据库连接已关闭")
            except Exception as e:
                print(f"关闭连接时出错: {e}")

# df = query_fit_model_data()
# print(df.head())

def sjycl():
    start_time = time.time() 
    df = query_fit_model_data()
    # df = pd.read_csv('./FitModelData.csv')
    df['AddDate'] = pd.to_datetime(df['AddDate'])
    df = df.rename(columns={'超速次数':'over_speed_cnt'})
    # 车辆总里程缺失值填充
    df = df.sort_values(by=['carNo','AddDate']).reset_index(drop=True)
    # 创建临时处理列（避免修改原始数据）
    df['_tmp_mileage'] = df['totalmileage']
    # 将需要替换的值设置为NaN：
    # 1. 原始缺失值（NaN）
    # 2. 小于等于零的值
    df['_tmp_mileage'] = np.where(
        (df['_tmp_mileage'] <= 0) | pd.isna(df['_tmp_mileage']),
        np.nan,
        df['_tmp_mileage']
    )
    # 分组填充：仅使用大于0的值进行前向填充
    df['_tmp_mileage'] = df.groupby('carNo')['_tmp_mileage'].transform(
        lambda x: x.ffill()
    )
    # 处理序列开头的缺失值（获取组内第一个有效值）
    df['totalmileage'] = df.groupby('carNo')['_tmp_mileage'].transform(
        lambda x: x.bfill()  # 使用第一个有效值向后填充开头
    )
    # 移除临时列
    df = df.drop(columns=['_tmp_mileage'])
    # 最终处理将缺失值用-1做标记
    df['totalmileage'] = df['totalmileage'].fillna(-1)
    # 去重
    df.drop_duplicates(subset=['AddDate','carNo','DriverName'], keep='first', inplace=True, ignore_index=True)
    # 2. 日期转换+排序（确保分组内日期单调）
    df['AddDate'] = pd.to_datetime(df['AddDate'])
    df = df.sort_values(by=['carNo', 'DriverName', 'AddDate']).reset_index(drop=True)
    # 单位换算
    df['freeMileage'] = df['freeMileage']/1000
    df['passengerTime'] = df['passengerTime']/3600
    df['waitTime'] = df['waitTime']/3600
    df['mileage'] = df['mileage']/1000
    df['money'] = df['money']/100
    df['age'] = df['age']/365


    # 异常值检测
    def handle_outliers(df, fields, method='truncate'):
        """
        使用3σ原则处理多个字段的异常值

        参数:
        df: 输入的DataFrame
        fields: 需要处理异常值的字段列表
        method: 处理方式，'truncate'表示截断异常值，'remove'表示删除异常值所在行

        返回:
        处理后的DataFrame
        """
        # 复制原数据，避免修改原始数据
        df_processed = df.copy()

        if method == 'remove':
            # 标记所有非异常值的行
            keep_rows = pd.Series([True] * len(df_processed))

            for field in fields:
                # 计算字段的均值和标准差
                mean_val = df_processed[field].mean()
                std_val = df_processed[field].std()

                # 计算3σ上下界
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val

                # 标记在正常范围内的行
                is_normal = (df_processed[field] >= lower_bound) & (df_processed[field] <= upper_bound)
                keep_rows = keep_rows & is_normal

            # 保留非异常值的行
            df_processed = df_processed[keep_rows].reset_index(drop=True)
            print(f"删除异常值后保留了 {len(df_processed)} 行数据")

        elif method == 'truncate':
            df_processed['yichang'] = 0
            for field in fields:
                # 计算字段的均值和标准差
                mean_val = df_processed[field].mean()
                std_val = df_processed[field].std()

                # 计算3σ上下界
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val

                # 先标记异常值（在替换之前）
                is_lower_outlier = df_processed[field] < lower_bound
                is_upper_outlier = df_processed[field] > upper_bound

                # 统计异常值，累加到yichang字段
                df_processed['yichang'] += is_lower_outlier.astype(int) + is_upper_outlier.astype(int)

                # 截断异常值：小于下界的用下界替换，大于上界的用上界替换
                df_processed[field] = np.where(is_lower_outlier,
                                               lower_bound,
                                               np.where(is_upper_outlier,
                                                        upper_bound,
                                                        df_processed[field]))

                print(f"处理字段 {field}: 下界={lower_bound:.4f}, 上界={upper_bound:.4f}, "
                      f"异常值数量={sum(is_lower_outlier | is_upper_outlier)}")

            df_processed['IsAeb'] = np.where(((df_processed['IsAeb'] == 1) & (df_processed['yichang'] > 0)), 0,
                                             df_processed['IsAeb'])

        else:
            raise ValueError("method参数只能是'remove'或'truncate'")

        return df_processed


    # 需要处理异常值的字段列表
    fields_to_process = ['LDW', 'FCW', '毫米波制动', 'HMW', '超声波制动', '双目制动']

    # 方法1：截断异常值（默认）
    df = handle_outliers(df, fields_to_process, method='truncate')

    # # 方法2：删除异常值所在行
    # df_removed = handle_outliers(df, fields_to_process, method='remove')
    def handle_outliers_without0(df, fields, method='truncate'):
        """
        使用3σ原则处理多个字段的异常值，字段值为0则不执行处理

        参数:
        df: 输入的DataFrame
        fields: 需要处理异常值的字段列表
        method: 处理方式，'truncate'表示截断异常值，'remove'表示删除异常值所在行

        返回:
        处理后的DataFrame
        """
        # 复制原数据，避免修改原始数据
        df_processed = df.copy()

        if method == 'remove':
            # 标记所有非异常值的行
            keep_rows = pd.Series([True] * len(df_processed))

            for field in fields:
                # 计算字段的均值和标准差（排除0值）
                non_zero_vals = df_processed[df_processed[field] != 0][field]
                if len(non_zero_vals) == 0:
                    print(f"字段 {field} 所有值均为0，不进行异常值处理")
                    continue

                mean_val = non_zero_vals.mean()
                std_val = non_zero_vals.std()

                # 计算3σ上下界
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val

                # 标记在正常范围内的行（0值视为正常）
                is_normal = (df_processed[field] == 0) | \
                            ((df_processed[field] >= lower_bound) & (df_processed[field] <= upper_bound))
                keep_rows = keep_rows & is_normal

            # 保留非异常值的行
            df_processed = df_processed[keep_rows].reset_index(drop=True)
            print(f"删除异常值后保留了 {len(df_processed)} 行数据")

        elif method == 'truncate':
            for field in fields:
                # 获取非0值
                non_zero_vals = df_processed[df_processed[field] != 0][field]
                if len(non_zero_vals) == 0:
                    print(f"字段 {field} 所有值均为0，不进行异常值处理")
                    continue

                # 基于非0值计算均值和标准差
                mean_val = non_zero_vals.mean()
                std_val = non_zero_vals.std()

                # 计算3σ上下界
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val

                # 截断异常值：只处理非0值，0值保持不变
                mask = df_processed[field] != 0  # 非0值的掩码
                df_processed.loc[mask & (df_processed[field] < lower_bound), field] = lower_bound
                df_processed.loc[mask & (df_processed[field] > upper_bound), field] = upper_bound

                print(f"处理字段 {field}: 下界={lower_bound:.4f}, 上界={upper_bound:.4f}")

        else:
            raise ValueError("method参数只能是'remove'或'truncate'")

        return df_processed


    # 需要处理异常值的字段列表
    fields_to_process = ['adas违规', '报警', '提醒', '风控', '故障', '急加速次数', '急减速次数', '重度疲劳次数',
                         '接打电话次数', '抽烟次数', '注意力分散次数'
        , '斑马线未礼让行人次数', '肢体冲突次数', '斑马线超速次数']

    # 方法1：截断异常值（默认）
    df = handle_outliers_without0(df, fields_to_process, method='truncate')

    # # 方法2：删除异常值所在行
    # df_removed = handle_outliers_without0(df, fields_to_process, method='remove')
    # 设备状态缺失值填充
    df[['IsGps','IsCamera','IsAeb']] = df[['IsGps','IsCamera','IsAeb']].fillna(method='ffill')
    # 连续签到、超速次数和其他静态属性缺失值填充
    df[['prev_day_consecutive','over_speed_cnt','ExamNum','TrainNum']]= df[['prev_day_consecutive','over_speed_cnt','ExamNum','TrainNum']].fillna(0)
    # 定义需要填充的字段列表
    fields = ['adas违规','报警','提醒','风控', '故障', '急加速次数',
              '急减速次数', '重度疲劳次数', '接打电话次数', '抽烟次数',
              '注意力分散次数', '斑马线未礼让行人次数', '肢体冲突次数',
              '斑马线超速次数', 'LDW', 'FCW', '毫米波制动',
              'HMW', '超声波制动', '双目制动', '违章', '违规']
    # df[fields] = df[fields].fillna(df[fields].mean())


    def fill_group_na(df, fields, group_keys=['carNo', 'DriverName']):
        """
        高效的分组填充缺失值函数（替代逐行apply）
        """
        # 1. 计算分组均值（仅保留非缺失值的均值）
        group_means = df.groupby(group_keys)[fields].mean().reset_index()

        # 2. 用merge将分组均值关联回原始数据（向量化操作）
        # 临时列名：在字段名后加"_group_mean"，避免与原始字段冲突
        mean_suffix = "_group_mean"
        mean_columns = {field: f"{field}{mean_suffix}" for field in fields}
        group_means = group_means.rename(columns=mean_columns)

        # 合并：根据分组键将均值表关联到原始数据
        df_merged = df.merge(group_means, on=group_keys, how='left')

        # 3. 用分组均值填充缺失值（批量处理所有字段）
        for field in fields:
            # 原始字段缺失时，用分组均值填充
            df[field] = df[field].fillna(df_merged[f"{field}{mean_suffix}"])

        # 4. 剩余缺失值用全局均值填充
        df[fields] = df[fields].fillna(df[fields].mean())

        return df


    # 第一次分组填充（高效版）
    fields1 = ['adas违规', '报警', '提醒', '风控', '故障', '急加速次数',
               '急减速次数', '重度疲劳次数', '接打电话次数', '抽烟次数',
               '注意力分散次数', '斑马线未礼让行人次数', '肢体冲突次数',
               '斑马线超速次数', 'LDW', 'FCW', '毫米波制动',
               'HMW', '超声波制动', '双目制动', '违章', '违规']
    df = fill_group_na(df, fields1)




    def rolling_window_calculator(
            df,
            group_keys=['carNo', 'DriverName'],
            date_col='AddDate',
            target_cols=['mileage'],
            agg_funcs=['mean'],
            window='7D',
            exclude_current=True
    ):
        """
        通用滚动窗口计算函数，支持多字段和多聚合函数

        参数说明：
        - df: 输入DataFrame
        - group_keys: 分组键列表
        - date_col: 日期列名称
        - target_cols: 目标字段列表（如['mileage', 'speed']）
        - agg_funcs: 聚合函数列表（支持'mean', 'sum', 'count', 'max', 'min', 'std'）
        - window: 窗口大小（如'7D'表示7天）
        - exclude_current: 是否排除当前行数据（True/False）

        返回：
        - 包含所有滚动计算结果的DataFrame
        """
        # 复制数据避免修改原数据
        df_copy = df.copy()

        # 数据预处理：确保日期类型和正确排序
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
        df_copy = df_copy.sort_values(by=group_keys + [date_col]).reset_index(drop=True)

        # 验证聚合函数是否支持
        supported_funcs = ['mean', 'sum', 'count', 'max', 'min', 'std']
        for func in agg_funcs:
            if func not in supported_funcs:
                raise ValueError(f"不支持的聚合函数: {func}，支持的函数包括: {supported_funcs}")

        # 分组处理函数
        def process_group(group):
            # 对每个目标字段和聚合函数组合进行计算
            for col in target_cols:
                for func in agg_funcs:
                    # 结果列名：字段_窗口_函数（如mileage_7D_mean）
                    window_suffix = window.replace('D', 'd')
                    result_col = f"{col}_{window_suffix}_{func}"

                    # 基础滚动窗口
                    rolling = group.rolling(window, on=date_col)[col]

                    # 计算包含当前行的滚动统计量
                    if func == 'mean':
                        current_stats = rolling.mean()
                    elif func == 'sum':
                        current_stats = rolling.sum()
                    elif func == 'count':
                        current_stats = rolling.count()
                    elif func == 'max':
                        current_stats = rolling.max()
                    elif func == 'min':
                        current_stats = rolling.min()
                    elif func == 'std':
                        current_stats = rolling.std()

                    # 如果需要排除当前行
                    if exclude_current:
                        # 根据不同函数处理排除当前行的情况
                        # 计算窗口大小
                        window_size = rolling.count()
                        if func == 'mean':
                            # 均值: (总和 - 当前值) / (窗口大小 - 1)
                            group[result_col] = np.where(
                                window_size <= 1,
                                np.nan,
                                (current_stats * window_size - group[col]) / (window_size - 1)
                            )
                        elif func == 'sum':
                            # 总和: 总和 - 当前值
                            group[result_col] = np.where(
                                window_size <= 1,
                                np.nan,
                                current_stats - group[col]
                            )
                        elif func == 'count':
                            # 计数: 计数 - 1
                            group[result_col] = np.where(
                                window_size <= 1,
                                np.nan,
                                current_stats - 1
                            )
                        else:  # max, min, std
                            # 这些函数无法通过简单计算排除当前行，使用shift
                            # group[result_col] = current_stats.shift(1)
                            group[result_col] = current_stats
                    else:
                        # 不排除当前行，直接使用计算结果
                        group[result_col] = current_stats

            return group

        # 应用分组处理
        result_df = df_copy.groupby(group_keys, group_keys=False).apply(process_group)

        return result_df
    # 计算多个字段的多个统计量
    df = rolling_window_calculator(
        df,
        target_cols=['freeMileage','mileage','passengerTime','waitTime','money','CarryPassengerNum','IsGps', 'IsCamera', 'IsAeb'],
        agg_funcs=['mean'],
        window='8D',
        exclude_current=True
    )
    df = rolling_window_calculator(
        df,
        target_cols=['NegativeReview','adas违规', '报警', '提醒', '风控', '故障', '急加速次数',
           '急减速次数', '重度疲劳次数', '接打电话次数', '抽烟次数', '注意力分散次数', '斑马线未礼让行人次数', '肢体冲突次数',
           '斑马线超速次数', 'LDW', 'FCW', '毫米波制动', 'HMW', '超声波制动', '双目制动', '违章',
           '违规', 'over_speed_cnt'],
        agg_funcs=['sum'],
        window='8D',
        exclude_current=True
    )
    df = rolling_window_calculator(
        df,
        target_cols=['急加速次数','急减速次数'],
        agg_funcs=['std','mean'],
        window='8D',
        exclude_current=True
    )
    # 第二次分组填充（高效版）
    fields2 = ['freeMileage_8d_mean',
               'mileage_8d_mean', 'passengerTime_8d_mean', 'waitTime_8d_mean',
               'money_8d_mean', 'CarryPassengerNum_8d_mean', 'IsGps_8d_mean',
               'IsCamera_8d_mean', 'IsAeb_8d_mean',
               'NegativeReview_8d_sum', 'adas违规_8d_sum', '报警_8d_sum', '提醒_8d_sum',
               '风控_8d_sum', '故障_8d_sum', '急加速次数_8d_sum', '急减速次数_8d_sum',
               '重度疲劳次数_8d_sum', '接打电话次数_8d_sum', '抽烟次数_8d_sum', '注意力分散次数_8d_sum',
               '斑马线未礼让行人次数_8d_sum', '肢体冲突次数_8d_sum', '斑马线超速次数_8d_sum', 'LDW_8d_sum',
               'FCW_8d_sum', '毫米波制动_8d_sum', 'HMW_8d_sum', '超声波制动_8d_sum',
               '双目制动_8d_sum', '违章_8d_sum', '违规_8d_sum', 'over_speed_cnt_8d_sum',
               '急加速次数_8d_std', '急加速次数_8d_mean', '急减速次数_8d_std', '急减速次数_8d_mean']
    df = fill_group_na(df, fields2)
    ohe = OneHotEncoder(drop=None, sparse=False)
    date_type_encoded = ohe.fit_transform(df[['date_type']])
    date_type_cols = ohe.get_feature_names(['date_type'])
    date_type_df = pd.DataFrame(date_type_encoded, columns=date_type_cols, index=df.index)
    df = pd.concat([df.drop(columns=['date_type']), date_type_df],axis=1)
    df['里程'] = df['freeMileage_8d_mean'] + df['mileage_8d_mean']
    df['空转里程占比'] = df['freeMileage_8d_mean'] / df['里程']
    df['时长'] = df['passengerTime_8d_mean'] + df['waitTime_8d_mean']
    df['空转时长占比'] = df['waitTime_8d_mean'] / df['时长']
    df['分散注意力'] = df['接打电话次数_8d_sum'] + df['抽烟次数_8d_sum'] + df['注意力分散次数_8d_sum']
    df['危险驾驶'] = df['斑马线未礼让行人次数_8d_sum'] + df['肢体冲突次数_8d_sum'] + df['斑马线超速次数_8d_sum']
    col = ['totalmileage',
       'money_8d_mean', 'CarryPassengerNum_8d_mean', 'IsGps_8d_mean',
       'IsCamera_8d_mean', 'IsAeb_8d_mean',
       'NegativeReview_8d_sum', 'adas违规_8d_sum', '报警_8d_sum', '提醒_8d_sum',
       '风控_8d_sum', '故障_8d_sum', '急加速次数_8d_sum', '急减速次数_8d_sum',
       '重度疲劳次数_8d_sum', 'LDW_8d_sum',
       'FCW_8d_sum', '毫米波制动_8d_sum', 'HMW_8d_sum', '超声波制动_8d_sum',
       '双目制动_8d_sum', '违章_8d_sum', '违规_8d_sum', 'over_speed_cnt_8d_sum',
       '急加速次数_8d_std', '急加速次数_8d_mean', '急减速次数_8d_std', '急减速次数_8d_mean']
    sc = StandardScaler()  # 标准化
    df[col] = sc.fit_transform(df[col])
    df.dropna(how='any',subset=['DriverName'],inplace=True)

    df.to_csv('./DataPreprocessing.csv',index=None)
    end_time = time.time()
    print(f'数据预处理完成,总执行时间：{round(end_time - start_time, 2)} 秒')
    return df

df = sjycl()


def fit():
    start_time = time.time()
    # df = pd.read_csv('./DataPreprocessing.csv')
    df['AddDate'] = pd.to_datetime(df['AddDate'])
    df_train = df[df['AddDate'].between(pd.to_datetime('2025-08-29'), pd.to_datetime('2025-09-04')) ]  # 包含这两个日期
    del_cols = ['freeMileage', 'passengerTime', 'waitTime', 'mileage', 'money',
                'CarryPassengerNum', 'NegativeReview', 'IsGps', 'IsCamera', 'IsAeb',
                'adas违规', '报警', '提醒', '风控', '故障', '急加速次数', '急减速次数',
                '重度疲劳次数', '接打电话次数', '抽烟次数', '注意力分散次数', '斑马线未礼让行人次数', '肢体冲突次数',
                '斑马线超速次数', 'LDW', 'FCW', '毫米波制动', 'HMW', '超声波制动', '双目制动', '违章', '违规', '无制动',
                '无效预警',		
                'over_speed_cnt', 'prcp', 'AddDate', 'carNo', 'DriverName', 'DriverTypeID', 'driveAge', 'EntryAge',
                'certificateId','Certified',''
        , 'IDNumber', 'DriverID', 'DeviceID', 'TerminalEquipmentID', 'VehicleID', 'tmin', 'tmax'
        , 'wspd', 'pres', 'over_speed_avg', 'over_speed_max', '无制动', 'yichang', 'tavg'
        , '接打电话次数_8d_sum', '抽烟次数_8d_sum', '注意力分散次数_8d_sum'
        , '斑马线未礼让行人次数_8d_sum', '肢体冲突次数_8d_sum', '斑马线超速次数_8d_sum', 'freeMileage_8d_mean',
                'mileage_8d_mean', 'passengerTime_8d_mean', 'waitTime_8d_mean'
                ]
    existing_cols_to_drop = df.columns.intersection(del_cols)
    df_train = df_train.drop(columns=existing_cols_to_drop)
    df_train = df_train.dropna()
    # 将target为0的部分值替换为1
    def replace_zeros_with_ones(df, column_name, replace_ratio=0.3, random_seed=None):
        """
        带随机种子的版本
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        zero_indices = df.index[df[column_name] == 0].tolist()
        n_replace = int(len(zero_indices) * replace_ratio)
        replace_indices = np.random.choice(zero_indices, size=n_replace, replace=False)
        df.loc[replace_indices, column_name] = 1

        return df


    # 使用固定种子确保可重复性
    df_train = replace_zeros_with_ones(df_train, 'target', replace_ratio=0.1, random_seed=42)
    # 使用SMOTE进行过采样
    smote = SMOTE(random_state=42)
    X_train = df_train.loc[:,df_train.columns != 'target']
    y_train = df_train.loc[:,df_train.columns == 'target']
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    X, y =  X_resampled, y_resampled
    y = y.values.ravel()
    print(X.shape ,y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 定义模型及其参数网格
    models = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [15, 25, 35, 50, 100],
                "max_depth": [3, 5, 7, 11, None],
                "min_samples_split": [2, 3, 5]
            }
        },
        # "SVM": {
        #     "model": SVC(random_state=42),
        #     "params": {
        #         "C": [0.01,0.1, 1, 10,100],
        #         "gamma": ["scale", "auto"]
        #     }
        # },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42  ), # verbose=1：显示训练过程
            "params": {
                "C": [0.01,0.1, 1, 10],
                "penalty": ["l2"],
                # "solver": ["lbfgs", "newton-cg", "sag", "saga"]
                "solver": ["newton-cg", "sag", "saga"]
            }
        },
        # "KNN": {
        #     "model": KNeighborsClassifier(),
        #     "params": {
        #         "n_neighbors": [3, 5, 7],
        #         "weights": ["uniform", "distance"]
        #     }
        # },
        # "LightGBM": {
        #     "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
        #     "params": {
        #         "n_estimators": [50, 100, 200],
        #         "max_depth": [3, 5, 7, -1],  # -1表示无限制
        #         "learning_rate": [0.01, 0.05, 0.1],
        #         "num_leaves": [15, 31, 63],  # 控制树复杂度，防过拟合
        #         "min_child_samples": [10, 20, 30],  # 防过拟合
        #         "subsample": [0.7, 0.8, 0.9, 1.0],  # 行采样，防过拟合
        #         "colsample_bytree": [0.7, 0.8, 0.9, 1.0],  # 列采样，防过拟合
        #         "reg_alpha": [0, 0.1, 0.5],  # L1正则化，防过拟合
        #         "reg_lambda": [0, 0.1, 0.5],  # L2正则化，防过拟合
        #         "min_split_gain": [0, 0.1, 0.2]  # 分裂最小增益，防过拟合
        #     }
        # },
        "LightGBM": {
            "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
            "params": {
                "n_estimators": [50 ],
                "max_depth": [3, 5],  # -1表示无限制
                "learning_rate": [0.01, 0.05],
                "num_leaves": [15, 31],  # 控制树复杂度，防过拟合
                "min_child_samples": [10, 20],  # 防过拟合
                "subsample": [0.7, 0.8],  # 行采样，防过拟合
                "colsample_bytree": [0.7, 0.8],  # 列采样，防过拟合
                "reg_alpha": [ 0.1],  # L1正则化，防过拟合
                "reg_lambda": [0.1],  # L2正则化，防过拟合
                "min_split_gain": [ 0.1]  # 分裂最小增益，防过拟合
            }
        },
        # "XGBoost": {
        #     "model": xgb.XGBClassifier(random_state=42, eval_metric='logloss',use_label_encoder=False),
        #     "params": {
        #         "n_estimators": [50, 100, 200],
        #         "max_depth": [3, 5, 7, 10],
        #         "learning_rate": [0.01, 0.05, 0.1],
        #         "min_child_weight": [1, 3, 5],  # 防过拟合
        #         "gamma": [0, 0.1, 0.2],  # 最小分裂损失，防过拟合
        #         "subsample": [0.7, 0.8, 0.9, 1.0],  # 行采样，防过拟合
        #         "colsample_bytree": [0.7, 0.8, 0.9, 1.0],  # 列采样，防过拟合
        #         "colsample_bylevel": [0.7, 0.8, 0.9, 1.0],  # 每层级列采样，防过拟合
        #         "reg_alpha": [0, 0.1, 0.5],  # L1正则化，防过拟合
        #         "reg_lambda": [0, 0.1, 0.5, 1.0],  # L2正则化，防过拟合
        #         "scale_pos_weight": [1, 3, 5]  # 处理类别不平衡，防过拟合
        #     }
        # },
        "XGBoost": {
            "model": xgb.XGBClassifier(random_state=42, eval_metric='logloss',use_label_encoder=False),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [3, 5],
                "learning_rate": [0.01, 0.05],
                "min_child_weight": [3, 5],  # 防过拟合
                "gamma": [0.1],  # 最小分裂损失，防过拟合
                "subsample": [0.7, 0.8],  # 行采样，防过拟合
                "colsample_bytree": [0.7, 0.8],  # 列采样，防过拟合
                "colsample_bylevel": [0.7, 0.8],  # 每层级列采样，防过拟合
                "reg_alpha": [0.1, 0.5],  # L1正则化，防过拟合
                "reg_lambda": [0.1, 0.5],  # L2正则化，防过拟合
                "scale_pos_weight": [3, 5]  # 处理类别不平衡，防过拟合
            }
        },
        # "GBDT": {
        #     "model": GradientBoostingClassifier(random_state=42),
        #     "params": {
        #         "n_estimators": [20, 30, 40],
        #         "max_depth": [3, 5, 7],
        #         "min_samples_split": range(100,301,100),
        #         "min_samples_leaf": range(10,51,20)
        #     }
        # }
    }
    # 执行网格搜索与交叉验证
    results = []
    best_models = {}

    for model_name, config in models.items():
        # 创建GridSearchCV对象
        grid = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=3,  # 3折交叉验证
            scoring="accuracy",
            n_jobs = 8,  # 使用所有CPU核心加速
            verbose=1
        )
        grid.fit(X_train, y_train)

        # 保存结果
        best_model = grid.best_estimator_
        test_acc = accuracy_score(y_test, best_model.predict(X_test))

        results.append({
            "Model": model_name,
            "Best CV Score": grid.best_score_,
            "Test Accuracy": test_acc,
            "Best Parameters": grid.best_params_
        })

        best_models[model_name] = best_model

    joblib.dump(best_models["XGBoost"], './model.joblib')
    joblib.dump(best_models, './all_model.joblib')
    end_time = time.time()
    print(f"训练完成时间：{round(end_time - start_time, 2)} 秒")
    return X, y

fit()

def predict():

    start_time = time.time()
    # df = pd.read_csv('./DataPreprocessing.csv')
    df['AddDate'] = pd.to_datetime(df['AddDate'])
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    # today = pd.to_datetime(today)
    yesterday = pd.to_datetime(yesterday)
    predict = df[df['AddDate']==yesterday]
    # predict = df[df['AddDate']=='2025-08-15']
    del_cols = ['freeMileage', 'passengerTime', 'waitTime', 'mileage', 'money',
                'CarryPassengerNum', 'NegativeReview', 'IsGps', 'IsCamera', 'IsAeb',
                'adas违规', '报警', '提醒', '风控', '故障', '急加速次数', '急减速次数',
                '重度疲劳次数', '接打电话次数', '抽烟次数', '注意力分散次数', '斑马线未礼让行人次数', '肢体冲突次数',
                '斑马线超速次数', 'LDW', 'FCW', '毫米波制动', 'HMW', '超声波制动', '双目制动', '违章', '违规', '无制动',
                '无效预警',		
                'over_speed_cnt', 'prcp', 'AddDate', 'carNo', 'DriverName', 'DriverTypeID', 'driveAge', 'EntryAge',
                'certificateId','Certified',''
        , 'IDNumber', 'DriverID', 'DeviceID', 'TerminalEquipmentID', 'VehicleID', 'tmin', 'tmax'
        , 'wspd', 'pres', 'over_speed_avg', 'over_speed_max', '无制动', 'yichang', 'tavg'
        , '接打电话次数_8d_sum', '抽烟次数_8d_sum', '注意力分散次数_8d_sum'
        , '斑马线未礼让行人次数_8d_sum', '肢体冲突次数_8d_sum', '斑马线超速次数_8d_sum', 'freeMileage_8d_mean',
                'mileage_8d_mean', 'passengerTime_8d_mean', 'waitTime_8d_mean'
                ]
    existing_cols_to_drop = df.columns.intersection(del_cols)
    df.drop(columns=existing_cols_to_drop,inplace=True)
    X = df.loc[:, df.columns != 'target']
    y = df.loc[:, df.columns == 'target']
    # 定义列的分类
    id_cols = ["AddDate", "carNo", "DriverName"]  # 标识符列（不参与训练）
    feature_cols = X.columns.to_list() # 训练特征列
    label_col = y.columns.to_list()  # 标签列
    # 分离数据
    df_ids = predict[id_cols].copy()  # 单独存储标识符（后续用于结果关联）
    X = predict[feature_cols].copy()  # 模型输入特征
    y = predict[label_col].copy()     # 模型标签
    # 加载已保存的模型
    loaded_model = joblib.load("./model.joblib")
    print("\n已加载模型：", loaded_model)
    y_pred_proba = loaded_model.predict_proba(X)

    # 生成预测结果（概率值和类别）
    y_pred = loaded_model.predict(X)  # 预测风险等级（0/1）
    y_pred_proba = loaded_model.predict_proba(X)[:, 1]  # 预测为高风险的概率

    # 重置所有参与拼接对象的索引（避免索引对齐导致行数变化）
    df_ids = df_ids.reset_index(drop=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_pred_series = pd.Series(y_pred, name="预测风险等级").reset_index(drop=True)
    y_pred_proba_series = pd.Series(y_pred_proba, name="高风险概率").reset_index(drop=True)

    # 合并所有信息：标识符 + 原始特征 + 真实标签 + 预测结果
    result_df = pd.concat([
        df_ids,          # 标识符（车牌号、日期、司机姓名）
        X,               # 原始特征（可选，用于排查异常）
        # y,  # 真实标签
        pd.Series(y_pred, name="预测风险等级"),  # 预测类别
        pd.Series(y_pred_proba, name="高风险概率")  # 预测概率
    ], axis=1)

    # 输出最终结果
    print("\n带标识符的训练与预测结果：")
    print(result_df)
    print(len(df_ids)==len(result_df))
    print(len(df_ids),len(X),len(y),len(y_pred),len(y_pred_proba),len(result_df))
    # （可选）保存结果到Excel，方便后续分析
    result_df.to_excel("./driver_risk_result_with_ids.xlsx", index=False)

    # 2. 构建 SQLAlchemy 连接引擎（格式：mysql+pymysql://用户:密码@主机:端口/数据库）
    engine_url = f"mysql+pymysql://{TARGET_CONFIG['user']}:{TARGET_CONFIG['password']}@{TARGET_CONFIG['host']}:{TARGET_CONFIG['port']}/{TARGET_CONFIG['database']}"
    engine = create_engine(engine_url)

    # 3. 定义写入参数（根据需求调整）
    table_name = "predictData"  # 目标表名（可自定义，如：司机车辆风险模型数据）
    if_exists = "replace"  # 表存在时的处理：replace=覆盖，append=追加，fail=报错（推荐首次用replace，后续用append）
    chunksize = 10000  # 分批写入（避免数据量过大时内存溢出，建议1-5万行/批）
    dtype = None  # 若需指定字段类型（如日期、字符串），可在这里定义（示例见下方说明）
    index = False  # 是否将 pandas 的索引作为表的一列（建议False，避免多余列）

    # 4. 执行数据写入
    try:
        print(f"开始写入数据到 MySQL 表：{TARGET_CONFIG['database']}.{table_name}")
        result_df.to_sql(
            name=table_name,
            con=engine,
            if_exists=if_exists,
            chunksize=chunksize,
            dtype=dtype,
            index=index
        )
        print(f"数据写入成功！共写入 {len(result_df)} 行数据")
    except Exception as e:
        print(f"数据写入失败！错误信息：{str(e)}")
    finally:
        # 关闭数据库连接（释放资源）
        engine.dispose()
        print("数据库连接已关闭")
    end_time = time.time()
    print(f"输出结果时间：{round(end_time - start_time, 2)} 秒")

predict()