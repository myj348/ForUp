import pandas as pd
import os

def read_mt_evaluation():
    """读取机器翻译评估数据"""
    try:
        # 读取Excel文件
        file_path = 'data/机器翻译效果评估.xlsx'
        excel_file = pd.ExcelFile(file_path)
        
        # 过滤掉 Sheet1
        sheet_names = [sheet for sheet in excel_file.sheet_names if sheet != 'Sheet1']
        
        # 显示基本信息
        print(f"\n找到 {len(sheet_names)} 个数据表:")
        print("Sheet names:", sheet_names)
        
        # 读取每个sheet的数据
        data_dict = {}
        for sheet in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet)
            data_dict[sheet] = df
            
            # 显示每个sheet的基本信息
            print(f"\n表格: {sheet}")
            print(f"形状: {df.shape}")
            print(f"列名: {df.columns.tolist()}")
            print(f"前5行数据示例:")
            print(df.head())
            
        return data_dict
            
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return None
    except Exception as e:
        print(f"读取文件时出错: {str(e)}")
        return None

if __name__ == "__main__":
    # 确保在正确的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 读取数据
    data = read_mt_evaluation()
    
    if data:
        print("\n数据读取成功!")
        print(f"共读取了 {len(data)} 个翻译方向的数据") 