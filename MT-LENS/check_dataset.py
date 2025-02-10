import pandas as pd
import os

def check_dataset(file_path: str):
    print(f"检查数据集: {file_path}")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print("❌ 错误：文件不存在！")
        return False
        
    try:
        # 读取Excel文件
        df = pd.read_excel(file_path)
        
        # 检查必要的列
        required_columns = ['source', 'reference']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 错误：缺少必要的列: {missing_columns}")
            print(f"当前列名: {df.columns.tolist()}")
            return False
            
        # 显示数据集基本信息
        print("\n📊 数据集信息:")
        print(f"总行数: {len(df)}")
        print(f"列名: {df.columns.tolist()}")
        
        # 检查空值
        null_counts = df[required_columns].isnull().sum()
        if null_counts.sum() > 0:
            print("\n⚠️ 警告：存在空值:")
            print(null_counts)
            
        # 显示前几行数据作为样例
        print("\n📝 数据样例 (前3行):")
        print(df[required_columns].head(3))
        
        # 显示文本长度统计
        print("\n📏 文本长度统计:")
        df['source_length'] = df['source'].str.len()
        df['reference_length'] = df['reference'].str.len()
        
        print("源文本长度:")
        print(f"- 最短: {df['source_length'].min()}")
        print(f"- 最长: {df['source_length'].max()}")
        print(f"- 平均: {df['source_length'].mean():.1f}")
        
        print("\n译文长度:")
        print(f"- 最短: {df['reference_length'].min()}")
        print(f"- 最长: {df['reference_length'].max()}")
        print(f"- 平均: {df['reference_length'].mean():.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误：读取文件时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 检查数据集
    dataset_path = os.path.join("data", "zh_en_test.xlsx")
    check_dataset(dataset_path) 