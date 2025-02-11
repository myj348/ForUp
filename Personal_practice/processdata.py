import streamlit as st
import pandas as pd
import os

def load_data():
    """加载Excel数据"""
    try:
        # 获取当前文件所在目录的父目录（ForUP）
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 构建Excel文件的完整路径
        file_path = os.path.join(parent_dir, 'Data', 'gansuSK.xlsx')
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            st.error(f"文件不存在，请检查路径: {file_path}")
            return None
            
        # 读取Excel文件的所有sheet，跳过前两行
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        # 创建一个字典来存储所有sheet的数据
        data_dict = {}
        for sheet in sheet_names:
            # skiprows=2 表示跳过前两行，header=0 表示使用第一行（实际是第三行）作为列名
            data_dict[sheet] = pd.read_excel(excel_file, 
                                           sheet_name=sheet, 
                                           skiprows=2,  # 跳过前两行
                                           header=0)    # 使用第一行（第三行）作为列名
            
        return data_dict
        
    except Exception as e:
        st.error(f"数据加载失败，错误信息: {str(e)}")
        st.error(f"错误类型: {type(e).__name__}")
        return None

def main():
    st.title('Excel数据查看器')
    
    # 加载数据
    data_dict = load_data()
    if not data_dict:
        return
    
    # 数据集选择
    sheet_name = st.selectbox(
        '选择要查看的Sheet',
        list(data_dict.keys())
    )
    
    # 获取选中的数据集
    df = data_dict[sheet_name]
    
    # 显示列名
    st.subheader(f'{sheet_name} 的列名:')
    st.write(list(df.columns))
    
    # 显示数据预览
    st.subheader('数据预览:')
    st.dataframe(df.head())
    
    # 显示数据基本信息
    st.subheader('数据基本信息:')
    st.write(f'行数: {df.shape[0]}')
    st.write(f'列数: {df.shape[1]}')

if __name__ == '__main__':
    main()