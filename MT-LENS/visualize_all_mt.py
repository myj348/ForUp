import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import numpy as np
from scipy import stats

def load_data():
    """加载数据并进行预处理"""
    file_path = 'data/机器翻译效果评估.xlsx'
    
    sheet_names = {
        'Chn2Eng': 'Chn2Eng',
        'Thai2Eng': 'Thai2Eng',
        'Eng2Indi': 'Eng2Indi',
        'Indi2Eng': 'Indi2Eng',
        'Eng2Indo': 'Eng2Indo'
    }
    
    data_dict = {}
    for key, sheet_name in sheet_names.items():
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df['序号'] = range(1, len(df) + 1)
        data_dict[key] = df
    
    return data_dict

def analyze_metrics_consistency(df, selected_metrics):
    """分析指标与 Gemini2.0 Score 的一致性"""
    if 'Gemini2.0 Score' not in df.columns:
        return None
    
    gemini_scores = df['Gemini2.0 Score']
    consistency_results = []
    
    for metric in selected_metrics:
        if metric != 'Gemini2.0 Score' and metric in df.columns:
            correlation, _ = stats.pearsonr(gemini_scores, df[metric])
            consistency_results.append({
                'metric': metric,
                'correlation': abs(correlation),
                'original_correlation': correlation
            })
    
    # 按相关性绝对值排序
    consistency_results.sort(key=lambda x: x['correlation'], reverse=True)
    
    if consistency_results:
        most_consistent = consistency_results[0]
        least_consistent = consistency_results[-1]
        return most_consistent, least_consistent
    return None

def plot_metrics(df, selected_metrics):
    """绘制指标折线图"""
    fig = go.Figure()
    
    # 始终添加 Gemini2.0 Score
    if 'Gemini2.0 Score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['序号'],
            y=df['Gemini2.0 Score'],
            name='Gemini2.0 Score',
            line=dict(width=3)
        ))
    
    # 添加选中的其他指标
    for metric in selected_metrics:
        if metric in df.columns and metric != 'Gemini2.0 Score':
            fig.add_trace(go.Scatter(
                x=df['序号'],
                y=df[metric],
                name=metric,
                line=dict(width=1)
            ))
    
    fig.update_layout(
        title='翻译评估指标对比',
        xaxis_title='序号',
        yaxis_title='分数',
        hovermode='x unified',
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig


def calculate_statistics(df, selected_metrics):
    """计算基本统计量"""
    stats_results = {}
    
    for metric in selected_metrics:
        if metric in df.columns:
            values = df[metric].values
            stats_results[metric] = {
                '均值': np.mean(values),
                '中位数': np.median(values),
                '标准差': np.std(values),
                '方差': np.var(values),
                '极差': np.ptp(values),
                '25%分位数': np.percentile(values, 25),
                '75%分位数': np.percentile(values, 75),
                '最小值': np.min(values),
                '最大值': np.max(values)
            }
    
    return stats_results

def analyze_threshold(df, selected_metrics, threshold=0.8):
    """阈值分析"""
    threshold_results = {}
    
    for metric in selected_metrics:
        if metric in df.columns:
            values = df[metric].values
            above_threshold = np.sum(values >= threshold)
            threshold_results[metric] = {
                '高于阈值的数量': above_threshold,
                '高于阈值的比例': above_threshold / len(values)
            }
    
    return threshold_results

def analyze_quartiles(df, selected_metrics):
    """分位数分组分析"""
    quartile_results = {}
    
    for metric in selected_metrics:
        if metric in df.columns:
            values = df[metric].values
            q1, q2, q3 = np.percentile(values, [25, 50, 75])
            quartile_results[metric] = {
                '高分组(前25%)': np.sum(values > q3),
                '中分组(25%-75%)': np.sum((values >= q1) & (values <= q3)),
                '低分组(后25%)': np.sum(values < q1),
                '各组占比': {
                    '高分组': np.sum(values > q3) / len(values),
                    '中分组': np.sum((values >= q1) & (values <= q3)) / len(values),
                    '低分组': np.sum(values < q1) / len(values)
                }
            }
    
    return quartile_results




def main():
    st.set_page_config(layout="wide")
    st.title('机器翻译评估可视化')
    
    try:
        # 加载数据
        data_dict = load_data()
        
        # 数据集选择
        dataset = st.selectbox(
            '选择数据集',
            list(data_dict.keys()),
            format_func=lambda x: f"{x} 数据集"
        )
        
        # 获取选中的数据集
        df = data_dict[dataset]
        
        # 定义所有可选的指标
        all_metrics = [
            'MT-IENS BLEU', 'MT-IENS TRE', 'MT-IENS CHRF', 
            'MT-IENS COMET', 'MT-IENS BLEURT', 'MT-IENS METEOR', 
            'ChatGPT error stat', 'BLEU','Prompt Template', 'xComet'
        ]
        
        # 创建指标选择的多选框
        col1, col2 = st.columns([3, 1])
        with col2:
            st.write("选择要显示的指标：")
            selected_metrics = ['Gemini2.0 Score']  # 默认始终包含
            for metric in all_metrics:
                if metric in df.columns:
                    if st.checkbox(metric, value=False):
                        selected_metrics.append(metric)
            
            # 显示一致性分析结果
            if len(selected_metrics) > 2:  # 至少需要两个其他指标才能比较
                st.write("---")
                st.write("指标一致性分析：")
                
                consistency_result = analyze_metrics_consistency(df, selected_metrics)
                if consistency_result:
                    most_consistent, least_consistent = consistency_result
                    
                    st.write("最一致的指标：")
                    st.write(f"📈 {most_consistent['metric']} (相关系数: {most_consistent['original_correlation']:.3f})")
                    
                    st.write("最不一致的指标：")
                    st.write(f"📉 {least_consistent['metric']} (相关系数: {least_consistent['original_correlation']:.3f})")
        
        
        # 显示基本统计量
            if len(selected_metrics) > 0:
                st.write("---")
                st.write("基本统计分析：")
                stats_results = calculate_statistics(df, selected_metrics)
                for metric, stats in stats_results.items():
                    st.write(f"**{metric}**")
                    for stat_name, value in stats.items():
                        st.write(f"{stat_name}: {value:.4f}")
                
                # 显示阈值分析
                st.write("---")
                st.write("阈值分析 (阈值=0.8)：")
                threshold_results = analyze_threshold(df, selected_metrics)
                for metric, results in threshold_results.items():
                    st.write(f"**{metric}**")
                    st.write(f"高于阈值的数量: {results['高于阈值的数量']}")
                    st.write(f"高于阈值的比例: {results['高于阈值的比例']:.2%}")
                
                # 显示分位数分组分析
                st.write("---")
                st.write("分位数分组分析：")
                quartile_results = analyze_quartiles(df, selected_metrics)
                for metric, results in quartile_results.items():
                    st.write(f"**{metric}**")
                    st.write("分组数量：")
                    st.write(f"高分组: {results['高分组(前25%)']}")
                    st.write(f"中分组: {results['中分组(25%-75%)']}")
                    st.write(f"低分组: {results['低分组(后25%)']}")
                    
                    # 使用进度条显示各组占比
                    st.write("各组占比：")
                    st.progress(results['各组占比']['高分组'])
                    st.write(f"高分组: {results['各组占比']['高分组']:.2%}")
                    st.progress(results['各组占比']['中分组'])
                    st.write(f"中分组: {results['各组占比']['中分组']:.2%}")
                    st.progress(results['各组占比']['低分组'])
                    st.write(f"低分组: {results['各组占比']['低分组']:.2%}")
        
        # 在主列中显示图表
        with col1:
            st.plotly_chart(plot_metrics(df, selected_metrics), use_container_width=True)
        
        # 添加序号选择器和对应的文本显示
        selected_index = st.selectbox('选择序号查看详细信息', df['序号'])
        
        # 显示选中序号的源文本和翻译结果
        selected_row = df[df['序号'] == selected_index].iloc[0]
        st.subheader('详细信息')
        text_col1, text_col2 = st.columns(2)
        with text_col1:
            st.text_area('Test Case Source Language', selected_row['Test Case Source Language'], height=150)
        with text_col2:
            st.text_area('Machine Translate Result', selected_row['Machine Translate Result'], height=150)
            
    except Exception as e:
        st.error(f"发生错误: {str(e)}")
        st.write("错误详情:", e)

if __name__ == '__main__':
    main() 