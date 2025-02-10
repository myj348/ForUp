import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List
import numpy as np
import seaborn as sns
from scipy import stats


@st.cache_data
def create_correlation_matrix(sample_scores: List[Dict], selected_metrics: List[str]) -> tuple:
    """创建相关性矩阵"""
    # 提取所有指标的分数
    scores_dict = {metric: [] for metric in selected_metrics}
    
    # 首先检查每个样本中有哪些指标
    available_metrics = set()
    for score in sample_scores:
        metrics_in_sample = set(score.keys()) & set(selected_metrics)
        available_metrics.update(metrics_in_sample)
    
    # 只使用所有样本都有的指标
    common_metrics = [m for m in selected_metrics if m in available_metrics]
    
    # 重新收集数据
    valid_samples = []
    for score in sample_scores:
        # 检查这个样本是否包含所有需要的指标
        if all(metric in score for metric in common_metrics):
            sample_data = {metric: score[metric] for metric in common_metrics}
            valid_samples.append(sample_data)
    
    # 创建数据框
    if not valid_samples:
        st.warning("没有找到包含所有选定指标的样本")
        return None, None
    
    df = pd.DataFrame(valid_samples)
    
    # 计算相关性矩阵
    correlation_matrix = df.corr()
    
    # 创建热力图
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        text=np.round(correlation_matrix, 3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu',
        zmid=0
    ))
    
    fig.update_layout(
        title='指标相关性热力图',
        width=700,
        height=700
    )
    
    # 添加缺失指标的说明
    missing_metrics = set(selected_metrics) - set(common_metrics)
    if missing_metrics:
        st.warning(f"以下指标在部分样本中缺失，未包含在相关性分析中：{', '.join(missing_metrics)}")
    
    return fig, correlation_matrix

@st.cache_data
def create_scatter_plot(sample_scores: List[Dict], metric1: str, metric2: str) -> go.Figure:
    """创建两个指标的散点图"""
    # 提取两个指标的分数
    scores = [(score[metric1], score[metric2]) 
             for score in sample_scores 
             if metric1 in score and metric2 in score]
    
    x, y = zip(*scores)
    
    # 计算相关系数
    correlation, p_value = stats.pearsonr(x, y)
    
    # 创建散点图
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y,
        mode='markers',
        marker=dict(
            size=8,
            opacity=0.6
        )
    ))
    
    fig.update_layout(
        title=f'{metric1.upper()} vs {metric2.upper()}<br>相关系数: {correlation:.3f} (p-value: {p_value:.3e})',
        xaxis_title=metric1.upper(),
        yaxis_title=metric2.upper(),
        width=600,
        height=400
    )
    
    return fig

@st.cache_data
def load_results(file_path: str) -> Dict:
    """加载评估结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        return None

@st.cache_data
def create_metrics_comparison(corpus_metrics: Dict, selected_metrics: List[str]) -> go.Figure:
    """创建指标对比图"""
    # 只选择指定的指标
    metrics = {k: v for k, v in corpus_metrics.items() if k in selected_metrics and v is not None}
    
    if not metrics:
        st.warning("没有找到有效的指标数据")
        return None

    fig = go.Figure(data=[
        go.Bar(
            x=list(metrics.keys()),
            y=list(metrics.values()),
            text=[f'{v:.2f}' for v in metrics.values()],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='各评估指标对比',
        xaxis_title='评估指标',
        yaxis_title='得分',
        yaxis_range=[0, max(metrics.values()) * 1.1]
    )
    
    return fig

@st.cache_data
def create_score_distribution(sample_scores: List[Dict], metric: str) -> go.Figure:
    """创建指标分布图"""
    scores = [score[metric] for score in sample_scores if metric in score and score[metric] is not None]

    if not scores:
        st.warning(f"没有找到 {metric} 的有效数据")
        return None
    
    fig = go.Figure(data=[
        go.Histogram(
            x=scores,
            nbinsx=30,
            name=metric.upper()
        )
    ])
    
    fig.update_layout(
        title=f'{metric.upper()} 分数分布',
        xaxis_title='分数',
        yaxis_title='样本数量'
    )
    
    return fig

def main():
    # 指定要显示的指标
    SELECTED_METRICS = [
        'bleu',
        'chrf',
        'ter',
        'meteor',
        'rouge1',
        'rouge2',
        'rougeL',
        'comet',
        'bleurt'
    ]
    
    # 指标说明
    METRICS_DESCRIPTION = {
        'bleu': 'BLEU score (0-100, 越高越好)',
        'chrf': 'chrF score (0-100, 越高越好)',
        'ter': 'Translation Edit Rate (0-100, 越低越好)',
        'meteor': 'METEOR score (0-1, 越高越好)',
        'rouge1': 'ROUGE-1 F1 score (0-1, 越高越好)',
        'rouge2': 'ROUGE-2 F1 score (0-1, 越高越好)',
        'rougeL': 'ROUGE-L F1 score (0-1, 越高越好)',
        'comet': 'COMET segment-level score (0-1, 越高越好, 基于人工评分标准,wmt22-comet-da)',
        'bleurt': 'BLEURT score (基于BERT的评估指标, 通常在-1到1之间, 越高越好)'
    }
    
    st.set_page_config(
        page_title="机器翻译评估结果分析",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("机器翻译评估结果分析 📊")

    # 添加数据集选择
    datasets = {
        "中英翻译": "results/comprehensive_evaluation_resultsZE.json",
        "泰英翻译": "results/comprehensive_evaluation_resultsTE.json",
        "英印地翻译": "results/comprehensive_evaluation_resultsEID.json",
        "印地英翻译": "results/comprehensive_evaluation_resultsIDE.json",
        "英印尼翻译": "results/comprehensive_evaluation_resultsEINI.json",
        
    }

    selected_dataset = st.sidebar.selectbox(
        "选择数据集",
        list(datasets.keys()),
        format_func=lambda x: x
    )

    results_path = datasets[selected_dataset]
    

    with st.spinner('正在加载数据...'):
        results = load_results(results_path)
        
        if results is None:
            st.error(f"无法找到或加载结果文件: {results_path}")
            st.stop()
    
    corpus_metrics = results['corpus_metrics']
    sample_scores = results['sample_scores']

    # 首先计算可用指标
    available_metrics = set()
    for score in sample_scores:
        metrics_in_sample = set(score.keys()) & set(SELECTED_METRICS)
        available_metrics.update(metrics_in_sample)
    
    missing_metrics = set(SELECTED_METRICS) - available_metrics
    if missing_metrics:
        st.warning(f"以下指标在数据中完全缺失: {', '.join(missing_metrics)}")
    
    partial_metrics = set()
    for metric in available_metrics:
        count = sum(1 for score in sample_scores if metric in score)
        if count < len(sample_scores):
            partial_metrics.add(metric)
    
    if partial_metrics:
        st.warning(f"以下指标在部分样本中缺失: {', '.join(partial_metrics)}")

    # 需要归一化的指标及其处理方法
    normalize_config = {
        'bleu': {'scale': 100, 'type': 'direct'},  # 除以100
        'chrf': {'scale': 100, 'type': 'direct'},  # 除以100
        'ter': {'scale': 100, 'type': 'inverse'},  # 1 - (值/100)
        'bleurt': {'type': 'minmax'}  # 最小最大值归一化
    }
    
    # 对corpus_metrics进行归一化
    for metric, config in normalize_config.items():
        if metric in corpus_metrics:
            if config['type'] == 'direct':
                corpus_metrics[metric] = corpus_metrics[metric] / config['scale']
            elif config['type'] == 'inverse':
                corpus_metrics[metric] = 1 - (corpus_metrics[metric] / config['scale'])
    
    # 对sample_scores进行归一化
    bleurt_range = None
    if 'bleurt' in available_metrics:
        bleurt_scores = [score['bleurt'] for score in sample_scores if 'bleurt' in score]
        if bleurt_scores:
            bleurt_min = min(bleurt_scores)
            bleurt_max = max(bleurt_scores)
            bleurt_range = bleurt_max - bleurt_min
    
    for score in sample_scores:
        for metric, config in normalize_config.items():
            if metric in score:
                if config['type'] == 'direct':
                    score[metric] = score[metric] / config['scale']
                elif config['type'] == 'inverse':
                    score[metric] = 1 - (score[metric] / config['scale'])
                elif config['type'] == 'minmax' and bleurt_range and bleurt_range > 0:
                    score[metric] = (score[metric] - bleurt_min) / bleurt_range
    
    # 更新指标描述
    METRICS_DESCRIPTION.update({
        'bleu': 'BLEU score (0-1, 越高越好)',
        'chrf': 'chrF score (0-1, 越高越好)',
        'ter': 'Translation Edit Rate (0-1, 越高越好)',
        'bleurt': 'BLEURT score (0-1, 越高越好, 已归一化)'
    })


    available_metrics = set()
    for score in sample_scores:
        metrics_in_sample = set(score.keys()) & set(SELECTED_METRICS)
        available_metrics.update(metrics_in_sample)
    
    missing_metrics = set(SELECTED_METRICS) - available_metrics
    if missing_metrics:
        st.warning(f"以下指标在数据中完全缺失: {', '.join(missing_metrics)}")
    
    partial_metrics = set()
    for metric in available_metrics:
        count = sum(1 for score in sample_scores if metric in score)
        if count < len(sample_scores):
            partial_metrics.add(metric)
    
    if partial_metrics:
        st.warning(f"以下指标在部分样本中缺失: {', '.join(partial_metrics)}")
    
    st.sidebar.title("导航")
    page = st.sidebar.radio(
        "选择查看内容",
        ["整体评估结果", "指标分布分析", "指标相关性分析", "样本详情"]
    )
    
    if page == "整体评估结果":
        st.header("整体评估结果")
        
        # 显示COMET系统级别得分
        if 'comet_system' in corpus_metrics:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric(
                    "COMET System Score",
                    f"{corpus_metrics['comet_system']:.4f}"
                )
        
        with st.spinner('生成对比图...'):
            fig = create_metrics_comparison(corpus_metrics, SELECTED_METRICS)
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("详细指标数值")
        metrics_df = pd.DataFrame({
            "指标": [m for m in SELECTED_METRICS if m in corpus_metrics],
            "得分": [corpus_metrics[m] for m in SELECTED_METRICS if m in corpus_metrics]
        })
        st.dataframe(metrics_df.round(4))
        
    elif page == "指标分布分析":
        st.header("指标分布分析")
        
        selected_metric = st.selectbox(
            "选择要查看的指标",
            SELECTED_METRICS,
            format_func=lambda x: x.upper()
        )
        
        with st.spinner('生成分布图...'):
            fig = create_score_distribution(sample_scores, selected_metric)
            st.plotly_chart(fig, use_container_width=True)
            
        scores = [score[selected_metric] for score in sample_scores if selected_metric in score]
        if scores:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("基本统计信息")
                stats_df = pd.DataFrame({
                    "统计量": ["平均值", "中位数", "最大值", "最小值", "标准差"],
                    "值": [
                        np.mean(scores),
                        np.median(scores),
                        np.max(scores),
                        np.min(scores),
                        np.std(scores)
                    ]
                })
                st.dataframe(stats_df.round(4))
            
            with col2:
                st.subheader("分位数信息")
                quantiles = np.percentile(scores, [25, 50, 75])
                quantiles_df = pd.DataFrame({
                    "分位数": ["25%", "50%", "75%"],
                    "值": quantiles
                })
                st.dataframe(quantiles_df.round(4))
    
    elif page == "指标相关性分析":
        st.header("指标相关性分析")
        
        # 显示相关性热力图
        st.subheader("1. 相关性热力图")
        with st.spinner('生成相关性热力图...'):
            result = create_correlation_matrix(sample_scores, SELECTED_METRICS)
            if result is not None:
                corr_fig, corr_matrix = result
                if corr_fig is not None:
                    st.plotly_chart(corr_fig, use_container_width=True)

                    st.info("""
                    注意：BLEURT 分数的范围与其他指标不同，但相关性分析已经考虑了这一点。
                    相关系数反映的是变化趋势的一致性，而不是绝对值的大小。
                    """)

                    st.subheader("指标相关性分析摘要")
                    
                    # 获取相关系数矩阵的上三角部分（不包括对角线）
                    upper_triangle = np.triu(corr_matrix, k=1)
                    
                    # 找出最相关的两个指标
                    max_corr_idx = np.unravel_index(np.argmax(np.abs(upper_triangle)), upper_triangle.shape)
                    max_corr = upper_triangle[max_corr_idx]
                    metric1, metric2 = corr_matrix.index[max_corr_idx[0]], corr_matrix.columns[max_corr_idx[1]]
                    
                    # 找出最不相关的两个指标
                    # 将0替换为nan以避免选中没有计算相关系数的指标对
                    upper_triangle_no_zeros = np.where(upper_triangle != 0, upper_triangle, np.nan)
                    min_corr_idx = np.unravel_index(np.nanargmin(np.abs(upper_triangle_no_zeros)), upper_triangle.shape)
                    min_corr = upper_triangle[min_corr_idx]
                    metric3, metric4 = corr_matrix.index[min_corr_idx[0]], corr_matrix.columns[min_corr_idx[1]]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**最相关的指标对：**")
                        st.markdown(f"**{metric1.upper()}** 和 **{metric2.upper()}**")
                        st.markdown(f"相关系数: {max_corr:.3f}")
                        if abs(max_corr) > 0.7:
                            st.markdown("👉 这两个指标具有很强的相关性，可能在评估相似的翻译特征。")
                    
                    with col2:
                        st.markdown("**最不相关的指标对：**")
                        st.markdown(f"**{metric3.upper()}** 和 **{metric4.upper()}**")
                        st.markdown(f"相关系数: {min_corr:.3f}")
                        st.markdown("👉 这两个指标可能在评估不同的翻译特征，组合使用可能更全面。")

                    st.info("""
                    💡 提示：
                    - 高相关性（>0.7）表示两个指标可能在评估相似的翻译特征
                    - 低相关性（<0.3）表示两个指标可能在评估不同的翻译特征
                    - 在实际评估中，建议选择相关性较低的指标组合，以获得更全面的评估
                    """)


        
        # 显示具体两个指标的相关性分析
        st.subheader("2. 指标对比分析")
        col1, col2 = st.columns(2)
        with col1:
            metric1 = st.selectbox(
                "选择第一个指标",
                SELECTED_METRICS,
                index=0,
                key='metric1'
            )
        with col2:
            metric2 = st.selectbox(
                "选择第二个指标",
                SELECTED_METRICS,
                index=1,
                key='metric2'
            )
        
        if metric1 != metric2:
            with st.spinner('生成散点图...'):
                scatter_fig = create_scatter_plot(sample_scores, metric1, metric2)
                st.plotly_chart(scatter_fig, use_container_width=True)
                
                # 显示相关性解释
                correlation = corr_matrix.loc[metric1, metric2]
                st.write("### 相关性解释")
                if abs(correlation) > 0.7:
                    strength = "强"
                elif abs(correlation) > 0.3:
                    strength = "中等"
                else:
                    strength = "弱"
                    
                direction = "正" if correlation > 0 else "负"
                st.write(f"这两个指标之间存在{strength}的{direction}相关性 (相关系数: {correlation:.3f})")
        else:
            st.warning("请选择两个不同的指标进行对比")    
    else:  # 样本详情
        st.header("样本详情查看")
        
        sample_index = st.number_input(
            "输入样本序号",
            min_value=1,
            max_value=len(sample_scores),
            value=1
        ) - 1
        
        sample = sample_scores[sample_index]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**源文本:**")
            st.write(sample['source'])
        with col2:
            st.markdown("**参考译文:**")
            st.write(sample['reference'])
        
        st.markdown("**模型译文:**")
        st.write(sample['hypothesis'])
        
        st.markdown("**评分详情:**")
        metrics = {k: v for k, v in sample.items() if k in SELECTED_METRICS}
        metrics_df = pd.DataFrame(metrics.items(), columns=['指标', '得分'])
        st.dataframe(metrics_df.round(4))
    
    # 添加指标说明
    with st.sidebar.expander("指标说明"):
        for metric in SELECTED_METRICS:
            if metric in METRICS_DESCRIPTION:
                st.markdown(f"**{metric.upper()}**: {METRICS_DESCRIPTION[metric]}")

if __name__ == "__main__":
    main() 