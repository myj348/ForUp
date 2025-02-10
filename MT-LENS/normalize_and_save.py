import json
import pandas as pd
import numpy as np
import os

def load_results(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载文件 {file_path} 时出错: {str(e)}")
        return None

def normalize_scores(sample_scores):
    """对指标进行归一化处理"""
    # 需要归一化的指标及其处理方法
    normalize_config = {
        'bleu': {'scale': 100, 'type': 'direct'},  # 除以100
        'chrf': {'scale': 100, 'type': 'direct'},  # 除以100
        'ter': {'scale': 100, 'type': 'inverse'},  # 1 - (值/100)
        'bleurt': {'type': 'minmax'}  # 最小最大值归一化
    }
    
    # 处理BLEURT分数
    bleurt_scores = [score['bleurt'] for score in sample_scores if 'bleurt' in score]
    if bleurt_scores:
        bleurt_min = min(bleurt_scores)
        bleurt_max = max(bleurt_scores)
        bleurt_range = bleurt_max - bleurt_min
        
        # 对每个样本进行归一化
        for score in sample_scores:
            if 'bleurt' in score and bleurt_range > 0:
                score['bleurt'] = (score['bleurt'] - bleurt_min) / bleurt_range
    
    # 处理其他需要归一化的指标
    for score in sample_scores:
        for metric, config in normalize_config.items():
            if metric in score and metric != 'bleurt':  # bleurt已经处理过
                if config['type'] == 'direct':
                    score[metric] = score[metric] / config['scale']
                elif config['type'] == 'inverse':
                    if metric == 'ter':
                        # 确保TER值在0-100范围内
                        ter_value = max(0, min(100, score[metric]))
                        score[metric] = 1 - (ter_value / config['scale'])
                    else:
                        score[metric] = 1 - (score[metric] / config['scale'])
                    
    
    return sample_scores

def save_to_excel(scores, dataset_name, writer):
    """保存归一化后的数据到Excel"""
    # 提取需要的列
    metric_cols = ['bleu', 'chrf', 'ter', 'meteor', 'rouge1', 
                  'rouge2', 'rougeL', 'comet', 'bleurt']
    text_cols = ['index', 'source', 'reference', 'hypothesis']
    
    # 转换为DataFrame
    df = pd.DataFrame(scores)
    all_cols = text_cols + metric_cols
    existing_cols = [col for col in all_cols if col in df.columns]
    
    # 保存到Excel
    sheet_name = dataset_name.replace(" ", "_")[:31]  # Excel限制sheet名最长31字符
    df[existing_cols].to_excel(writer, sheet_name=sheet_name, index=False)
    return df

def main():
    # 定义数据集
    datasets = {
        "中英翻译": "results/comprehensive_evaluation_resultsZE.json",
        "泰英翻译": "results/comprehensive_evaluation_resultsTE.json",
        "英印地翻译": "results/comprehensive_evaluation_resultsEID.json",
        "印地英翻译": "results/comprehensive_evaluation_resultsIDE.json",
        "英印尼翻译": "results/comprehensive_evaluation_resultsEINI.json"
    }
    
    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    
    # 处理所有数据集并保存到Excel
    with pd.ExcelWriter('results/evaluation_samples_normalized.xlsx') as writer:
        for dataset_name, file_path in datasets.items():
            print(f"\n处理数据集: {dataset_name}")
            
            try:
                # 加载数据
                results = load_results(file_path)
                if results is None:
                    print(f"跳过数据集: {dataset_name}")
                    continue
                
                # 获取样本分数并归一化
                sample_scores = results['sample_scores']
                normalized_scores = normalize_scores(sample_scores)
                
                # 保存到Excel
                save_to_excel(normalized_scores, dataset_name, writer)
                print(f"成功处理数据集: {dataset_name}")
                
            except Exception as e:
                print(f"处理数据集 {dataset_name} 时出错: {str(e)}")
                continue
    
    print("\n所有数据已保存到 results/evaluation_samples_normalized.xlsx")

if __name__ == "__main__":
    main() 