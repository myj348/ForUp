from ctranslate2 import Translator
import torch
from typing import List, Dict
import pandas as pd
import json
from .mt_evaluator import MTEvaluator

class MTEvaluation:  # 确保类名完全一致
    def __init__(self, model_dir: str):
        print(f"正在加载模型，路径: {model_dir}")
        try:
            self.translator = Translator(
                model_path=model_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
                compute_type="int8_float16" if torch.cuda.is_available() else "int8",
                inter_threads=2,
                intra_threads=4
            )
            # 确保初始化 evaluator
            from .mt_evaluator import MTEvaluator
            self.evaluator = MTEvaluator()
            print("模型和评估器加载成功")
        except Exception as e:
            print(f"初始化失败: {str(e)}")
            raise
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        print(f"正在加载数据，路径: {data_path}")
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(data_path)
            else:
                raise ValueError("Unsupported file format")
            
            required_columns = ['source', 'reference']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"Dataset must contain columns: {required_columns}")
            
            print(f"数据加载成功，共 {len(df)} 条记录")
            print("数据样例:")
            print(df.head(2))
            
            return df
        except Exception as e:
            print(f"数据加载失败: {str(e)}")
            raise
    
    def translate_batch(self, texts: List[str], batch_size: int = 32) -> List[str]:
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"\n批次 {i//batch_size + 1}:")
            print(f"输入样本: {batch[:2]}...")
            
            try:
                # 将每个句子转换为列表格式 - 修改这里
                batch = [[text] for text in batch]  # 每个文本都包装在一个列表中
                results = self.translator.translate_batch(batch)
                
                # 获取翻译结果
                batch_translations = [r.hypotheses[0] for r in results]
                print(f"翻译结果: {batch_translations[:2]}...")
                
                translations.extend(batch_translations)
                
            except Exception as e:
                print(f"翻译批次 {i} 时出错: {str(e)}")
                print(f"当前批次内容: {batch}")
                raise
                
        return translations
                
        # 验证整体翻译结果的多样性
        unique_translations = len(set(translations))
        total_translations = len(translations)
        print(f"\n翻译统计:")
        print(f"总翻译数: {total_translations}")
        print(f"不同翻译数: {unique_translations}")
        
        if unique_translations == 1:
            print("错误：所有输入得到了相同的翻译结果！")
            print(f"统一的翻译结果是: {translations[0]}")
            
        return translations
    
    def evaluate_dataset(self, data_path: str, output_path: str = None):
        df = self.load_data(data_path)
        hypotheses = self.translate_batch(df['source'].tolist())
        
        results = self.evaluator.evaluate(
            source_texts=df['source'].tolist(),
            references=df['reference'].tolist(),
            hypotheses=hypotheses
        )
        
        if output_path:
            detailed_results = {
                'metrics': results,
                'samples': []
            }
            
            for src, ref, hyp in zip(df['source'], df['reference'], hypotheses):
                detailed_results['samples'].append({
                    'source': src,
                    'reference': ref,
                    'hypothesis': hyp
                })
                
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        return results
