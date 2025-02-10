import pandas as pd
from mt_evaluator import MTEvaluator
from ctranslate2 import Translator
import json

class MTEvaluation:
    def __init__(self, model_dir: str):
        self.translator = Translator(
            model_path=model_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type="int8_float16" if torch.cuda.is_available() else "int8",
            inter_threads=2,
            intra_threads=4
        )
        self.evaluator = MTEvaluator()
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        加载数据集，支持CSV、Excel等格式
        """
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(data_path)
        else:
            raise ValueError("Unsupported file format")
        
        # 确保数据集包含必要的列
        required_columns = ['source', 'reference']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Dataset must contain columns: {required_columns}")
            
        return df
    
    def translate_batch(self, texts: List[str], batch_size: int = 32) -> List[str]:
        """
        批量翻译文本
        """
        translations = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results = self.translator.translate_batch(batch)
            translations.extend([r[0]['text'] for r in results])
        return translations
    
    def evaluate_dataset(self, data_path: str, output_path: str = None):
        """
        评估整个数据集
        """
        # 加载数据
        df = self.load_data(data_path)
        
        # 获取模型翻译结果
        hypotheses = self.translate_batch(df['source'].tolist())
        
        # 评估翻译质量
        results = self.evaluator.evaluate(
            source_texts=df['source'].tolist(),
            references=df['reference'].tolist(),
            hypotheses=hypotheses
        )
        
        # 保存详细结果
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