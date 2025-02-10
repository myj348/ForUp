import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from comet import download_model, load_from_checkpoint
import nltk
import json
import os
import numpy as np
import torch
from bleurt import score

class ComprehensiveEvaluator:
    def __init__(self):
        self.bleu = BLEU(effective_order=True)
        self.chrf = CHRF()
        self.ter = TER()
        
        nltk.download('wordnet', quiet=True)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        print("\n正在加载COMET模型...")
        try:
            model_path = download_model("Unbabel/wmt22-comet-da")
            self.comet_model = load_from_checkpoint(model_path)
            self.comet_available = True
            print("COMET模型加载成功!")
        except Exception as e:
            print(f"COMET模型加载失败: {str(e)}")
            self.comet_available = False

        print("\n正在加载BLEURT模型...")
        try:
            self.bleurt_scorer = score.BleurtScorer()
            self.bleurt_available = True
            print("BLEURT模型加载成功!")
        except Exception as e:
            print(f"BLEURT模型加载失败: {str(e)}")
            self.bleurt_available = False

    def evaluate_comet(self, sources, hypotheses, references):
        if not self.comet_available:
            return None, None
            
        print("开始COMET评估...")
        data = [{
            "src": src,
            "mt": hyp,
            "ref": ref
        } for src, hyp, ref in zip(sources, hypotheses, references)]
        
        try:
            with torch.no_grad():
                model_output = self.comet_model.predict(data, batch_size=8, gpus=0)
            print("COMET评估完成!")
            return model_output.scores, model_output.system_score
        except Exception as e:
            print(f"COMET评估过程中出错: {str(e)}")
            return None, None

    def evaluate(self, data_path: str, output_path: str = None):
        print(f"正在读取数据: {data_path}")
        df = pd.read_excel(data_path)
        
        required_columns = ['source', 'reference', 'hypothesis']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"数据集必须包含以下列: {required_columns}")
        
        print(f"数据集大小: {len(df)} 条")
        
        sample_scores = []
        all_scores = {metric: [] for metric in [
            'bleu', 'chrf', 'ter',
            'meteor', 'rouge1', 'rouge2', 'rougeL','comet','bleurt'
        ]}
        
        print("\n计算COMET分数...")
        comet_scores, comet_system_score = self.evaluate_comet(
            df['source'].tolist(),
            df['hypothesis'].tolist(),
            df['reference'].tolist()
        )
        
        print("\n计算其他评估指标...")
        for idx, row in df.iterrows():
            scores = {
                'bleu': self.bleu.sentence_score(row['hypothesis'], [row['reference']]).score,
                'chrf': self.chrf.sentence_score(row['hypothesis'], [row['reference']]).score,
                'ter': self.ter.sentence_score(row['hypothesis'], [row['reference']]).score,
                'meteor': meteor_score([row['reference'].split()], row['hypothesis'].split()),
            }

            if self.bleurt_available:
                try:
                    bleurt_score = float(self.bleurt_scorer.score(
                        references=[row['reference']], 
                        candidates=[row['hypothesis']]
                    )[0])
                    scores['bleurt'] = bleurt_score
                except Exception as e:
                    print(f"BLEURT评分出错 (样本 {idx}): {str(e)}")
                    scores['bleurt'] = None


            rouge_scores = self.rouge_scorer.score(row['reference'], row['hypothesis'])
            scores.update({
                'rouge1': rouge_scores['rouge1'].fmeasure,
                'rouge2': rouge_scores['rouge2'].fmeasure,
                'rougeL': rouge_scores['rougeL'].fmeasure
            })
            
            if comet_scores is not None:
                scores['comet'] = comet_scores[idx]
            
            scores.update({
                'index': idx + 1,
                'source': row['source'],
                'reference': row['reference'],
                'hypothesis': row['hypothesis']
            })
            
            sample_scores.append(scores)
            
            for metric in all_scores.keys():
                if metric in scores:
                    all_scores[metric].append(scores[metric])
        
        corpus_results = {
            metric: np.mean(scores) for metric, scores in all_scores.items()
            if scores
        }
        
        
        print("\n整体评估结果:")
        for metric, score in corpus_results.items():
            print(f"{metric.upper():15} Score: {score:.4f}")
        
        print("\n各样本评分示例 (前3个):")
        metrics = list(all_scores.keys()) + (['comet'] if comet_scores is not None else [])
        print("序号\t" + "\t".join(metrics))
        print("-" * 120)
        for score in sample_scores[:3]:
            print(f"{score['index']}\t" + 
                  "\t".join([f"{score.get(m, 'N/A'):.4f}" for m in metrics]))
        print("... ...")
        
        if output_path:
            detailed_results = {
                'corpus_metrics': corpus_results,
                'sample_scores': sample_scores,
                'metrics_description': {
                    'bleu': 'BLEU score (0-100)',
                    'chrf': 'chrF score (0-100)',
                    'ter': 'Translation Edit Rate (lower is better)',
                    'meteor': 'METEOR score (0-1)',
                    'rouge1': 'ROUGE-1 F1 score (0-1)',
                    'rouge2': 'ROUGE-2 F1 score (0-1)',
                    'rougeL': 'ROUGE-L F1 score (0-1)',
                    'comet': 'COMET segment-level score',
                    'bleurt': 'BLEURT score (基于BERT的评估指标, 分数范围不固定, 越高越好)'
                }
            }
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, ensure_ascii=False, indent=2)
            print(f"\n详细结果已保存至: {output_path}")
        
        return corpus_results, sample_scores

def main():
    evaluator = ComprehensiveEvaluator()

    datasets = [
        {
            'name': '中英',
            'input': 'data/zh_en_test.xlsx',
            'output': 'results/comprehensive_evaluation_resultsZE.json'
        },
        {
            'name': '泰英',
            'input': 'data/tai_en_test.xlsx',
            'output': 'results/comprehensive_evaluation_resultsTE.json'
        },
        {
            'name': '英印地',
            'input': 'data/en_yindi_test.xlsx',
            'output': 'results/comprehensive_evaluation_resultsEID.json'
        },
        {
            'name': '印地英',
            'input': 'data/yindi_en_test.xlsx',
            'output': 'results/comprehensive_evaluation_resultsIDE.json'
        },
        {
            'name': '英印尼',
            'input': 'data/en_ini_test.xlsx',
            'output': 'results/comprehensive_evaluation_resultsEINI.json'
        }

    ]

    # 依次评估每个数据集
    for dataset in datasets:
        print(f"\n开始评估 {dataset['name']}...")
        try:
            corpus_results, sample_scores = evaluator.evaluate(
                data_path=dataset['input'],
                output_path=dataset['output']
            )
            print(f"{dataset['name']} 评估完成!")
            
        except Exception as e:
            print(f"{dataset['name']} 评估失败: {str(e)}")
            continue


if __name__ == "__main__":
    main() 