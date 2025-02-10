from sacrebleu.metrics import BLEU, CHRF, TER
from nltk.translate.meteor_score import meteor_score
import nltk
from typing import List, Dict

class MTEvaluator:
    def __init__(self):
        self.bleu = BLEU()
        self.chrf = CHRF()
        self.ter = TER()
        nltk.download('wordnet')
        
    def evaluate(self, source_texts: List[str], references: List[str], hypotheses: List[str]) -> Dict:
        results = {}
        
        # 确保所有输入都是字符串
        hypotheses = [str(h) if h is not None else "" for h in hypotheses]
        references = [str(r) if r is not None else "" for r in references]
        
        print(f"评估样本数量: {len(hypotheses)}")
        print(f"示例翻译结果: {hypotheses[:2]}")
        
        try:
            bleu_score = self.bleu.corpus_score(hypotheses, [references])
            results['bleu'] = bleu_score.score
            
            chrf_score = self.chrf.corpus_score(hypotheses, [references])
            results['chrf'] = chrf_score.score
            
            ter_score = self.ter.corpus_score(hypotheses, [references])
            results['ter'] = ter_score.score
            
            meteor_scores = []
            for hyp, ref in zip(hypotheses, references):
                score = meteor_score([ref.split()], hyp.split())
                meteor_scores.append(score)
                    
            results['meteor'] = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
            
        except Exception as e:
            print(f"评估过程中出错: {str(e)}")
            raise
            
        return results
