from sacrebleu.metrics import BLEU, CHRF, TER
from nltk.translate.meteor_score import meteor_score
import nltk
from typing import List, Dict
import pandas as pd

class MTEvaluator:
    def __init__(self):
        # 初始化各种评估指标
        self.bleu = BLEU()
        self.chrf = CHRF()
        self.ter = TER()
        # 确保下载了METEOR所需的数据
        nltk.download('wordnet')
        
    def evaluate(self, source_texts: List[str], references: List[str], hypotheses: List[str]) -> Dict:
        """
        评估翻译质量
        Args:
            source_texts: 源文本列表
            references: 参考译文列表
            hypotheses: 模型译文列表
        """
        results = {}
        
        # BLEU评分
        bleu_score = self.bleu.corpus_score(hypotheses, [references])
        results['bleu'] = bleu_score.score
        
        # CHRF评分
        chrf_score = self.chrf.corpus_score(hypotheses, [references])
        results['chrf'] = chrf_score.score
        
        # TER评分
        ter_score = self.ter.corpus_score(hypotheses, [references])
        results['ter'] = ter_score.score
        
        # METEOR评分 (一句一句计算)
        meteor_scores = []
        for hyp, ref in zip(hypotheses, references):
            score = meteor_score([ref.split()], hyp.split())
            meteor_scores.append(score)
        results['meteor'] = sum(meteor_scores) / len(meteor_scores)
        
        return results 