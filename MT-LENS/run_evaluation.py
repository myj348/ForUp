import os
import torch
from src.evaluate_mt import MTEvaluation  # 确保这行在文件开头

def main():
    try:
        # 设置路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(current_dir, "ctranslate2", "opus-mt-zh-en-ctranslate2")
        data_path = os.path.join(current_dir, "data", "zh_en_test.xlsx")
        output_dir = os.path.join(current_dir, "results")
        
        print(f"初始化评估器...")
        print(f"模型路径: {model_dir}")
        
        # 初始化评估器
        evaluator = MTEvaluation(model_dir=model_dir)
        
        # 运行评估
        print(f"开始评估数据集: {data_path}")
        results = evaluator.evaluate_dataset(
            data_path=data_path,
            output_path=os.path.join(output_dir, "evaluation_results.json")
        )
        
        # 打印评估结果
        print("\nEvaluation Results:")
        print(f"BLEU Score: {results['bleu']:.2f}")
        print(f"CHRF Score: {results['chrf']:.2f}")
        print(f"TER Score: {results['ter']:.2f}")
        print(f"METEOR Score: {results['meteor']:.2f}")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 