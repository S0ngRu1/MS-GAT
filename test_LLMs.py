import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from loguru import logger
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

load_dotenv()
# --- 初始化 API 客户端 ---
client = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE"),
)
llm_model = "doubao-seed-1.6-flash"
datasets_name = "Weibo17"
datasets_path = os.path.join("datasets", datasets_name, "test.csv")


# --- 加载数据 ---
def load_datasets(datasets_path: str):
    """加载数据集，返回文本列表和对应的真实标签列表"""
    try:
        datasets = pd.read_csv(datasets_path)
        texts = datasets.iloc[:, 2].astype(str).tolist()  # 确保文本为字符串类型
        labels = datasets.iloc[:, 3].astype(int).tolist()  # 确保标签为整数类型
        logger.info(f"成功加载数据集，共{len(texts)}个样本")
        return texts, labels
    except Exception as e:
        logger.error(f"加载数据集失败：{str(e)}")
        raise


# --- 提示词模板 ---
PROMPT_TEMPLATE = """请严格判断以下文本是否为虚假信息，遵循以下规则：
1. 虚假信息请返回数字1
2. 真实信息请返回数字0
3. 仅返回0或1，不添加任何额外文字、标点或解释

文本内容：
{text}
"""


# --- 预测函数 ---
def predict_with_llm(texts):
    """使用大模型对文本列表进行预测，返回预测标签列表"""
    predictions = []
    for i, text in enumerate(texts, 1):
        logger.info(f"正在处理第{i}/{len(texts)}个样本")
        try:
            # 构建提示词
            prompt = PROMPT_TEMPLATE.format(text=text)

            # 调用大模型
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0  # 降低随机性，确保输出稳定
            )

            # 解析结果
            if not response.choices:
                raise ValueError("模型返回结果为空（无choices）")

            message = response.choices[0].message
            if message.content is None:  # 关键修复：检查content是否为None
                raise ValueError("模型返回内容为空（content为None）")
            pred_str = message.content.strip()
            if pred_str not in ["0", "1"]:
                raise ValueError(f"模型返回格式错误，预期0或1，实际返回：{pred_str}")

            predictions.append(int(pred_str))

        except Exception as e:
            logger.warning(f"第{i}个样本处理失败：{str(e)}，将跳过该样本")
            predictions.append(None)  # 标记失败样本

    return predictions


# --- 计算评估指标 ---
def calculate_metrics(true_labels, pred_labels):
    """计算评估指标（处理NaN和None）"""
    # 过滤无效样本（排除None和NaN）
    valid_pairs = [
        (t, p) for t, p in zip(true_labels, pred_labels)
        if p is not None and not np.isnan(p)
    ]
    if not valid_pairs:
        raise ValueError("没有有效的预测结果用于计算指标")

    true_valid, pred_valid = zip(*valid_pairs)
    # 确保标签为整数类型
    true_valid = [int(t) for t in true_valid]
    pred_valid = [int(p) for p in pred_valid]

    accuracy = accuracy_score(true_valid, pred_valid)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_valid, pred_valid, average=None, labels=[0, 1]
    )

    return {
        "accuracy": float(accuracy),
        "class_0": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0])
        },
        "class_1": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1])
        }
    }


# --- 主函数 ---
def main():
    # 加载数据
    texts, true_labels = load_datasets(datasets_path)

    # 模型预测
    pred_labels = predict_with_llm(texts)

    # 保存预测结果
    result_df = pd.DataFrame({
        "text": texts,
        "true_label": true_labels,
        "pred_label": pred_labels
    })
    result_file_path = f"results/{datasets_name}_{llm_model}_prediction_results.csv"
    result_df.to_csv(result_file_path, index=False)
    logger.info(f"预测结果已保存至 {result_file_path}")
    # result_df = pd.read_csv("results/CFND_Doubao_prediction_results.csv")
    # true_labels = result_df["true_label"].tolist()
    # pred_labels = result_df["pred_label"].tolist()
    # 计算并打印指标
    metrics = calculate_metrics(true_labels, pred_labels)

    print("\n===== 评估指标 =====")
    print(f"总体准确率: {metrics['accuracy']:.4f}")
    print("\n类别0（真实信息）:")
    print(f"  精确率: {metrics['class_0']['precision']:.4f}")
    print(f"  召回率: {metrics['class_0']['recall']:.4f}")
    print(f"  F1分数: {metrics['class_0']['f1']:.4f}")
    print(f"  支持样本数: {metrics['class_0']['support']}")
    print("\n类别1（虚假信息）:")
    print(f"  精确率: {metrics['class_1']['precision']:.4f}")
    print(f"  召回率: {metrics['class_1']['recall']:.4f}")
    print(f"  F1分数: {metrics['class_1']['f1']:.4f}")
    print(f"  支持样本数: {metrics['class_1']['support']}")


if __name__ == "__main__":
    main()
