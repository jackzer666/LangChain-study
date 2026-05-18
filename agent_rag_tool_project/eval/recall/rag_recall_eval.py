import json

from agent_rag_tool_project.rag.vector_store import VectorStoreService
from agent_rag_tool_project.utils.logger_handler import logger
from agent_rag_tool_project.utils.path_tool import get_abs_path


def evaluate_recall():
    with open(get_abs_path("eval/recall/recall_eval.json"), "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 测试条数
    total = len(dataset)
    # 命中数
    hit = 0

    # 按 question_type 统计
    type_stats = {}

    vs = VectorStoreService()
    retriever = vs.get_retriever()

    for item in dataset:
        question = item["question"]
        gold_chunk_ids = item["gold_chunk_ids"]
        question_type = item["question_type"]

        docs = retriever.invoke(question)
        retrieved_ids = [doc.metadata.get("chunk_id") for doc in docs]
        logger.info(f"[recall_eval]: 查询到的{retrieved_ids}，预设的{gold_chunk_ids}")

        if len(gold_chunk_ids) == 0:
            # 针对negative类型问题，rag没有检索到才是命中
            is_hit = len(docs) == 0
        else:
            # 第一期只要有一个命中即算作命中，先关注有没有找到，而不是有没有找全
            is_hit = any(gold_id in retrieved_ids for gold_id in gold_chunk_ids)

        if is_hit:
            hit += 1
        else:
            logger.info(f"[recall_eval]: 当前回答没有命中{question}")

        # 统计每个 question_type 的命中情况
        if question_type not in type_stats:
            type_stats[question_type] = {"total": 0, "hit": 0}
        type_stats[question_type]["total"] += 1
        if is_hit:
            type_stats[question_type]["hit"] += 1

    # 召回率
    recall = hit / total
    logger.info(f"[recall_eval]: 总召回率 {recall}")

    # 按 question_type 输出命中率
    for q_type, stats in type_stats.items():
        type_recall = stats["hit"] / stats["total"] if stats["total"] > 0 else 0
        logger.info(f"[recall_eval] {q_type}: {type_recall} ({stats['hit']}/{stats['total']})")

if __name__ == '__main__':
    evaluate_recall()