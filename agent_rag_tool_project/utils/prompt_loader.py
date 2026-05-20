from agent_rag_tool_project.utils.config_handler import prompts_conf
from agent_rag_tool_project.utils.path_tool import get_abs_path
from agent_rag_tool_project.utils.logger_handler import logger

def load_system_prompts():
    """
    读取系统提示词
    :return:
    """
    try:
        system_prompt_path = get_abs_path(prompts_conf['main_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置中没有main_prompt_path配置项")
        raise e

    try:
        return open(system_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_system_prompts]解析系统提示词错误，{str(e)}")


def load_rag_prompts():
    """
    读取系统提示词
    :return:
    """
    try:
        rag_prompt_path = get_abs_path(prompts_conf['rag_summarize_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置中没有rag_summarize_prompt_path配置项")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]解析RAG提示词错误，{str(e)}")


def load_report_prompts():
    """
    读取系统提示词
    :return:
    """
    try:
        report_prompt_path = get_abs_path(prompts_conf['report_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置中没有report_prompt_path配置项")
        raise e

    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]解析报告提示词错误，{str(e)}")


def load_query_rewrite_prompts():
    """
    读取query rewrite提示词
    :return:
    """
    try:
        query_rewrite_prompt_path = get_abs_path(prompts_conf['query_rewrite_prompt_path'])
    except KeyError as e:
        logger.error(f"[load_query_rewrite_prompts]在yaml配置中没有query_rewrite_prompt_path配置项")
        raise e

    try:
        return open(query_rewrite_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_query_rewrite_prompts]解析query rewrite提示词错误，{str(e)}")

