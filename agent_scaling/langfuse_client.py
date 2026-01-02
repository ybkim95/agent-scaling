from langfuse import get_client, Langfuse
from agent_scaling.logger import logger


def get_lf_client() -> Langfuse | None:
    try:
        client = get_client()
        return client
    except Exception as e:
        logger.warning(
            f"Error getting Langfuse client:\n{e}\nProceeding without Langfuse client."
        )
        return None
