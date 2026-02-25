from utils.logger import get_logger

logger = get_logger("hello_secrets")


def load_secrets():
    """加载密钥配置"""
    logger.info("Loading secrets...")
    return {}


def get_secret(key: str, default=None):
    """获取密钥值"""
    return default
