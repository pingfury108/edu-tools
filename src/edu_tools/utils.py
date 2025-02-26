import base64
import tempfile
import re
import time
import functools


def timed_lru_cache(seconds: int, maxsize: int = 128):
    """
    结合了LRU缓存和时间过期的装饰器。

    Args:
        seconds (int): 缓存的过期时间（秒）
        maxsize (int, optional): 缓存的最大条目数. 默认 128

    Returns:
        function: 装饰器函数
    """
    def wrapper_decorator(func):
        func = functools.lru_cache(maxsize=maxsize)(func)
        func.lifetime = seconds
        func.expiration = time.time() + func.lifetime

        @functools.wraps(func)
        def wrapped_func(*args, **kwargs):
            if time.time() >= func.expiration:
                func.cache_clear()
                func.expiration = time.time() + func.lifetime
            return func(*args, **kwargs)

        return wrapped_func

    return wrapper_decorator


def save_base64_image(base64_data):
    try:
        # 1. 清理数据
        # 移除所有换行符和空格
        base64_data = (
            base64_data.strip().replace("\n", "").replace("\r", "").replace(" ", "")
        )

        # 2. 提取实际的base64内容
        if "base64," in base64_data:
            base64_data = base64_data.split("base64,")[1]

        # 3. 添加错误检查
        if not base64_data:
            return "Error: Empty base64 data"

        # 4. 尝试解码
        try:
            image_data = base64.b64decode(base64_data)
        except Exception as e:
            return f"Base64 decoding error: {str(e)}"

        # 5. 保存文件
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(image_data)
            return temp_file.name

    except Exception as e:
        return f"General error: {str(e)}"
