import os
import httpx
from logging import getLogger
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

log = getLogger(__name__)

pb_host = os.getenv("PB_HOST", "http://192.168.31.95:809")

pb_set_key = "baidu_edu_users"


def fetch_key_info(key):
    """
    Fetch information for a given key from PocketBase.
    
    Args:
        key: The key to fetch information for
        
    Returns:
        dict: The JSON response data if successful, None if failed
    """
    r = httpx.get(url=f"{pb_host}/api/collections/{pb_set_key}/records/{key}")
    if r.status_code != 200:
        log.error(f"fetch auth key err: {r.text}")
        return None
    return r.json()


def auth_key_is_ok(key) -> bool:
    data = fetch_key_info(key)
    if not data:
        return False
        
    log.info(data)
    _ = data.get("id")
    exp_time = data.get("exp_time")

    if compare_times(exp_time):
        return True
    else:
        return False


def compare_times(time_str):
    """
    Parses a UTC time string, compares it to the current UTC time, and returns True if the current time is less than the parsed time.

    Args:
        time_str: The UTC time string to parse (e.g., "2025-03-26 12:00:00.000Z").

    Returns:
        True if the current UTC time is less than the parsed UTC time, False otherwise. Returns an error message if the time string is invalid.
    """
    try:
        # 使用`fromisoformat`方法，它能直接解析ISO 8601格式的UTC时间字符串，包括'Z'后缀。
        parsed_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))

        # 获取当前UTC时间。
        now = datetime.now(timezone.utc)

        # 比较时间。
        return now < parsed_time
    except ValueError:
        return "无效的时间字符串格式"
