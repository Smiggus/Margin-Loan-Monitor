"""Small, polite HTTP client for SEC EDGAR (identifies itself and stays under the rate limit)."""
import threading
import time

import requests

from mlm import config


class SecClient:
    def __init__(self, user_agent: str = config.SEC_USER_AGENT, max_rps: float = config.SEC_MAX_RPS):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self._interval = 1.0 / max_rps
        self._lock = threading.Lock()
        self._last = 0.0

    def _throttle(self) -> None:
        with self._lock:
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get(self, url: str, params: dict | None = None, retries: int = 3, timeout: int = 90):
        for attempt in range(retries):
            self._throttle()
            try:
                r = self.session.get(url, params=params, timeout=timeout)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(5 * (attempt + 1))
                continue
            return r
        return None
