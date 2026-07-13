"""Built-in gateway platform adapters."""

from leapflow.gateway.adapters.api_server import APIServerAdapter
from leapflow.gateway.adapters.dingtalk import DingTalkAdapter
from leapflow.gateway.adapters.feishu import FeishuAdapter
from leapflow.gateway.adapters.telegram import TelegramAdapter
from leapflow.gateway.adapters.webhook import WebhookAdapter

__all__ = [
    "APIServerAdapter",
    "DingTalkAdapter",
    "FeishuAdapter",
    "TelegramAdapter",
    "WebhookAdapter",
]
