# Copyright (c) Alibaba, Inc. and its affiliates.
"""Built-in gateway platform adapters."""

from leapflow.gateway.adapters.api_server import APIServerAdapter
from leapflow.gateway.adapters.api_server import plugin as api_server_plugin
from leapflow.gateway.adapters.dingtalk import DingTalkAdapter
from leapflow.gateway.adapters.dingtalk import plugin as dingtalk_plugin
from leapflow.gateway.adapters.feishu import FeishuAdapter
from leapflow.gateway.adapters.feishu import plugin as feishu_plugin
from leapflow.gateway.adapters.telegram import TelegramAdapter
from leapflow.gateway.adapters.telegram import plugin as telegram_plugin
from leapflow.gateway.adapters.webhook import WebhookAdapter
from leapflow.gateway.adapters.webhook import plugin as webhook_plugin

BUILTIN_PLUGINS = [
    feishu_plugin,
    telegram_plugin,
    dingtalk_plugin,
    webhook_plugin,
    api_server_plugin,
]

__all__ = [
    "APIServerAdapter",
    "DingTalkAdapter",
    "FeishuAdapter",
    "TelegramAdapter",
    "WebhookAdapter",
    # Plugin instances
    "api_server_plugin",
    "dingtalk_plugin",
    "feishu_plugin",
    "telegram_plugin",
    "webhook_plugin",
    "BUILTIN_PLUGINS",
]
