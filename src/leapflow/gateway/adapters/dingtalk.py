"""DingTalk gateway adapter."""
from __future__ import annotations

from typing import Any, Mapping

from leapflow.gateway.adapters.common import (
    AdapterLifecycle,
    HttpRequest,
    HttpResponse,
    JsonHttpClient,
    TinyJsonHttpServer,
    UrlLibJsonHttpClient,
    parse_bind_port,
    parse_json_object,
    stable_message_id,
)
from leapflow.gateway.protocol import InboundMessage, OutboundContent, SendResult, SendTarget


class DingTalkAdapter(AdapterLifecycle):
    """DingTalk adapter with access token refresh and text message sending."""

    platform_id = "dingtalk"
    supports_async_delivery = True
    max_message_length = 5000

    def __init__(
        self,
        app_key: str,
        app_secret: str,
        robot_code: str = "",
        agent_id: str = "",
        connection_mode: str = "webhook",
        profile: str = "default",
        api_base: str = "https://oapi.dingtalk.com",
        auth_base: str = "https://api.dingtalk.com",
        host: str = "127.0.0.1",
        port: str | int = "9092",
        path: str = "/dingtalk/events",
        http_client: JsonHttpClient | None = None,
        **_: Any,
    ) -> None:
        super().__init__(profile=profile)
        self._app_key = app_key
        self._app_secret = app_secret
        self._robot_code = robot_code
        self._agent_id = agent_id
        self._connection_mode = connection_mode or "webhook"
        self._api_base = api_base.rstrip("/")
        self._auth_base = auth_base.rstrip("/")
        self._host = host or "127.0.0.1"
        self._port = parse_bind_port(port, 9092)
        self._path = path if str(path).startswith("/") else f"/{path}"
        self._http = http_client or UrlLibJsonHttpClient()
        self._access_token = ""
        self._server: TinyJsonHttpServer | None = None

    @property
    def local_url(self) -> str:
        if self._server is None:
            return f"http://{self._host}:{self._port}{self._path}"
        return f"{self._server.url_base}{self._path}"

    async def connect(self, *, is_reconnect: bool = False) -> None:
        self._access_token = await self._fetch_access_token()
        if self._connection_mode != "webhook":
            raise NotImplementedError(
                "DingTalk built-in adapter currently supports connection_mode='webhook'. "
                "dingtalk-stream mode is planned as a later adapter enhancement.",
            )
        if self._server is None:
            self._server = TinyJsonHttpServer(self._host, self._port, self._handle_request)
            await self._server.start()
        await super().connect(is_reconnect=is_reconnect)

    async def disconnect(self) -> None:
        if self._server is not None:
            await self._server.stop()
            self._server = None
        await super().disconnect()

    async def send(self, target: SendTarget, content: OutboundContent) -> SendResult:
        token = self._access_token or await self._fetch_access_token()
        if self._robot_code:
            return await self._send_robot_message(token, target, content)
        if self._agent_id:
            return await self._send_corp_message(token, target, content)
        return SendResult(
            ok=False,
            error="DingTalk send requires robot_code or agent_id option",
        )

    async def handle_event(self, payload: Mapping[str, Any]) -> InboundMessage | None:
        """Normalise a DingTalk stream/callback payload and emit it."""
        message = self.message_from_event(payload)
        if message is None:
            return None
        await self._emit(message)
        return message

    async def _handle_request(self, request: HttpRequest) -> HttpResponse:
        if request.path.split("?", 1)[0] != self._path:
            return HttpResponse(404, {"ok": False, "error": "not found"})
        if request.method != "POST":
            return HttpResponse(405, {"ok": False, "error": "method not allowed"})
        payload = parse_json_object(request.body)
        message = await self.handle_event(payload)
        if message is None:
            return HttpResponse(202, {"ok": True, "ignored": True})
        return HttpResponse(202, {"ok": True, "message_id": message.message_id})

    def message_from_event(self, payload: Mapping[str, Any]) -> InboundMessage | None:
        text = self._extract_text(payload)
        if not text:
            return None
        chat_id = str(
            payload.get("conversationId")
            or payload.get("conversation_id")
            or payload.get("chat_id")
            or "dingtalk"
        )
        chat_type = "group" if payload.get("conversationType") == "2" else "dm"
        user_id = str(payload.get("senderStaffId") or payload.get("senderId") or "")
        user_name = str(payload.get("senderNick") or payload.get("senderName") or "")
        message_id = str(payload.get("msgId") or payload.get("message_id") or stable_message_id("dingtalk"))
        return InboundMessage(
            source=self._source(
                chat_id=chat_id,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
            ),
            text=text,
            message_id=message_id,
            metadata={"robot_code": self._robot_code},
        )

    async def _fetch_access_token(self) -> str:
        status, data = await self._http.request_json(
            "POST",
            f"{self._auth_base}/v1.0/oauth2/accessToken",
            json_body={"appKey": self._app_key, "appSecret": self._app_secret},
            timeout_s=10,
        )
        token = data.get("accessToken") or data.get("access_token", "")
        if status >= 400 or data.get("errcode", 0) != 0 or not token:
            raise RuntimeError(str(data.get("errmsg") or "DingTalk token request failed"))
        return str(token)

    async def _send_robot_message(
        self,
        token: str,
        target: SendTarget,
        content: OutboundContent,
    ) -> SendResult:
        url = f"{self._api_base}/topapi/robot/send?access_token={token}"
        payload = {
            "robotCode": self._robot_code,
            "conversationId": target.chat_id,
            "msgKey": "sampleText",
            "msgParam": {"content": content.text[:self.max_message_length]},
        }
        status, data = await self._http.request_json("POST", url, json_body=payload, timeout_s=10)
        if status >= 400 or data.get("errcode", 0) != 0:
            return SendResult(ok=False, error=str(data.get("errmsg") or data))
        return SendResult(ok=True, message_id=str(data.get("task_id") or data.get("request_id") or ""))

    async def _send_corp_message(
        self,
        token: str,
        target: SendTarget,
        content: OutboundContent,
    ) -> SendResult:
        url = f"{self._api_base}/topapi/message/corpconversation/asyncsend_v2?access_token={token}"
        payload = {
            "agent_id": self._agent_id,
            "userid_list": target.chat_id,
            "msg": {"msgtype": "text", "text": {"content": content.text[:self.max_message_length]}},
        }
        status, data = await self._http.request_json("POST", url, json_body=payload, timeout_s=10)
        if status >= 400 or data.get("errcode", 0) != 0:
            return SendResult(ok=False, error=str(data.get("errmsg") or data))
        task_id = data.get("task_id") or data.get("result", {}).get("task_id", "")
        return SendResult(ok=True, message_id=str(task_id))

    @staticmethod
    def _extract_text(payload: Mapping[str, Any]) -> str:
        text = payload.get("text")
        if isinstance(text, dict):
            return str(text.get("content") or "")
        if text:
            return str(text)
        content = payload.get("content")
        if isinstance(content, dict):
            return str(content.get("text") or content.get("content") or "")
        return str(payload.get("msgContent") or "")
