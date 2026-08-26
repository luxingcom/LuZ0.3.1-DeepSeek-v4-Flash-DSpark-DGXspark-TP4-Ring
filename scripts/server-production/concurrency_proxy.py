#!/usr/bin/env python3
# =============================================================
# SCRIPT: concurrency_proxy.py
# VERSION: v1.0-r12 (2026-08-26, SRE)
# ROLE: TP4 vLLM 入口并发限制代理 (CONC<=12) — dgxspark01 head 宿主
# 对外入口不变: 0.0.0.0:8001 (aiohttp) → 后端 127.0.0.1:8002 (vLLM)
#   在飞(in-flight) 请求 <= MAX_CONCURRENCY(12), 超过入队等待; 队列满 → 429 + Retry-After:5
#   与 LLM 层 --max-num-seqs 12 联动: 代理保证 vLLM 同时只会收到 <=12 个推理请求,
#   消除 max-num-seqs=12 > shm_broadcast 环容量 6 的结构性不匹配引发的广播块/卡死。
# 依赖: head 宿主 Python 3.12.3 + aiohttp 3.14.3 (已装, 零新依赖)
# 运行: /usr/bin/python3 /opt/aicad-prod/scripts/concurrency_proxy.py  (systemd 托管)
# 配置(环境变量):
#   PROXY_LISTEN           默认 0.0.0.0:8001
#   VLLM_BACKEND           默认 http://127.0.0.1:8002
#   MAX_CONCURRENCY        默认 12   (在飞上限, 与 vLLM --max-num-seqs 对齐)
#   MAX_QUEUE              默认 64   (等待队列上限; 队列容量 = MAX_QUEUE+MAX_CONCURRENCY)
#   STREAM_IDLE_TIMEOUT_S  默认 300  (流空闲超时, 防冻结请求永久占槽; 无总超时)
#   QUEUE_PUT_TIMEOUT_S    默认 0.5  (队列入队等待上限; 超时 → 429)
# 控制面直通(不入队): /health /v1/models /metrics /v1/proxy/health(代理自探活)
# 安全: Authorization 原样透传(不校验/不落日志); 日志脱敏(不记 header/body/query)
# CHANGE: 改脚本须 py_compile + .bak-<tag> 留档 + 更新 REFERENCE.md
# =============================================================
import asyncio
import json
import logging
import os
import signal
import time

from aiohttp import ClientSession, ClientTimeout, web

LOG = logging.getLogger("concurrency-proxy")

# 控制面路径: 不入并发队列, 直通后端 (代理自探活除外)
CONTROL_PLANE = frozenset({"/health", "/v1/models", "/metrics", "/v1/proxy/health"})

# hop-by-hop / 传输层头部: 转发时剔除 (aiohttp 自行管理连接/长度/编码)
HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
})

CONNECT_TIMEOUT_S = 10
CONTROL_READ_TIMEOUT_S = 30


class RequestTask:
    """一个待转发的客户端请求 (由 worker 协程取出转发)."""
    __slots__ = ("request", "response", "done", "exc", "cancelled")

    def __init__(self, request):
        self.request = request
        self.response = None
        self.done = asyncio.Event()
        self.exc = None
        self.cancelled = False


def _parse_listen(spec):
    """'host:port' -> (host, port). 无冒号时按默认 host 0.0.0.0 + 端口."""
    host, _, port = spec.rpartition(":")
    if not host:
        return "0.0.0.0", int(port or 8001)
    return host, int(port)


def _strip_transport_headers(headers):
    """剔除 hop-by-hop/传输层头部, 保留业务头部 (Authorization/Content-Type/...)."""
    out = {}
    for k, v in headers.items():
        kl = k.lower()
        if kl in HOP_BY_HOP or kl.startswith(":"):
            continue
        out[k] = v
    return out


class ConcurrencyProxy:
    def __init__(self):
        self.listen = os.getenv("PROXY_LISTEN", "0.0.0.0:8001")
        self.backend = os.getenv("VLLM_BACKEND", "http://127.0.0.1:8002").rstrip("/")
        self.max_conc = int(os.getenv("MAX_CONCURRENCY", "12"))
        self.max_queue = int(os.getenv("MAX_QUEUE", "64"))
        self.stream_idle = float(os.getenv("STREAM_IDLE_TIMEOUT_S", "300"))
        self.queue_put_timeout = float(os.getenv("QUEUE_PUT_TIMEOUT_S", "0.5"))
        self.queue = asyncio.Queue(maxsize=self.max_queue + self.max_conc)
        self.session = None
        self.workers = []
        self.in_flight = 0
        self._inflight_lock = asyncio.Lock()
        self._last_429_log = 0.0

    # ---------------- setup / shutdown ----------------
    async def _setup(self):
        self.session = ClientSession(auto_decompress=False)
        for i in range(self.max_conc):
            self.workers.append(asyncio.create_task(self._worker(i), name=f"proxy-worker-{i}"))
        app = web.Application(client_max_size=None)
        app.router.add_route("*", "/{tail:.*}", self.handle)
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        host, port = _parse_listen(self.listen)
        site = web.TCPSite(runner, host, port)
        await site.start()
        LOG.info("concurrency-proxy listening on %s:%s backend=%s max_conc=%s max_queue=%s stream_idle=%ss",
                 host, port, self.backend, self.max_conc, self.max_queue, self.stream_idle)

    async def _shutdown(self):
        LOG.info("shutting down concurrency-proxy")
        for w in self.workers:
            w.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        if self.session:
            await self.session.close()

    async def run(self):
        await self._setup()
        stop = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except NotImplementedError:
                pass
        try:
            await stop.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await self._shutdown()

    # ---------------- helpers ----------------
    async def _incr_inflight(self, delta):
        async with self._inflight_lock:
            self.in_flight += delta

    def _log_request(self, request, msg):
        LOG.info("%s %s %s", request.method, request.path, msg)

    def _rate_limit_response(self):
        now = time.monotonic()
        if now - self._last_429_log > 1.0:
            self._last_429_log = now
            LOG.warning("queue full -> 429 (in_flight=%s waiting=%s)", self.in_flight, self.queue.qsize())
        return web.json_response(
            {
                "error": {
                    "message": "TP4 concurrency limit reached (CONC<=12, queue full). Retry after 5s.",
                    "type": "concurrency_limit_reached",
                    "code": 429,
                }
            },
            status=429,
            headers={"Retry-After": "5"},
        )

    # ---------------- 主入口 ----------------
    async def handle(self, request):
        if request.path in CONTROL_PLANE:
            return await self._forward_direct(request)
        task = RequestTask(request)
        try:
            await asyncio.wait_for(self.queue.put(task), timeout=self.queue_put_timeout)
        except (asyncio.QueueFull, asyncio.TimeoutError):
            self._log_request(request, "-> 429 (queue full)")
            return self._rate_limit_response()
        try:
            await task.done.wait()
        except asyncio.CancelledError:
            # 客户端在排队/转发期间断开: 标记取消, worker 弹出后跳过, 释放槽位
            task.cancelled = True
            raise
        if task.exc is not None:
            self._log_request(request, f"-> 502 (backend error: {task.exc})")
            return web.json_response(
                {"error": {"message": f"backend error: {task.exc}", "type": "backend_error"}},
                status=502,
            )
        return task.response

    # ---------------- worker ----------------
    async def _worker(self, idx):
        while True:
            task = await self.queue.get()
            if task.cancelled:
                self.queue.task_done()
                continue
            await self._incr_inflight(1)
            try:
                task.response = await self._forward_request(task.request)
            except asyncio.CancelledError:
                task.exc = "cancelled"
            except Exception as exc:  # noqa: BLE001 - 记录后按 502 返回
                task.exc = exc
                LOG.warning("worker forward error: %s %s -> %r", task.request.method,
                            task.request.path, str(exc)[:300])
            finally:
                await self._incr_inflight(-1)
                task.done.set()
                self.queue.task_done()

    # ---------------- 转发 ----------------
    async def _forward_request(self, request):
        body = await request.read()
        url = self.backend + request.path
        if request.query_string:
            url += "?" + request.query_string
        timeout = ClientTimeout(
            total=None,              # 无总超时: 长生成请求可长时间运行
            connect=CONNECT_TIMEOUT_S,
            sock_connect=CONNECT_TIMEOUT_S,
            sock_read=self.stream_idle,  # 流空闲超时: 防冻结请求永久占槽
        )
        headers = _strip_transport_headers(request.headers)
        async with self.session.request(
            request.method, url, data=body, headers=headers, timeout=timeout
        ) as resp:
            if self._is_streaming(request, body, resp):
                return await self._relay_stream(request, resp)
            buf = await resp.read()
            return web.Response(body=buf, status=resp.status,
                                headers=_strip_transport_headers(resp.headers))

    @staticmethod
    def _is_streaming(request, body, resp):
        ctype = resp.headers.get("Content-Type", "").lower()
        if "text/event-stream" in ctype:
            return True
        if body and body.strip().startswith((b"{", b"[")):
            try:
                obj = json.loads(body)
                if isinstance(obj, dict) and obj.get("stream"):
                    return True
            except Exception:
                pass
        return False

    async def _relay_stream(self, request, resp):
        """SSE / chunked 流式透传 (不缓冲)."""
        client_resp = web.StreamResponse(
            status=resp.status,
            headers=_strip_transport_headers(resp.headers),
        )
        await client_resp.prepare(request)
        try:
            async for chunk in resp.content.iter_any():
                if not chunk:
                    break
                await client_resp.write(chunk)
        except (ConnectionResetError, ConnectionAbortedError):
            pass  # 客户端断开: 停止透传, 关闭后端连接, 释放 worker 槽位
        finally:
            try:
                await client_resp.write_eof()
            except Exception:
                pass
        return client_resp

    # ---------------- 控制面直通 / 自探活 ----------------
    async def _forward_direct(self, request):
        if request.path == "/v1/proxy/health":
            return await self._proxy_health()
        body = await request.read()
        url = self.backend + request.path
        if request.query_string:
            url += "?" + request.query_string
        timeout = ClientTimeout(total=None, connect=CONNECT_TIMEOUT_S,
                                sock_connect=CONNECT_TIMEOUT_S,
                                sock_read=CONTROL_READ_TIMEOUT_S)
        headers = _strip_transport_headers(request.headers)
        async with self.session.request(
            request.method, url, data=body, headers=headers, timeout=timeout
        ) as resp:
            buf = await resp.read()
            return web.Response(body=buf, status=resp.status,
                                headers=_strip_transport_headers(resp.headers))

    async def _proxy_health(self):
        """代理自探活: 不依赖队列, 返回在飞/等待/后端状态."""
        backend = {"url": self.backend}
        try:
            async with self.session.get(
                self.backend + "/health",
                timeout=ClientTimeout(total=3, connect=2, sock_connect=2, sock_read=3),
            ) as r:
                backend["status"] = r.status
                backend["health"] = "ok" if r.status == 200 else f"http_{r.status}"
        except Exception as exc:  # noqa: BLE001
            backend["status"] = 0
            backend["health"] = "down"
            backend["error"] = str(exc)[:200]
        return web.json_response(
            {
                "in_flight": self.in_flight,
                "waiting": self.queue.qsize(),
                "backend": backend,
            }
        )


def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    proxy = ConcurrencyProxy()
    try:
        asyncio.run(proxy.run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
