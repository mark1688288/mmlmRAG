"""Loopback Ollama client, unload, and the vm_stat generator gate.

Documents never leave this Mac. The client refuses non-loopback base URLs,
does not follow redirects, and does not read proxy variables. Pulling uses
``POST /api/pull`` rather than a shell. ``vm_stat`` is the only memory probe.

``think`` is always the top-level JSON false. It is not a setting and it is
never placed under ``options``. Embedding requests always send ``truncate``
false and ``options.num_ctx``; they do not send ``dimensions``.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import httpx

from ragcli.config import Config, normalize_base_url
from ragcli.errors import (
    EXIT_MODEL_MISSING,
    EXIT_OLLAMA_UNREACHABLE,
    EXIT_RESOURCES,
    EXIT_UNCLASSIFIED,
    EXIT_USAGE,
    RagError,
)
from ragcli.i18n import t

logger = logging.getLogger("ragcli.ollama")

NINE_B_MODEL = "qwen3.5:9b"
FOUR_B_MODEL = "qwen3.5:4b"

CONNECT_TIMEOUT_SECONDS = 2.0
META_READ_TIMEOUT_SECONDS = 30.0
EMBED_READ_TIMEOUT_SECONDS = 120.0
VISION_READ_TIMEOUT_SECONDS = 180.0
GENERATE_READ_TIMEOUT_SECONDS = 300.0
# Pull's body is short status lines. The weight download can sit quietly
# between them for longer than a generate call, so the read timeout is open.
# The connect timeout is still 2 seconds.
PULL_READ_TIMEOUT_SECONDS: float | None = None

KEEP_ALIVE_BATCH = "30m"
KEEP_ALIVE_DONE = "0"

_MODEL_NAME = re.compile(r"^[A-Za-z0-9_.:/-]+$")
_KEEP_ALIVE = re.compile(r"^(?:0|[1-9][0-9]*)(?:ms|s|m|h)?$")
_PAGE_SIZE = re.compile(r"page size of (\d+) bytes", re.IGNORECASE)
_PAGE_COUNTS = (
    ("free", re.compile(r"^Pages free:\s+(\d+)\.?[ \t]*$", re.MULTILINE)),
    ("inactive", re.compile(r"^Pages inactive:\s+(\d+)\.?[ \t]*$", re.MULTILINE)),
    ("speculative", re.compile(r"^Pages speculative:\s+(\d+)\.?[ \t]*$", re.MULTILINE)),
)
_ALLOWED_PROBES = (
    ("/usr/bin/vm_stat",),
    ("/usr/sbin/sysctl", "-n", "hw.pagesize"),
)


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """One row of ``GET /api/ps``. ``context_length`` is Ollama's CONTEXT column."""

    name: str
    context_length: int | None


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One chat message. ``images`` are raw base64, without a data-URL prefix."""

    role: str
    content: str
    images: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Assistant text. Thinking is kept apart and is never copied into ``content``."""

    content: str
    thinking: str
    prompt_eval_count: int | None
    eval_count: int | None

    @property
    def thinking_only(self) -> bool:
        """True when the server put the reply in ``thinking`` and left content empty."""
        return self.content.strip() == "" and self.thinking.strip() != ""


def valid_model_name(name: object) -> bool:
    """Ollama model tags this client is willing to send back to Ollama."""
    return isinstance(name, str) and _MODEL_NAME.fullmatch(name) is not None


def request_timeout(read: float | None) -> httpx.Timeout:
    """Connect timeout is always 2 seconds. ``read`` is the operation's budget."""
    write = 30.0 if read is None else read
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT_SECONDS,
        read=read,
        write=write,
        pool=CONNECT_TIMEOUT_SECONDS,
    )


def generation_min_bytes(
    model: str,
    *,
    nine_b_min_bytes: int,
    four_b_min_bytes: int,
) -> int:
    """Bytes of available memory required before loading ``model``.

    The 6 GiB bar belongs only to ``qwen3.5:4b``. Every other generator,
    including ``qwen3.5:9b``, uses the 9 GiB bar so a ``--model`` tag cannot
    step under the gate by using a different name.
    """
    if model == FOUR_B_MODEL:
        return four_b_min_bytes
    return nine_b_min_bytes


def parse_vm_stat(text: str, *, page_size: int | None = None) -> int:
    """Available bytes from ``vm_stat`` text.

    Available pages are free + inactive + speculative. Purgeable pages are
    already inside those counts, so they are not subtracted again. The page
    size is the header's ``page size of N bytes`` when that line is present.
    """
    if not isinstance(text, str):
        raise RagError(EXIT_RESOURCES, t("ollama.vm_stat_unreadable"))
    header = _PAGE_SIZE.search(text)
    if header is not None:
        size = int(header.group(1))
    elif page_size is None:
        raise RagError(EXIT_RESOURCES, t("ollama.vm_stat_no_page_size"))
    else:
        size = page_size
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise RagError(EXIT_RESOURCES, t("ollama.vm_stat_no_page_size"))
    total = 0
    for label, pattern in _PAGE_COUNTS:
        found = pattern.search(text)
        if found is None:
            raise RagError(
                EXIT_RESOURCES,
                t("ollama.vm_stat_missing_pages", label=label),
            )
        total += int(found.group(1))
    return total * size


def available_bytes(run: Callable[[Sequence[str]], str] | None = None) -> int:
    """Read available memory. The default runner only executes vm_stat or sysctl."""
    runner = _run_readonly if run is None else run
    try:
        text = runner(["/usr/bin/vm_stat"])
    except RagError:
        raise
    except OSError as exc:
        raise RagError(EXIT_RESOURCES, t("ollama.vm_stat_unreadable")) from exc
    if not isinstance(text, str) or text.strip() == "":
        raise RagError(EXIT_RESOURCES, t("ollama.vm_stat_unreadable"))
    if _PAGE_SIZE.search(text) is not None:
        return parse_vm_stat(text)
    try:
        raw = runner(["/usr/sbin/sysctl", "-n", "hw.pagesize"])
    except RagError:
        raise
    except OSError as exc:
        raise RagError(EXIT_RESOURCES, t("ollama.vm_stat_unreadable")) from exc
    try:
        page_size = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise RagError(EXIT_RESOURCES, t("ollama.page_size_unreadable")) from exc
    return parse_vm_stat(text, page_size=page_size)


def choose_generator(
    *,
    preferred: str,
    fallback: str,
    tags: Sequence[str],
    available: int,
    nine_b_min_bytes: int,
    four_b_min_bytes: int,
    allow_fallback: bool,
) -> str:
    """Pick the generator that passes the memory gate.

    ``allow_fallback`` is false for ``--no-fallback``. A fallback tag that is
    not in ``tags`` is also not used: the 9B gate then fails with exit 13
    instead of loading 9B. A model that is not installed at all is exit 7.
    The fallback is never a model with a higher memory bar than ``preferred``.
    """
    if not valid_model_name(preferred) or not valid_model_name(fallback):
        raise RagError(EXIT_USAGE, t("model.name_invalid"))
    if isinstance(available, bool) or not isinstance(available, int) or available < 0:
        raise RagError(EXIT_RESOURCES, t("ollama.memory_unavailable"))
    present = {name for name in tags if valid_model_name(name)}
    preferred_need = generation_min_bytes(
        preferred,
        nine_b_min_bytes=nine_b_min_bytes,
        four_b_min_bytes=four_b_min_bytes,
    )
    fallback_need = generation_min_bytes(
        fallback,
        nine_b_min_bytes=nine_b_min_bytes,
        four_b_min_bytes=four_b_min_bytes,
    )
    if preferred in present and available >= preferred_need:
        return preferred

    can_switch = (
        allow_fallback
        and fallback in present
        and fallback != preferred
        and fallback_need <= preferred_need
    )
    if can_switch and available >= fallback_need:
        if preferred in present:
            logger.warning(
                "%s",
                t(
                    "log.memory_switch",
                    available=available,
                    preferred=preferred,
                    need=preferred_need,
                    fallback=fallback,
                ),
            )
        else:
            logger.warning(
                "%s",
                t(
                    "log.model_missing_switch",
                    preferred=preferred,
                    fallback=fallback,
                ),
            )
        return fallback

    if preferred not in present and not can_switch:
        missing = [preferred]
        if allow_fallback and fallback != preferred and fallback not in present:
            missing.append(fallback)
        raise _missing_models(missing)

    checks: list[tuple[str, int]] = []
    if preferred in present:
        checks.append((preferred, preferred_need))
    if (
        allow_fallback
        and fallback in present
        and fallback != preferred
        and fallback_need <= preferred_need
    ):
        checks.append((fallback, fallback_need))
    if not checks:
        checks.append((preferred, preferred_need))
    return _refuse_memory(
        available,
        checks,
        allow_fallback=allow_fallback,
        fallback=fallback,
        present=present,
    )


def model_pull_list(
    config: Config,
    *,
    embed_only: bool = False,
    generator_only: bool = False,
    no_fallback_model: bool = False,
) -> list[str]:
    """Default order is the embed model, then 9B, then 4B."""
    if embed_only and generator_only:
        raise RagError(EXIT_USAGE, t("usage.flags_exclusive"))
    names: list[str] = []
    if not generator_only:
        names.append(config.embed_model)
    if not embed_only:
        names.append(config.generate_model)
        if not no_fallback_model:
            names.append(config.generate_fallback_model)
    ordered: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not valid_model_name(name):
            raise RagError(EXIT_USAGE, t("model.name_invalid"))
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    if not ordered:
        raise RagError(EXIT_USAGE, t("ollama.pull_empty"))
    return ordered


def pull_models(client: OllamaClient, names: Sequence[str]) -> None:
    """Pull one model at a time, then unload so ``ps`` is empty."""
    try:
        for name in names:
            client.pull(name)
    finally:
        client.unload_all()


class OllamaClient:
    """HTTP calls to one already-validated loopback Ollama."""

    def __init__(
        self,
        base_url: str,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = normalize_base_url(base_url)
        self._transport = transport

    def version(self) -> str:
        payload = self._json("GET", "/api/version", read=META_READ_TIMEOUT_SECONDS)
        version = payload.get("version")
        if not isinstance(version, str) or version.strip() == "":
            raise RagError(EXIT_UNCLASSIFIED, t("ollama.version_missing"))
        return version

    def tags(self) -> tuple[str, ...]:
        payload = self._json("GET", "/api/tags", read=META_READ_TIMEOUT_SECONDS)
        models = payload.get("models")
        if not isinstance(models, list):
            raise RagError(EXIT_UNCLASSIFIED, t("ollama.tags_invalid"))
        names: list[str] = []
        seen: set[str] = set()
        for entry in models:
            if not isinstance(entry, dict):
                continue
            name = entry.get("name")
            if valid_model_name(name) and name not in seen:
                seen.add(name)
                names.append(name)
        return tuple(names)

    def ps(self) -> tuple[LoadedModel, ...]:
        payload = self._json("GET", "/api/ps", read=META_READ_TIMEOUT_SECONDS)
        models = payload.get("models")
        if not isinstance(models, list):
            raise RagError(EXIT_UNCLASSIFIED, t("ollama.ps_invalid"))
        loaded: list[LoadedModel] = []
        for entry in models:
            if not isinstance(entry, dict):
                raise RagError(EXIT_UNCLASSIFIED, t("ollama.ps_invalid"))
            name = entry.get("name", entry.get("model"))
            if not valid_model_name(name):
                raise RagError(EXIT_RESOURCES, t("ollama.ps_name_refused"))
            ctx = entry.get("context_length")
            if isinstance(ctx, bool) or not isinstance(ctx, int):
                ctx = None
            loaded.append(LoadedModel(name=name, context_length=ctx))
        return tuple(loaded)

    def embed(
        self,
        model: str,
        inputs: Sequence[str],
        *,
        num_ctx: int,
        keep_alive: str = KEEP_ALIVE_BATCH,
    ) -> list[list[float]]:
        """Embed text. The request carries ``num_ctx`` and never ``dimensions``."""
        _require_model(model)
        _require_num_ctx(num_ctx)
        alive = _require_keep_alive(keep_alive)
        texts = _require_texts(inputs)
        self.prepare_stage(model, num_ctx)
        payload = self._json(
            "POST",
            "/api/embed",
            json_body={
                "model": model,
                "input": texts,
                "truncate": False,
                "keep_alive": alive,
                "options": {"num_ctx": num_ctx},
            },
            read=EMBED_READ_TIMEOUT_SECONDS,
            model=model,
        )
        self.verify_stage(model, num_ctx, keep_alive=alive)
        vectors = _embeddings(payload, expected=len(texts))
        _log_counts("embed", payload)
        return vectors

    def chat(
        self,
        model: str,
        messages: Sequence[ChatMessage],
        *,
        num_ctx: int,
        num_predict: int,
        temperature: float,
        keep_alive: str = KEEP_ALIVE_DONE,
        read_timeout: float = GENERATE_READ_TIMEOUT_SECONDS,
    ) -> ChatResult:
        """Chat with top-level ``think: false``. Vision passes its own timeout."""
        _require_model(model)
        _require_num_ctx(num_ctx)
        alive = _require_keep_alive(keep_alive)
        if isinstance(num_predict, bool) or not isinstance(num_predict, int) or num_predict < 1:
            raise RagError(EXIT_USAGE, t("usage.named_positive_int", name="num_predict"))
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise RagError(EXIT_USAGE, t("config.must_be_number", label="temperature"))
        if temperature < 0:
            raise RagError(EXIT_USAGE, t("config.not_negative", label="temperature"))
        body_messages = _message_payloads(messages)
        self.prepare_stage(model, num_ctx)
        payload = self._json(
            "POST",
            "/api/chat",
            json_body={
                "model": model,
                "stream": False,
                "think": False,
                "keep_alive": alive,
                "messages": body_messages,
                "options": {
                    "num_ctx": num_ctx,
                    "num_predict": num_predict,
                    "temperature": temperature,
                },
            },
            read=read_timeout,
            model=model,
        )
        self.verify_stage(model, num_ctx, keep_alive=alive)
        result = _chat_result(payload)
        _log_counts("chat", payload)
        return result

    def pull(self, model: str) -> None:
        """Download one model. The HTTP body is status JSON, not the weights."""
        _require_model(model)
        logger.info("%s", t("log.pull", model=model))
        _status, raw = self._exchange(
            "POST",
            "/api/pull",
            json_body={"model": model, "stream": True},
            read=PULL_READ_TIMEOUT_SECONDS,
            model=model,
        )
        saw_success = False
        last_status = ""
        text = raw.decode("utf-8", errors="replace")
        for line in text.splitlines():
            if line.strip() == "":
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RagError(
                    EXIT_UNCLASSIFIED,
                    t("ollama.pull_failed", model=model),
                ) from exc
            if not isinstance(event, dict):
                raise RagError(EXIT_UNCLASSIFIED, t("ollama.pull_failed", model=model))
            if event.get("error"):
                raise RagError(EXIT_UNCLASSIFIED, t("ollama.pull_failed", model=model))
            status = event.get("status")
            if status == "success":
                saw_success = True
            elif isinstance(status, str) and status != "downloading" and status != last_status:
                logger.info("%s", t("log.pull_status", model=model, status=status))
                last_status = status
        if not saw_success:
            raise RagError(EXIT_UNCLASSIFIED, t("ollama.pull_failed", model=model))

    def unload_model(self, model: str) -> None:
        """Ask Ollama to drop ``model``. Generate first; embed if that is refused."""
        _require_model(model)
        logger.info("%s", t("log.unload", model=model))
        status, _raw = self._exchange(
            "POST",
            "/api/generate",
            json_body={"model": model, "keep_alive": 0},
            read=GENERATE_READ_TIMEOUT_SECONDS,
            accept_error=True,
        )
        if status < 400:
            return
        status, _raw = self._exchange(
            "POST",
            "/api/embed",
            json_body={"model": model, "input": ".", "keep_alive": 0},
            read=EMBED_READ_TIMEOUT_SECONDS,
            accept_error=True,
        )
        if status >= 400:
            raise RagError(EXIT_UNCLASSIFIED, t("ollama.unload_model_failed", model=model))

    def unload_all(self) -> list[str]:
        """Unload every model in ``ps``. Afterwards ``ps`` must be empty."""
        names = [item.name for item in self.ps()]
        for name in names:
            self.unload_model(name)
        if self.ps():
            raise RagError(EXIT_RESOURCES, t("ollama.ps_not_empty"))
        return names

    def prepare_stage(self, model: str, num_ctx: int) -> None:
        """Unload every other model. Unload the target when its CONTEXT differs.

        A second model that is still loaded is exit 13, and this method does
        not start the stage request.
        """
        _require_model(model)
        _require_num_ctx(num_ctx)
        for item in self.ps():
            if item.name != model:
                self.unload_model(item.name)
        loaded = self.ps()
        others = [item for item in loaded if item.name != model]
        if others:
            raise RagError(EXIT_RESOURCES, t("ask.models_still_loaded"))
        target = next((item for item in loaded if item.name == model), None)
        if target is not None and target.context_length != num_ctx:
            actual = target.context_length
            logger.info(
                "%s",
                t(
                    "log.unload_context",
                    model=model,
                    actual="None" if actual is None else actual,
                    expected=num_ctx,
                ),
            )
            self.unload_model(model)
            if self.ps():
                raise RagError(EXIT_RESOURCES, t("ollama.ps_not_empty"))

    def verify_stage(self, model: str, num_ctx: int, *, keep_alive: str | int) -> None:
        """After the request, CONTEXT must match. A mismatch is unloaded and exit 13.

        ``keep_alive`` 0 may leave ``ps`` empty before this read. An empty list
        is then success. A batch keep-alive must still show the target.
        """
        loaded = self.ps()
        others = [item for item in loaded if item.name != model]
        target = next((item for item in loaded if item.name == model), None)
        if others or (target is not None and target.context_length != num_ctx):
            self._drop_quietly(model, others)
            if target is not None and target.context_length != num_ctx:
                shown = target.context_length
                raise RagError(
                    EXIT_RESOURCES,
                    t(
                        "ollama.context_mismatch",
                        shown="None" if shown is None else shown,
                        num_ctx=num_ctx,
                    ),
                )
            raise RagError(EXIT_RESOURCES, t("ask.models_still_loaded"))
        if target is None:
            if keep_alive in (0, "0"):
                logger.info("%s", t("status.unloaded", name=model))
                return
            raise RagError(EXIT_RESOURCES, t("ollama.ps_missing_model", model=model))
        logger.info("%s", t("log.loaded", model=model, num_ctx=num_ctx))

    def release_stage(self, model: str) -> None:
        """End a stage: ``keep_alive`` 0, then require an empty ``ps``."""
        _require_model(model)
        self.unload_model(model)
        if self.ps():
            raise RagError(EXIT_RESOURCES, t("ollama.ps_not_empty"))

    def _drop_quietly(self, model: str, others: Sequence[LoadedModel]) -> None:
        for item in others:
            try:
                self.unload_model(item.name)
            except RagError:
                logger.error("%s", t("log.unload_failed", model=item.name))
        try:
            self.unload_model(model)
        except RagError:
            logger.error("%s", t("log.unload_failed", model=model))

    def _json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        read: float | None,
        model: str | None = None,
    ) -> dict:
        _status, raw = self._exchange(
            method,
            path,
            json_body=json_body,
            read=read,
            model=model,
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RagError(EXIT_UNCLASSIFIED, t("ollama.not_json")) from exc
        if not isinstance(payload, dict):
            raise RagError(EXIT_UNCLASSIFIED, t("ollama.not_json"))
        return payload

    def _exchange(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None,
        read: float | None,
        model: str | None = None,
        accept_error: bool = False,
    ) -> tuple[int, bytes]:
        timeout = request_timeout(read)
        try:
            with httpx.Client(
                base_url=self.base_url,
                transport=self._transport,
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as http:
                response = http.request(method, path, json=json_body, timeout=timeout)
                status = response.status_code
                if 300 <= status < 400:
                    raise RagError(EXIT_UNCLASSIFIED, t("ollama.redirect"))
                raw = response.content
        except RagError:
            raise
        except httpx.TransportError as exc:
            logger.error("%s", t("ollama.unreachable"))
            raise RagError(EXIT_OLLAMA_UNREACHABLE, t("ollama.unreachable")) from exc
        if status == 404 and not accept_error:
            raise _missing_models([model] if model else [])
        if status >= 400 and not accept_error:
            detail = _safe_error(raw)
            if detail:
                raise RagError(
                    EXIT_UNCLASSIFIED,
                    t("ollama.rejected_detail", detail=detail),
                )
            raise RagError(EXIT_UNCLASSIFIED, t("ollama.rejected"))
        return status, raw


def _run_readonly(argv: Sequence[str]) -> str:
    command = tuple(argv)
    if command not in _ALLOWED_PROBES:
        raise RagError(EXIT_RESOURCES, t("ollama.vm_stat_unreadable"))
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        raise RagError(EXIT_RESOURCES, t("ollama.vm_stat_unreadable")) from exc
    if completed.returncode != 0:
        raise RagError(EXIT_RESOURCES, t("ollama.vm_stat_unreadable"))
    return completed.stdout


def _missing_models(names: Sequence[str]) -> RagError:
    labels = [name for name in names if name]
    if labels:
        shown = t("ollama.list_joiner").join(labels)
        return RagError(EXIT_MODEL_MISSING, t("ollama.model_missing", names=shown))
    return RagError(EXIT_MODEL_MISSING, t("ollama.model_missing_none"))


def _refuse_memory(
    available: int,
    checks: Sequence[tuple[str, int]],
    *,
    allow_fallback: bool,
    fallback: str,
    present: set[str],
) -> str:
    pieces = t("ollama.memory.joiner").join(
        t("ollama.memory.below", model=model, need=need) for model, need in checks
    )
    if not allow_fallback:
        raise RagError(
            EXIT_RESOURCES,
            t("ollama.memory.no_fallback", available=available, pieces=pieces),
        )
    if fallback not in present:
        raise RagError(
            EXIT_RESOURCES,
            t("ollama.memory.fallback_absent", available=available, pieces=pieces),
        )
    raise RagError(EXIT_RESOURCES, t("ollama.memory", available=available, pieces=pieces))


def _require_model(model: str) -> None:
    if not valid_model_name(model):
        raise RagError(EXIT_USAGE, t("model.name_invalid"))


def _require_num_ctx(num_ctx: int) -> None:
    if isinstance(num_ctx, bool) or not isinstance(num_ctx, int) or num_ctx < 1:
        raise RagError(EXIT_USAGE, t("usage.named_positive_int", name="num_ctx"))


def _require_keep_alive(value: str | int) -> str | int:
    if isinstance(value, bool):
        raise RagError(EXIT_USAGE, t("ollama.keep_alive_invalid"))
    if isinstance(value, int):
        if value < 0:
            raise RagError(EXIT_USAGE, t("ollama.keep_alive_invalid"))
        return value
    if isinstance(value, str) and _KEEP_ALIVE.fullmatch(value):
        return value
    raise RagError(EXIT_USAGE, t("ollama.keep_alive_invalid"))


def _require_texts(inputs: Sequence[str]) -> list[str]:
    if isinstance(inputs, str) or not isinstance(inputs, Sequence) or not inputs:
        raise RagError(EXIT_USAGE, t("ollama.embed_input_invalid"))
    texts: list[str] = []
    for item in inputs:
        if not isinstance(item, str) or item == "":
            raise RagError(EXIT_USAGE, t("ollama.embed_input_invalid"))
        texts.append(item)
    return texts


def _message_payloads(messages: Sequence[ChatMessage]) -> list[dict]:
    if not isinstance(messages, Sequence) or isinstance(messages, str) or not messages:
        raise RagError(EXIT_USAGE, t("ollama.chat_messages_required"))
    payloads: list[dict] = []
    for message in messages:
        if not isinstance(message, ChatMessage):
            raise RagError(EXIT_USAGE, t("ollama.chat_message_invalid"))
        if message.role not in {"system", "user", "assistant"}:
            raise RagError(EXIT_USAGE, t("ollama.chat_message_invalid"))
        if not isinstance(message.content, str):
            raise RagError(EXIT_USAGE, t("ollama.chat_message_invalid"))
        payload: dict = {"role": message.role, "content": message.content}
        if message.images:
            images: list[str] = []
            for image in message.images:
                if not isinstance(image, str) or image == "" or image.startswith("data:"):
                    raise RagError(EXIT_USAGE, t("ollama.image_base64"))
                images.append(image)
            payload["images"] = images
        payloads.append(payload)
    return payloads


def _embeddings(payload: Mapping, *, expected: int) -> list[list[float]]:
    raw = payload.get("embeddings")
    if not isinstance(raw, list) or len(raw) != expected:
        raise RagError(EXIT_UNCLASSIFIED, t("ask.embed_count"))
    vectors: list[list[float]] = []
    for item in raw:
        if not isinstance(item, list) or not item:
            raise RagError(EXIT_UNCLASSIFIED, t("ollama.embed_format_invalid"))
        vector: list[float] = []
        for number in item:
            if isinstance(number, bool) or not isinstance(number, (int, float)):
                raise RagError(EXIT_UNCLASSIFIED, t("ollama.embed_format_invalid"))
            vector.append(float(number))
        vectors.append(vector)
    return vectors


def _chat_result(payload: Mapping) -> ChatResult:
    message = payload.get("message")
    if not isinstance(message, dict):
        raise RagError(EXIT_UNCLASSIFIED, t("ollama.chat_format_invalid"))
    content = message.get("content", "")
    thinking = message.get("thinking", payload.get("thinking", ""))
    if not isinstance(content, str):
        content = ""
    if thinking is None:
        thinking = ""
    if not isinstance(thinking, str):
        thinking = ""
    return ChatResult(
        content=content,
        thinking=thinking,
        prompt_eval_count=_optional_count(payload.get("prompt_eval_count")),
        eval_count=_optional_count(payload.get("eval_count")),
    )


def _optional_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _log_counts(kind: str, payload: Mapping) -> None:
    logger.debug(
        "%s prompt_eval_count=%s eval_count=%s",
        kind,
        _optional_count(payload.get("prompt_eval_count")),
        _optional_count(payload.get("eval_count")),
    )


def _safe_error(raw: bytes) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    error = payload.get("error")
    if not isinstance(error, str) or not error.isprintable() or len(error) > 160:
        return ""
    return error
