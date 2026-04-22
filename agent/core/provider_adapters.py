"""Provider adapters for runtime params and model catalog metadata."""

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

_LM_STUDIO_TIMEOUT_SECONDS = 2.0
_LM_STUDIO_CACHE_TTL_SECONDS = 30.0
_LM_STUDIO_DEFAULT_BASE = "http://127.0.0.1:1234/v1"


@dataclass(frozen=True)
class SuggestedModel:
    id: str
    label: str
    description: str
    provider: str
    provider_label: str
    avatar_url: str
    recommended: bool = False
    source: str = "static"


@dataclass(frozen=True)
class ProviderAdapter:
    provider_id: str
    provider_label: str
    prefixes: tuple[str, ...] = ()
    supports_custom_model: bool = False
    custom_model_hint: str | None = None

    def matches(self, model_name: str) -> bool:
        return bool(self.prefixes) and model_name.startswith(self.prefixes)

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return ()

    def available_models(self) -> tuple[SuggestedModel, ...]:
        return self.suggested_models()

    def should_show(self) -> bool:
        return bool(self.available_models())

    def build_params(
        self,
        model_name: str,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        raise NotImplementedError

    def allows_model_name(self, model_name: str) -> bool:
        if any(model.id == model_name for model in self.available_models()):
            return True
        return self.supports_custom_model and self.matches(model_name)

    def to_summary(self) -> dict[str, Any]:
        return {
            "id": self.provider_id,
            "label": self.provider_label,
            "supportsCustomModel": self.supports_custom_model,
            "customModelHint": self.custom_model_hint,
        }


@dataclass(frozen=True)
class NativeAdapter(ProviderAdapter):
    prefixes: tuple[str, ...] = ("anthropic/", "openai/")

    def matches(self, model_name: str) -> bool:
        return model_name.startswith(self.prefixes)

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return (
            SuggestedModel(
                id="anthropic/claude-opus-4-6",
                label="Claude Opus 4.6",
                description="Anthropic",
                provider="anthropic",
                provider_label="Anthropic",
                avatar_url="https://huggingface.co/api/avatars/Anthropic",
                recommended=True,
            ),
        )

    def build_params(
        self,
        model_name: str,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        params: dict[str, Any] = {"model": model_name}
        if reasoning_effort:
            params["reasoning_effort"] = reasoning_effort
        return params


@dataclass(frozen=True)
class HfRouterAdapter(ProviderAdapter):
    allowed_efforts: tuple[str, ...] = ("low", "medium", "high")

    def _is_hf_model_name(self, model_name: str) -> bool:
        if model_name.startswith(_direct_prefixes()):
            return False

        bare = model_name.removeprefix("huggingface/").split(":", 1)[0]
        parts = bare.split("/")
        return len(parts) >= 2 and all(parts)

    def matches(self, model_name: str) -> bool:
        return self._is_hf_model_name(model_name)

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return (
            SuggestedModel(
                id="MiniMaxAI/MiniMax-M2.7",
                label="MiniMax M2.7",
                description="HF Router",
                provider="huggingface",
                provider_label="Hugging Face Router",
                avatar_url="https://huggingface.co/api/avatars/MiniMaxAI",
                recommended=True,
            ),
            SuggestedModel(
                id="moonshotai/Kimi-K2.6",
                label="Kimi K2.6",
                description="HF Router",
                provider="huggingface",
                provider_label="Hugging Face Router",
                avatar_url="https://huggingface.co/api/avatars/moonshotai",
            ),
            SuggestedModel(
                id="zai-org/GLM-5.1",
                label="GLM 5.1",
                description="HF Router",
                provider="huggingface",
                provider_label="Hugging Face Router",
                avatar_url="https://huggingface.co/api/avatars/zai-org",
            ),
        )

    def allows_model_name(self, model_name: str) -> bool:
        return self._is_hf_model_name(model_name)

    def build_params(
        self,
        model_name: str,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        hf_model = model_name.removeprefix("huggingface/")
        inference_token = os.environ.get("INFERENCE_TOKEN")
        api_key = inference_token or session_hf_token or os.environ.get("HF_TOKEN")

        params: dict[str, Any] = {
            "model": f"openai/{hf_model}",
            "api_base": "https://router.huggingface.co/v1",
            "api_key": api_key,
        }

        if inference_token:
            bill_to = os.environ.get("HF_BILL_TO", "smolagents")
            params["extra_headers"] = {"X-HF-Bill-To": bill_to}

        if reasoning_effort:
            hf_level = "low" if reasoning_effort == "minimal" else reasoning_effort
            if hf_level in self.allowed_efforts:
                params["extra_body"] = {"reasoning_effort": hf_level}

        return params


@dataclass(frozen=True)
class OpenAICompatAdapter(ProviderAdapter):
    api_base_url: str = ""
    api_key_env: str = ""
    default_api_key: str = ""
    supports_reasoning_effort: bool = True

    def suggested_model_defs(self) -> tuple[tuple[str, str, bool], ...]:
        return ()

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return tuple(
            SuggestedModel(
                id=f"{self.prefixes[0]}{model_id}",
                label=label,
                description=self.provider_label,
                provider=self.provider_id,
                provider_label=self.provider_label,
                avatar_url=_provider_avatar_url(self.provider_id),
                recommended=recommended,
            )
            for model_id, label, recommended in self.suggested_model_defs()
        )

    def allows_model_name(self, model_name: str) -> bool:
        if not self.matches(model_name):
            return False
        return bool(model_name.removeprefix(self.prefixes[0]))

    def build_params(
        self,
        model_name: str,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        del session_hf_token

        model_id = model_name.removeprefix(self.prefixes[0])
        params: dict[str, Any] = {
            "model": f"openai/{model_id}",
            "api_base": self.api_base_url,
            "api_key": os.environ.get(self.api_key_env, self.default_api_key),
        }
        if reasoning_effort and self.supports_reasoning_effort:
            params["extra_body"] = {"reasoning_effort": reasoning_effort}
        return params


@dataclass(frozen=True)
class OpenCodeGoAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("opencode-go/",)
    api_base_url: str = "https://opencode.ai/zen/go/v1"
    api_key_env: str = "OPENCODE_GO_API_KEY"

    def suggested_model_defs(self) -> tuple[tuple[str, str, bool], ...]:
        return (("kimi-k2.6", "Kimi K2.6", True),)


@dataclass(frozen=True)
class OpenRouterAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("openrouter/",)
    api_base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"

    def suggested_model_defs(self) -> tuple[tuple[str, str, bool], ...]:
        return (("anthropic/claude-sonnet-4.5", "Claude Sonnet 4.5", True),)


@dataclass(frozen=True)
class OpenCodeZenAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("opencode/",)
    api_base_url: str = "https://opencode.ai/zen/v1"
    api_key_env: str = "OPENCODE_ZEN_API_KEY"

    def suggested_model_defs(self) -> tuple[tuple[str, str, bool], ...]:
        return (("kimi-k2.6", "Kimi K2.6", True),)


_lm_studio_cache: dict[str, Any] = {"expires_at": 0.0, "models": ()}


@dataclass(frozen=True)
class LmStudioAdapter(OpenAICompatAdapter):
    prefixes: tuple[str, ...] = ("lm_studio/",)
    api_key_env: str = "LMSTUDIO_API_KEY"
    default_api_key: str = "lm-studio"
    supports_reasoning_effort: bool = False

    def resolved_api_base(self) -> str:
        return os.environ.get("LMSTUDIO_BASE_URL", _LM_STUDIO_DEFAULT_BASE).rstrip("/")

    def should_show(self) -> bool:
        return bool(self.available_models())

    def suggested_models(self) -> tuple[SuggestedModel, ...]:
        return ()

    def available_models(self) -> tuple[SuggestedModel, ...]:
        now = time.monotonic()
        if _lm_studio_cache["expires_at"] > now:
            return _lm_studio_cache["models"]

        models: list[SuggestedModel] = []
        try:
            with urlopen(
                f"{self.resolved_api_base()}/models", timeout=_LM_STUDIO_TIMEOUT_SECONDS
            ) as response:
                payload = json.load(response)
        except (OSError, URLError, TimeoutError, ValueError):
            payload = {"data": []}

        for item in payload.get("data", []):
            model_id = item.get("id")
            if not model_id:
                continue
            models.append(
                SuggestedModel(
                    id=f"lm_studio/{model_id}",
                    label=model_id,
                    description="LM Studio",
                    provider="lm_studio",
                    provider_label="LM Studio",
                    avatar_url=_provider_avatar_url("lm_studio"),
                    source="dynamic",
                )
            )

        cached = tuple(models)
        _lm_studio_cache["expires_at"] = now + _LM_STUDIO_CACHE_TTL_SECONDS
        _lm_studio_cache["models"] = cached
        return cached

    def build_params(
        self,
        model_name: str,
        session_hf_token: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict:
        del session_hf_token
        del reasoning_effort

        return {
            "model": model_name,
            "api_base": self.resolved_api_base(),
            "api_key": os.environ.get(self.api_key_env, self.default_api_key),
        }


ADAPTERS: tuple[ProviderAdapter, ...] = (
    NativeAdapter(provider_id="native", provider_label="Native"),
    HfRouterAdapter(
        provider_id="huggingface",
        provider_label="Hugging Face Router",
        supports_custom_model=True,
        custom_model_hint=(
            "Paste any Hugging Face model id, optionally with "
            ":fastest, :cheapest, :preferred, or :<provider>"
        ),
    ),
    LmStudioAdapter(
        provider_id="lm_studio",
        provider_label="LM Studio",
        supports_custom_model=True,
        custom_model_hint="Use lm_studio/<model-id> from the local LM Studio server",
    ),
    OpenRouterAdapter(
        provider_id="openrouter",
        provider_label="OpenRouter",
        supports_custom_model=True,
        custom_model_hint="Use openrouter/<model-id>, for example openrouter/anthropic/claude-sonnet-4.5",
    ),
    OpenCodeZenAdapter(
        provider_id="opencode_zen",
        provider_label="OpenCode Zen",
        supports_custom_model=True,
        custom_model_hint="Use opencode/<model-id>, for example opencode/kimi-k2.6",
    ),
    OpenCodeGoAdapter(
        provider_id="opencode_go",
        provider_label="OpenCode Go",
        supports_custom_model=True,
        custom_model_hint="Use opencode-go/<model-id>, for example opencode-go/kimi-k2.6",
    ),
)


def _provider_avatar_url(provider_id: str) -> str:
    avatars = {
        "anthropic": "https://huggingface.co/api/avatars/Anthropic",
        "huggingface": "https://huggingface.co/api/avatars/huggingface",
        "lm_studio": "https://avatars.githubusercontent.com/u/16906759?s=200&v=4",
        "openrouter": "https://openrouter.ai/favicon.ico",
        "opencode_zen": "https://huggingface.co/api/avatars/opencode-ai",
        "opencode_go": "https://huggingface.co/api/avatars/opencode-ai",
    }
    return avatars.get(provider_id, "https://huggingface.co/api/avatars/huggingface")


def _direct_prefixes() -> tuple[str, ...]:
    prefixes: list[str] = ["anthropic/", "openai/"]
    for adapter in ADAPTERS:
        prefixes.extend(adapter.prefixes)
    return tuple(dict.fromkeys(prefixes))


def resolve_adapter(model_name: str) -> ProviderAdapter | None:
    for adapter in ADAPTERS:
        if adapter.matches(model_name):
            return adapter
    return None


def is_valid_model_name(model_name: str) -> bool:
    adapter = resolve_adapter(model_name)
    if not adapter:
        return False
    return adapter.allows_model_name(model_name)


def _serialized_model(model: SuggestedModel) -> dict[str, Any]:
    return {
        "id": model.id,
        "label": model.label,
        "description": model.description,
        "provider": model.provider,
        "providerLabel": model.provider_label,
        "avatarUrl": model.avatar_url,
        "recommended": model.recommended,
        "source": model.source,
    }


def get_available_models() -> list[dict[str, Any]]:
    available: list[dict[str, Any]] = []
    for adapter in ADAPTERS:
        if not adapter.should_show():
            continue
        for model in adapter.available_models():
            available.append(_serialized_model(model))
    return available


def get_provider_summaries() -> list[dict[str, Any]]:
    providers: list[dict[str, Any]] = []
    for adapter in ADAPTERS:
        if not adapter.should_show():
            continue
        providers.append(adapter.to_summary())
    return providers


def find_model_option(model_name: str) -> dict[str, Any] | None:
    for model in get_available_models():
        if model["id"] == model_name:
            return model

    adapter = resolve_adapter(model_name)
    if not adapter or not adapter.supports_custom_model:
        return None

    label = model_name
    if adapter.provider_id == "huggingface":
        label = model_name.removeprefix("huggingface/")
    elif adapter.prefixes:
        label = model_name.removeprefix(adapter.prefixes[0])

    return {
        "id": model_name,
        "label": label,
        "description": f"Custom {adapter.provider_label} model",
        "provider": adapter.provider_id,
        "providerLabel": adapter.provider_label,
        "avatarUrl": _provider_avatar_url(adapter.provider_id),
        "recommended": False,
        "source": "custom",
    }


def build_model_catalog(current_model: str) -> dict[str, Any]:
    return {
        "current": current_model,
        "available": get_available_models(),
        "providers": get_provider_summaries(),
        "currentInfo": find_model_option(current_model),
    }
