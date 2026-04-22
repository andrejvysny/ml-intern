from agent.core.llm_params import _resolve_llm_params
import agent.core.provider_adapters as providers


def test_native_adapter_keeps_model_name():
    params = _resolve_llm_params("anthropic/claude-opus-4-6", reasoning_effort="high")

    assert params == {
        "model": "anthropic/claude-opus-4-6",
        "reasoning_effort": "high",
    }


def test_hf_adapter_builds_router_params(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf-test")

    params = _resolve_llm_params(
        "moonshotai/Kimi-K2.6:novita", reasoning_effort="minimal"
    )

    assert params == {
        "model": "openai/moonshotai/Kimi-K2.6:novita",
        "api_base": "https://router.huggingface.co/v1",
        "api_key": "hf-test",
        "extra_body": {"reasoning_effort": "low"},
    }


def test_hf_adapter_adds_bill_to_header(monkeypatch):
    monkeypatch.setenv("INFERENCE_TOKEN", "hf-space-token")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    params = _resolve_llm_params("MiniMaxAI/MiniMax-M2.7")

    assert params["extra_headers"] == {"X-HF-Bill-To": "smolagents"}
    assert params["api_key"] == "hf-space-token"


def test_lm_studio_uses_dynamic_models(monkeypatch):
    monkeypatch.setattr(
        providers.LmStudioAdapter,
        "available_models",
        lambda self: (
            providers.SuggestedModel(
                id="lm_studio/google/gemma-3-12b",
                label="google/gemma-3-12b",
                description="LM Studio",
                provider="lm_studio",
                provider_label="LM Studio",
                avatar_url="avatar",
                source="dynamic",
            ),
        ),
    )

    catalog = providers.build_model_catalog("anthropic/claude-opus-4-6")

    assert any(
        model["id"] == "lm_studio/google/gemma-3-12b" for model in catalog["available"]
    )
    assert any(provider["id"] == "lm_studio" for provider in catalog["providers"])


def test_openrouter_adapter_uses_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-key")

    params = _resolve_llm_params("openrouter/anthropic/claude-sonnet-4.5")

    assert params == {
        "model": "openai/anthropic/claude-sonnet-4.5",
        "api_base": "https://openrouter.ai/api/v1",
        "api_key": "router-key",
    }


def test_opencode_zen_adapter_uses_api_key(monkeypatch):
    monkeypatch.setenv("OPENCODE_ZEN_API_KEY", "zen-key")

    params = _resolve_llm_params("opencode/kimi-k2.6")

    assert params == {
        "model": "openai/kimi-k2.6",
        "api_base": "https://opencode.ai/zen/v1",
        "api_key": "zen-key",
    }


def test_opencode_go_adapter_uses_api_key(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "go-test-key")

    params = _resolve_llm_params("opencode-go/kimi-k2.6")

    assert params == {
        "model": "openai/kimi-k2.6",
        "api_base": "https://opencode.ai/zen/go/v1",
        "api_key": "go-test-key",
    }


def test_model_catalog_comes_from_adapters(monkeypatch):
    monkeypatch.setattr(providers.LmStudioAdapter, "available_models", lambda self: ())

    catalog = providers.build_model_catalog("anthropic/claude-opus-4-6")

    assert catalog["current"] == "anthropic/claude-opus-4-6"
    assert any(model["provider"] == "anthropic" for model in catalog["available"])
    assert any(model["provider"] == "huggingface" for model in catalog["available"])
    assert any(model["provider"] == "openrouter" for model in catalog["available"])
    assert any(model["provider"] == "opencode_zen" for model in catalog["available"])
    assert any(model["provider"] == "opencode_go" for model in catalog["available"])
    assert not any(provider["id"] == "lm_studio" for provider in catalog["providers"])


def test_model_validation_accepts_free_form_provider_ids():
    assert providers.is_valid_model_name("moonshotai/Kimi-K2.6:fastest") is True
    assert (
        providers.is_valid_model_name("huggingface/moonshotai/Kimi-K2.6:novita") is True
    )
    assert providers.is_valid_model_name("openrouter/google/gemini-2.5-pro") is True
    assert providers.is_valid_model_name("opencode/glm-5.1") is True
    assert providers.is_valid_model_name("opencode-go/glm-5.1") is True
    assert providers.is_valid_model_name("lm_studio/google/gemma-3-12b") is True
