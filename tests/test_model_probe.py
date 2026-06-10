"""Tests for provider model discovery and cheap compressor probing."""

from unittest.mock import patch

from token_optimizer.core.model_probe import (
    ProviderModelProbe,
    ProbeModelInfo,
    discover_cheap_options,
    infer_provider,
    normalize_models_payload,
    route_from_probe,
)


class TestProviderInference:
    def test_infer_mimo_from_url(self):
        assert infer_provider("https://platform.xiaomimimo.com/v1", "mimo-v2.5-pro") == "mimo"

    def test_infer_openai_from_model(self):
        assert infer_provider("", "gpt-4o") == "openai"


class TestNormalizeModelsPayload:
    def test_openai_like_payload(self):
        models = normalize_models_payload({"data": [{"id": "mimo-v2.5-pro"}, {"id": "mimo-v2-flash"}]})
        assert [model.id for model in models] == ["mimo-v2.5-pro", "mimo-v2-flash"]

    def test_plain_list_payload(self):
        models = normalize_models_payload(["qwen-max", "qwen-turbo"])
        assert [model.id for model in models] == ["qwen-max", "qwen-turbo"]


class TestCheapOptionDiscovery:
    def test_discovers_mimo_flash_from_inventory(self):
        options = discover_cheap_options(
            "mimo-v2.5-pro",
            [ProbeModelInfo("mimo-v2.5-pro"), ProbeModelInfo("mimo-v2-flash")],
            provider="mimo",
        )
        assert options[0].model == "mimo-v2-flash"
        assert options[0].max_context == 256_000
        assert options[0].cross_generation is True

    def test_route_from_probe_builds_model_route(self):
        result = route_from_probe(
            "mimo-v2.5-pro",
            [ProbeModelInfo("mimo-v2.5-pro"), ProbeModelInfo("mimo-v2-flash")],
            base_url="https://platform.xiaomimimo.com/v1",
        )
        assert result.available is True
        assert result.provider == "mimo"
        assert result.route is not None
        assert result.route.cheap_options[0].model == "mimo-v2-flash"
        assert result.route.main_input_price == 1.00

    def test_static_fallback_when_inventory_empty_but_known_route_exists(self):
        result = route_from_probe("mimo-v2.5-pro", (), base_url="https://platform.xiaomimimo.com/v1")
        assert result.available is True
        assert result.source == "static_fallback"
        assert result.cheap_options[0].model == "mimo-v2-flash"

    def test_no_candidate_for_unknown_model(self):
        result = route_from_probe("custom-large", [ProbeModelInfo("custom-large")], provider="unknown")
        assert result.available is False
        assert result.route is None
        assert "no cheap" in result.failure_reason


class TestProviderModelProbe:
    def test_probe_success_uses_models_endpoint(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"data": [{"id": "mimo-v2.5-pro"}, {"id": "mimo-v2-flash"}]}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, headers=None):
                assert url.endswith("/models")
                assert headers["Authorization"] == "Bearer sk-test"
                return FakeResponse()

        probe = ProviderModelProbe("https://platform.xiaomimimo.com/v1", api_key="sk-test")
        with patch("token_optimizer.core.model_probe.httpx.Client", FakeClient):
            result = probe.probe("mimo-v2.5-pro")
        assert result.available is True
        assert result.source == "models_inventory"
        assert result.route.cheap_options[0].model == "mimo-v2-flash"

    def test_probe_failure_falls_back_to_static_route(self):
        class BrokenClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def get(self, url, headers=None):
                raise RuntimeError("network down")

        probe = ProviderModelProbe("https://platform.xiaomimimo.com/v1", api_key="sk-test")
        with patch("token_optimizer.core.model_probe.httpx.Client", BrokenClient):
            result = probe.probe("mimo-v2.5-pro")
        assert result.available is True
        assert result.source == "static_fallback_after_probe_failure"
        assert "network down" in result.failure_reason
        assert result.route.cheap_options[0].model == "mimo-v2-flash"
