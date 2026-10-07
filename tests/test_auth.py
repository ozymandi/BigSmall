"""LM Studio API token: config key, env override, Bearer header on the HTTP client."""
from lmagent.client import LMStudioClient
from lmagent.config import load_config


def test_no_key_means_no_header():
    c = LMStudioClient("http://127.0.0.1:9", timeout=1)
    assert "authorization" not in c.http.headers


def test_key_sets_bearer_header():
    c = LMStudioClient("http://127.0.0.1:9", timeout=1, api_key="sk-lm-test")
    assert c.http.headers["authorization"] == "Bearer sk-lm-test"


def test_from_config_reads_server_key(cfg):
    cfg["server"]["api_key"] = "sk-lm-cfg"
    assert LMStudioClient.from_config(cfg).http.headers["authorization"] == "Bearer sk-lm-cfg"
    cfg["server"]["api_key"] = ""
    assert "authorization" not in LMStudioClient.from_config(cfg).http.headers


def test_env_overrides_config(project, monkeypatch):
    from lmagent import config as config_mod
    monkeypatch.setattr(config_mod, "USER_PATH", project / "no-user-config.yaml")
    monkeypatch.setenv("LMSTUDIO_API_KEY", "sk-lm-env")
    assert load_config(cwd=project)["server"]["api_key"] == "sk-lm-env"
