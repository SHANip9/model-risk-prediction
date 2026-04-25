from src.common.config import CONFIG_DIR, load_yaml


def test_plans_yaml_loads():
    payload = load_yaml(CONFIG_DIR / "pricing" / "plans.yaml")
    assert "plans" in payload
    assert "basic" in payload["plans"]
