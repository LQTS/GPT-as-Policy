"""Tests for the DexHand Codex policy launcher."""

from hybrid_rollout.dexhand.policy import app_server_argv, build_policy_prompt


def test_app_server_argv_omits_version_specific_view_image_feature():
    config = {
        "model": "gpt-test",
        "features.shell_tool": True,
        "features.view_image": True,
    }

    argv = app_server_argv("codex", config)

    assert argv[:4] == ["codex", "app-server", "--stdio", "--strict-config"]
    assert "features.shell_tool=true" in argv
    assert not any("features.view_image" in item for item in argv)


def test_policy_prompt_template_differs_only_by_workspace():
    first = build_policy_prompt("skill", "contract", "/tmp/first")
    second = build_policy_prompt("skill", "contract", "/tmp/second")
    template = build_policy_prompt("skill", "contract", "<WORKSPACE>")

    assert first.replace("/tmp/first", "<WORKSPACE>") == template
    assert second.replace("/tmp/second", "<WORKSPACE>") == template
