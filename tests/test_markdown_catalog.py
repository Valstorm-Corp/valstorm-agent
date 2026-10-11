"""Tests for Markdown Skills and AI Agents Catalog, Visibility Segregation, and Cloud Sync."""

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from cli.main import app
from cli.cmds.sync_cmds import (
    load_local_skills_catalog,
    load_local_agents_catalog,
    validate_catalog_integrity,
    _dump_frontmatter,
    _parse_frontmatter,
    _async_push_catalog,
    _async_pull_catalog,
)

runner = CliRunner()


@pytest.fixture
def temp_catalog(tmp_path):
    """Creates a temporary isolated Markdown catalog with public and internal directories."""
    skills_dir = tmp_path / "skills"
    agents_dir = tmp_path / "agents"

    pub_skills = skills_dir / "public" / "software-development"
    int_skills = skills_dir / "internal" / "agency-operations"
    pub_agents = agents_dir / "public"
    int_agents = agents_dir / "internal"

    pub_skills.mkdir(parents=True)
    int_skills.mkdir(parents=True)
    pub_agents.mkdir(parents=True)
    int_agents.mkdir(parents=True)

    # 1. Public Skill
    (pub_skills / "test-driven-development.md").write_text(
        """---
name: "Test-Driven Development"
slug: "test-driven-development"
api_name: "test_driven_development"
category: "software-development"
description: "Executes unit and integration tests before writing implementation code."
visibility: "public"
is_active: true
version: "1.0.0"
tags:
  - testing
  - tdd
---

# Test-Driven Development Guide
1. Write failing test...
2. Verify test failure...
3. Implement minimal code...
""",
        encoding="utf-8",
    )

    # 2. Internal Skill
    (int_skills / "client-torch-pass-protocol.md").write_text(
        """---
name: "Client Torch Pass Protocol"
slug: "client-torch-pass-protocol"
api_name: "client_torch_pass_protocol"
category: "agency-operations"
description: "Standard operating procedure for client onboarding and torch passing."
visibility: "internal"
is_active: true
version: "1.0.0"
tags:
  - agency-operations
  - torch-passing
---

# Client Torch Pass Protocol
1. Verify build deliverables...
2. Conduct live training...
""",
        encoding="utf-8",
    )

    # 3. Public Agent
    (pub_agents / "developer.md").write_text(
        """---
name: "Monorepo Developer"
slug: "developer"
api_name: "developer"
description: "Specialized in monorepo coding, refactoring, and automated testing."
visibility: "public"
model_tier: "tier_1"
model: "gemini-pro-latest"
provider: "gemini"
is_active: true
allowed_tools:
  - execute_code
  - terminal_exec
  - patch_file
  - write_file
  - read_file
  - search_files
skills:
  - test-driven-development
---

You are a senior monorepo developer specializing in TypeScript and Python FastAPI.
""",
        encoding="utf-8",
    )

    # 4. Internal Agent
    (int_agents / "orchestrator.md").write_text(
        """---
name: "Valstorm Orchestrator"
slug: "orchestrator"
api_name: "orchestrator"
description: "Master orchestrator for task delegation and swarm execution."
visibility: "internal"
model_tier: "tier_2"
model: "gemini-flash-latest"
provider: "gemini"
is_active: true
allowed_tools:
  - delegate_task
  - clarify
  - confirmation_required
skills:
  - test-driven-development
  - client-torch-pass-protocol
---

You are the Valstorm Lead Orchestrator. Coordinate complex workflows and delegate subtasks.
""",
        encoding="utf-8",
    )

    return tmp_path


class TestMarkdownCatalogParsingAndSegregation:
    def test_load_all_skills(self, temp_catalog):
        skills = load_local_skills_catalog(temp_catalog, visibility_filter="all")
        assert len(skills) == 2
        assert "test-driven-development" in skills
        assert "client-torch-pass-protocol" in skills

        tdd = skills["test-driven-development"]
        assert tdd["name"] == "Test-Driven Development"
        assert tdd["category"] == "software-development"
        assert tdd["visibility"] == "public"
        assert "Write failing test" in tdd["body"]

        torch = skills["client-torch-pass-protocol"]
        assert torch["visibility"] == "internal"
        assert torch["category"] == "agency-operations"

    def test_load_skills_visibility_filter(self, temp_catalog):
        pub_skills = load_local_skills_catalog(temp_catalog, visibility_filter="public")
        assert len(pub_skills) == 1
        assert "test-driven-development" in pub_skills
        assert "client-torch-pass-protocol" not in pub_skills

        int_skills = load_local_skills_catalog(temp_catalog, visibility_filter="internal")
        assert len(int_skills) == 1
        assert "client-torch-pass-protocol" in int_skills
        assert "test-driven-development" not in int_skills

    def test_load_all_agents(self, temp_catalog):
        agents = load_local_agents_catalog(temp_catalog, visibility_filter="all")
        assert len(agents) == 2
        assert "developer" in agents
        assert "orchestrator" in agents

        dev = agents["developer"]
        assert dev["name"] == "Monorepo Developer"
        assert dev["visibility"] == "public"
        assert dev["model_tier"] == "tier_1"
        assert dev["skills"] == ["test-driven-development"]
        assert "senior monorepo developer" in dev["system_prompt"]

        orch = agents["orchestrator"]
        assert orch["visibility"] == "internal"
        assert orch["model_tier"] == "tier_2"
        assert orch["skills"] == ["test-driven-development", "client-torch-pass-protocol"]


class TestCatalogValidationAndLinter:
    def test_valid_catalog_passes(self, temp_catalog):
        errors, warnings = validate_catalog_integrity(temp_catalog)
        assert len(errors) == 0, f"Expected 0 errors, got: {errors}"

    def test_missing_required_field_detected(self, temp_catalog):
        # Corrupt an agent file by removing model_tier
        bad_file = temp_catalog / "agents" / "public" / "bad_agent.md"
        bad_file.write_text(
            """---
name: "Bad Agent"
slug: "bad-agent"
visibility: "public"
allowed_tools: ["read_file"]
---
Prompt body
""",
            encoding="utf-8",
        )
        errors, _ = validate_catalog_integrity(temp_catalog)
        assert any("model_tier" in e for e in errors)

    def test_invalid_visibility_detected(self, temp_catalog):
        bad_file = temp_catalog / "skills" / "public" / "test" / "invalid_vis.md"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text(
            """---
name: "Invalid Visibility"
slug: "invalid-vis"
category: "test"
description: "Test desc"
visibility: "super-secret"
---
Body
""",
            encoding="utf-8",
        )
        errors, _ = validate_catalog_integrity(temp_catalog)
        assert any("Invalid visibility" in e for e in errors)

    def test_directory_visibility_mismatch_detected(self, temp_catalog):
        # File placed in public/ but marked internal
        bad_file = temp_catalog / "skills" / "public" / "test" / "mismatch.md"
        bad_file.parent.mkdir(parents=True, exist_ok=True)
        bad_file.write_text(
            """---
name: "Mismatch"
slug: "mismatch"
category: "test"
description: "Test desc"
visibility: "internal"
---
Body
""",
            encoding="utf-8",
        )
        errors, _ = validate_catalog_integrity(temp_catalog)
        assert any("skills/public/ but has visibility='internal'" in e for e in errors)

    def test_nonexistent_skill_reference_detected(self, temp_catalog):
        bad_agent = temp_catalog / "agents" / "public" / "broken_ref.md"
        bad_agent.write_text(
            """---
name: "Broken Ref Agent"
slug: "broken-ref"
visibility: "public"
model_tier: "tier_2"
allowed_tools: ["read_file"]
skills:
  - non-existent-magic-skill
---
Prompt body
""",
            encoding="utf-8",
        )
        errors, _ = validate_catalog_integrity(temp_catalog)
        assert any("References non-existent skill 'non-existent-magic-skill'" in e for e in errors)


class TestCatalogCliCommands:
    def test_catalog_validate_cli(self, temp_catalog, monkeypatch):
        monkeypatch.chdir(temp_catalog)
        result = runner.invoke(app, ["catalog", "validate"])
        assert result.exit_code == 0
        assert "VALID" in result.stdout

    def test_catalog_list_cli(self, temp_catalog, monkeypatch):
        monkeypatch.chdir(temp_catalog)
        result = runner.invoke(app, ["catalog", "list"])
        assert result.exit_code == 0
        assert "developer" in result.stdout
        assert "orchestrator" in result.stdout
        assert "test-driven-development" in result.stdout
        assert "client-torch-pass-protocol" in result.stdout

    def test_catalog_build_profiles_cli(self, temp_catalog, tmp_path, monkeypatch):
        monkeypatch.chdir(temp_catalog)
        out_prof_dir = tmp_path / "out_profiles"
        result = runner.invoke(app, ["catalog", "build-profiles", "--output-dir", str(out_prof_dir)])
        assert result.exit_code == 0
        assert "Profiles Compiled Successfully" in result.stdout

        dev_json = out_prof_dir / "developer.json"
        assert dev_json.is_file()
        data = json.loads(dev_json.read_text(encoding="utf-8"))
        assert data["name"] == "Monorepo Developer"
        assert data["model_tier"] == "tier_1"
        assert data["attached_skill_slugs"] == ["test-driven-development"]

    def test_catalog_push_dry_run_cli(self, temp_catalog, monkeypatch):
        monkeypatch.chdir(temp_catalog)
        result = runner.invoke(app, ["catalog", "push", "--dry-run"])
        assert result.exit_code == 0
        assert "DRY RUN COMPLETE" in result.stdout
        assert "Push Execution Plan" in result.stdout
