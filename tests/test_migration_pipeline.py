"""Unit tests for the Skills & Agent Migration and Preparation Scripts."""

import json
import pytest
from pathlib import Path

pytest.importorskip("scripts.migrate_hermes_profiles_and_skills")
from scripts.migrate_hermes_profiles_and_skills import parse_skill_md, run_migration
from scripts.upload_skills_and_agents import prepare_and_validate_bundles


def test_parse_skill_md():
    raw = """---
name: test-skill
title: Test Skill Display
description: This is a test skill description.
author: TestAuthor
is_active: true
---

# Test Skill

## Steps
1. Do this.
2. Do that.
"""
    fm, body = parse_skill_md(raw)
    assert fm["name"] == "test-skill"
    assert fm["title"] == "Test Skill Display"
    assert fm["description"] == "This is a test skill description."
    assert fm["author"] == "TestAuthor"
    assert fm["is_active"] is True
    assert "## Steps" in body
    assert "1. Do this." in body


def test_run_migration_local_tree(tmp_path):
    """Tests the Hermes extractor creating clean profiles and skills without IDs."""
    skills, profiles = run_migration(output_dir=tmp_path)
    assert isinstance(skills, list)
    assert len(profiles) >= 5

    # Verify every skill has required schema fields and NO id
    for s in skills:
        assert "api_name" in s
        assert "name" in s
        assert "body" in s
        assert "category" in s
        assert "id" not in s

    # Verify every profile has model_tier, allowed_tools, and NO id
    for p in profiles:
        assert "api_name" in p
        assert "name" in p
        assert "model_tier" in p
        assert "model" in p
        assert "provider" in p
        assert "allowed_tools" in p
        assert "id" not in p

    # Check Architect specifically (should be tier_1)
    arch = next(p for p in profiles if p["api_name"] == "architect")
    assert arch["model_tier"] == "tier_1"
    assert arch["model"] == "gemini-pro-latest"

    # Check Developer (should be tier_2)
    dev = next(p for p in profiles if p["api_name"] == "developer")
    assert dev["model_tier"] == "tier_2"
    assert dev["model"] == "gemini-flash-latest"


def test_prepare_and_validate_bundles_with_skills_map(tmp_path):
    """Tests bundle preparation and resolving remote skill IDs into agent profiles."""
    skills_file = tmp_path / "raw_skills.json"
    agents_file = tmp_path / "raw_agents.json"
    remote_skills_map_file = tmp_path / "remote_skills.json"

    sample_skills = [
        {
            "name": "Bash Scripting",
            "api_name": "bash-scripting",
            "category": "devops",
            "description": "Write bash scripts",
            "body": "# Bash SOP",
            "is_active": True,
        }
    ]

    sample_agents = [
        {
            "name": "Software Developer",
            "api_name": "developer",
            "description": "Developer Agent",
            "model_tier": "tier_2",
            "model": "gemini-flash-latest",
            "provider": "Gemini",
            "system_prompt": "You are a dev",
            "allowed_tools": ["terminal_exec"],
            "attached_skill_slugs": ["bash-scripting"],
            "is_active": True,
        }
    ]

    sample_remote_map = {
        "records": [
            {"id": "aisk_remote_12345", "api_name": "bash-scripting"}
        ]
    }

    skills_file.write_text(json.dumps(sample_skills))
    agents_file.write_text(json.dumps(sample_agents))
    remote_skills_map_file.write_text(json.dumps(sample_remote_map))

    out_skills, out_agents = prepare_and_validate_bundles(
        output_dir=tmp_path,
        skills_map_path=remote_skills_map_file,
        skills_source=skills_file,
        agents_source=agents_file,
    )

    assert out_skills.is_file()
    assert out_agents.is_file()

    agent_records = json.loads(out_agents.read_text())
    assert len(agent_records) == 1
    assert agent_records[0]["api_name"] == "developer"
    assert agent_records[0]["model_tier"] == "tier_2"
    # Live skill ID should be wired into ai_skills
    assert agent_records[0]["ai_skills"] == ["aisk_remote_12345"]
    assert "id" not in agent_records[0]
