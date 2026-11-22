"""GitHub Integration tools for Valstorm Agent Runtime.

Provides first-class tools for reading and writing GitHub repositories, commits,
push/merge activity, branches, pull requests, code reviews, issues, file contents,
search, and GitHub Actions workflow runs via the Valstorm GitHub plugin (`platform.github`).
"""

import json
from typing import Any, Dict, List, Optional

from core.tools import ToolRegistry, tool
from tools.valstorm_client import ValstormApiClient


def create_github_tools(client: Optional[ValstormApiClient] = None) -> List[Any]:
    """Creates GitHub plugin tool functions bound to a specific ValstormApiClient instance."""
    api_client = client or ValstormApiClient()

    async def _invoke(method: str, **kwargs: Any) -> str:
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        res = await api_client.invoke_platform_method(f"github/{method}", **clean_kwargs)
        return json.dumps(res, indent=2, default=str)

    @tool
    async def github_list_repositories() -> str:
        """List GitHub repositories accessible to the connected user AND the organization (`Org_github`)."""
        return await _invoke("list_repositories")

    @tool
    async def github_rest_request(
        path: str,
        method: str = "GET",
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        api_version: str = "2026-03-10",
        prefer_org: bool = False,
    ) -> str:
        """Execute a direct GitHub REST API (`X-GitHub-Api-Version: 2026-03-10`) request with automatic `Org_github` + User Scope token resolution.

        Args:
            path: GitHub REST API path (e.g. '/repos/Valstorm/monorepo/commits').
            method: HTTP method ('GET', 'POST', 'PUT', 'PATCH', 'DELETE').
            params: Optional query parameters dictionary.
            json_body: Optional JSON body dictionary.
            api_version: GitHub REST API version header (default: '2026-03-10').
            prefer_org: If True, prefer the `Org_github` credential over personal user token.
        """
        return await _invoke(
            "rest_request",
            method=method,
            path=path,
            params=params,
            json_body=json_body,
            api_version=api_version,
            prefer_org=prefer_org,
        )

    @tool
    async def github_get_repository(owner: str, repo: str) -> str:
        """Fetch metadata, default branch, visibility, and stats for a specific GitHub repository.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
        """
        return await _invoke("get_repository", owner=owner, repo=repo)

    @tool
    async def github_list_commits(
        owner: str,
        repo: str,
        sha: Optional[str] = None,
        path: Optional[str] = None,
        author: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        per_page: int = 20,
        page: int = 1,
    ) -> str:
        """List commits (including push and merge commits with commit_title, message, author, date, and is_merge) in a GitHub repository.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            sha: Optional branch name, tag, or commit SHA to start listing from.
            path: Optional file path filter.
            author: Optional GitHub login or email address to filter commits by author.
            since: Optional ISO-8601 timestamp (YYYY-MM-DDTHH:MM:SSZ).
            until: Optional ISO-8601 timestamp (YYYY-MM-DDTHH:MM:SSZ).
            per_page: Number of commits to return (default: 20, max: 100).
            page: Page number (default: 1).
        """
        return await _invoke(
            "list_commits",
            owner=owner,
            repo=repo,
            sha=sha,
            path=path,
            author=author,
            since=since,
            until=until,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_get_commit(owner: str, repo: str, ref: str) -> str:
        """Get detailed information, commit title/message, stats, and changed files for a specific commit SHA or ref.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            ref: Commit SHA, branch name, or tag.
        """
        return await _invoke("get_commit", owner=owner, repo=repo, ref=ref)

    @tool
    async def github_compare_commits(owner: str, repo: str, base: str, head: str) -> str:
        """Compare two branches, tags, or commit SHAs (`base...head`) to inspect commits ahead/behind and file diffs.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            base: Base branch or commit SHA (e.g. 'main').
            head: Head branch or commit SHA (e.g. 'feature-branch').
        """
        return await _invoke("compare_commits", owner=owner, repo=repo, base=base, head=head)

    @tool
    async def github_list_repository_activity(
        owner: str,
        repo: str,
        activity_type: Optional[str] = None,
        ref: Optional[str] = None,
        actor: Optional[str] = None,
        time_period: Optional[str] = None,
        per_page: int = 20,
    ) -> str:
        """List recent repository push, merge, force_push, and branch activity events (`push`, `pr_merge`, `merge_queue_merge`).

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            activity_type: Optional filter ('push', 'force_push', 'branch_creation', 'branch_deletion', 'pr_merge', 'merge_queue_merge').
            ref: Optional Git ref filter (e.g. 'refs/heads/main').
            actor: Optional GitHub username of actor.
            time_period: Optional time period ('day', 'week', 'month', 'quarter', 'year').
            per_page: Maximum records to return (default: 20).
        """
        return await _invoke(
            "list_repository_activity",
            owner=owner,
            repo=repo,
            activity_type=activity_type,
            ref=ref,
            actor=actor,
            time_period=time_period,
            per_page=per_page,
        )

    @tool
    async def github_list_branches(
        owner: str,
        repo: str,
        protected: Optional[bool] = None,
        per_page: int = 30,
        page: int = 1,
    ) -> str:
        """List branches in a GitHub repository.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            protected: Optional filter for branch protection status.
            per_page: Number of branches to return (default: 30).
            page: Page number (default: 1).
        """
        return await _invoke(
            "list_branches",
            owner=owner,
            repo=repo,
            protected=protected,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_create_branch(
        owner: str,
        repo: str,
        branch: str,
        from_branch: str = "main",
        sha: Optional[str] = None,
    ) -> str:
        """Create a new branch in a GitHub repository from a base branch or commit SHA.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            branch: New branch name to create.
            from_branch: Base branch to branch off of (default: 'main').
            sha: Optional explicit commit SHA to branch from.
        """
        return await _invoke(
            "create_branch",
            owner=owner,
            repo=repo,
            branch=branch,
            from_branch=from_branch,
            sha=sha,
        )

    @tool
    async def github_delete_branch(owner: str, repo: str, branch: str) -> str:
        """Delete a branch reference in a GitHub repository.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            branch: Branch name to delete.
        """
        return await _invoke("delete_branch", owner=owner, repo=repo, branch=branch)

    @tool
    async def github_list_pull_requests(
        owner: str,
        repo: str,
        state: str = "all",
        head: Optional[str] = None,
        base: Optional[str] = None,
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 20,
        page: int = 1,
    ) -> str:
        """List Pull Requests in a GitHub repository (supports state='open', 'closed', or 'all', including merged status).

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            state: PR state filter ('open', 'closed', or 'all').
            head: Optional head user/branch filter.
            base: Optional base branch filter.
            sort: Sort field ('created', 'updated', 'popularity', 'long-running').
            direction: Sort direction ('asc' or 'desc').
            per_page: Number of PRs to return (default: 20).
            page: Page number (default: 1).
        """
        return await _invoke(
            "list_pull_requests",
            owner=owner,
            repo=repo,
            state=state,
            head=head,
            base=base,
            sort=sort,
            direction=direction,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_get_pull_request(owner: str, repo: str, pull_number: int) -> str:
        """Fetch full details of a specific Pull Request by number.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            pull_number: Pull Request number.
        """
        return await _invoke("get_pull_request", owner=owner, repo=repo, pull_number=pull_number)

    @tool
    async def github_create_pull_request(
        owner: str,
        repo: str,
        title: str,
        head: str,
        base: str = "main",
        body: Optional[str] = None,
        draft: bool = False,
    ) -> str:
        """Create a Pull Request in a target GitHub repository.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            title: Pull Request title.
            head: Branch containing changes.
            base: Target branch to merge into (default: 'main').
            body: Optional PR description body.
            draft: Whether to open as a draft PR.
        """
        return await _invoke(
            "create_pull_request",
            owner=owner,
            repo=repo,
            title=title,
            head=head,
            base=base,
            body=body,
            draft=draft,
        )

    @tool
    async def github_update_pull_request(
        owner: str,
        repo: str,
        pull_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        base: Optional[str] = None,
    ) -> str:
        """Update an existing Pull Request's title, body, state ('open' or 'closed'), or base branch.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            pull_number: Pull Request number.
            title: Optional updated title.
            body: Optional updated description.
            state: Optional state ('open' or 'closed').
            base: Optional target base branch.
        """
        return await _invoke(
            "update_pull_request",
            owner=owner,
            repo=repo,
            pull_number=pull_number,
            title=title,
            body=body,
            state=state,
            base=base,
        )

    @tool
    async def github_merge_pull_request(
        owner: str,
        repo: str,
        pull_number: int,
        commit_title: Optional[str] = None,
        commit_message: Optional[str] = None,
        merge_method: str = "squash",
        sha: Optional[str] = None,
    ) -> str:
        """Merge an open Pull Request using 'merge', 'squash', or 'rebase'.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            pull_number: Pull Request number.
            commit_title: Optional custom merge commit title.
            commit_message: Optional custom merge commit message.
            merge_method: Merge strategy ('merge', 'squash', or 'rebase').
            sha: Optional head SHA that must match to allow merge.
        """
        return await _invoke(
            "merge_pull_request",
            owner=owner,
            repo=repo,
            pull_number=pull_number,
            commit_title=commit_title,
            commit_message=commit_message,
            merge_method=merge_method,
            sha=sha,
        )

    @tool
    async def github_list_pull_request_files(
        owner: str,
        repo: str,
        pull_number: int,
        per_page: int = 50,
        page: int = 1,
    ) -> str:
        """List changed files and patch diffs in a Pull Request.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            pull_number: Pull Request number.
            per_page: Max files per page (default: 50).
            page: Page number (default: 1).
        """
        return await _invoke(
            "list_pull_request_files",
            owner=owner,
            repo=repo,
            pull_number=pull_number,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_list_pull_request_commits(
        owner: str,
        repo: str,
        pull_number: int,
        per_page: int = 50,
        page: int = 1,
    ) -> str:
        """List commits belonging to a specific Pull Request.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            pull_number: Pull Request number.
            per_page: Max commits per page (default: 50).
            page: Page number (default: 1).
        """
        return await _invoke(
            "list_pull_request_commits",
            owner=owner,
            repo=repo,
            pull_number=pull_number,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_create_pull_request_review(
        owner: str,
        repo: str,
        pull_number: int,
        event: str = "COMMENT",
        body: Optional[str] = None,
        commit_id: Optional[str] = None,
    ) -> str:
        """Submit a review ('APPROVE', 'REQUEST_CHANGES', or 'COMMENT') on a Pull Request.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            pull_number: Pull Request number.
            event: Review action ('APPROVE', 'REQUEST_CHANGES', or 'COMMENT').
            body: Review comment text.
            commit_id: Optional SHA of the commit being reviewed.
        """
        return await _invoke(
            "create_pull_request_review",
            owner=owner,
            repo=repo,
            pull_number=pull_number,
            event=event,
            body=body,
            commit_id=commit_id,
        )

    @tool
    async def github_list_issues(
        owner: str,
        repo: str,
        state: str = "open",
        labels: Optional[str] = None,
        assignee: Optional[str] = None,
        creator: Optional[str] = None,
        per_page: int = 20,
        page: int = 1,
    ) -> str:
        """List Issues in a GitHub repository.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            state: Issue state ('open', 'closed', or 'all').
            labels: Optional comma-separated label names.
            assignee: Optional assignee username.
            creator: Optional creator username.
            per_page: Max issues to return (default: 20).
            page: Page number (default: 1).
        """
        return await _invoke(
            "list_issues",
            owner=owner,
            repo=repo,
            state=state,
            labels=labels,
            assignee=assignee,
            creator=creator,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_get_issue(owner: str, repo: str, issue_number: int) -> str:
        """Fetch full details of a specific GitHub Issue by number.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            issue_number: Issue number.
        """
        return await _invoke("get_issue", owner=owner, repo=repo, issue_number=issue_number)

    @tool
    async def github_create_issue(
        owner: str,
        repo: str,
        title: str,
        body: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> str:
        """Create an Issue in a target GitHub repository.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            title: Issue title.
            body: Optional issue description body.
            labels: Optional list of label names.
            assignees: Optional list of GitHub usernames to assign.
        """
        return await _invoke(
            "create_issue",
            owner=owner,
            repo=repo,
            title=title,
            body=body,
            labels=labels,
            assignees=assignees,
        )

    @tool
    async def github_update_issue(
        owner: str,
        repo: str,
        issue_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None,
        state: Optional[str] = None,
        state_reason: Optional[str] = None,
        labels: Optional[List[str]] = None,
        assignees: Optional[List[str]] = None,
    ) -> str:
        """Update an existing GitHub Issue (title, body, state, labels, or assignees).

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            issue_number: Issue number.
            title: Optional updated title.
            body: Optional updated body.
            state: Optional state ('open' or 'closed').
            state_reason: Optional reason ('completed', 'not_planned', or 'reopened').
            labels: Optional replacement label list.
            assignees: Optional replacement assignee list.
        """
        return await _invoke(
            "update_issue",
            owner=owner,
            repo=repo,
            issue_number=issue_number,
            title=title,
            body=body,
            state=state,
            state_reason=state_reason,
            labels=labels,
            assignees=assignees,
        )

    @tool
    async def github_list_issue_comments(
        owner: str,
        repo: str,
        issue_number: int,
        per_page: int = 30,
        page: int = 1,
    ) -> str:
        """List comments on a GitHub Issue or Pull Request.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            issue_number: Issue or Pull Request number.
            per_page: Max comments to return (default: 30).
            page: Page number (default: 1).
        """
        return await _invoke(
            "list_issue_comments",
            owner=owner,
            repo=repo,
            issue_number=issue_number,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_create_issue_comment(
        owner: str,
        repo: str,
        issue_number: int,
        body: str,
    ) -> str:
        """Add a comment to a GitHub Issue or Pull Request.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            issue_number: Issue or Pull Request number.
            body: Comment Markdown text.
        """
        return await _invoke(
            "create_issue_comment",
            owner=owner,
            repo=repo,
            issue_number=issue_number,
            body=body,
        )

    @tool
    async def github_get_file_contents(
        owner: str,
        repo: str,
        path: str = "",
        ref: Optional[str] = None,
    ) -> str:
        """Read a file (with UTF-8 decoded_content) or list a directory from a GitHub repository.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            path: File or directory path inside the repository.
            ref: Optional branch, tag, or commit SHA.
        """
        return await _invoke("get_file_contents", owner=owner, repo=repo, path=path, ref=ref)

    @tool
    async def github_create_or_update_file(
        owner: str,
        repo: str,
        path: str,
        message: str,
        content: str,
        branch: Optional[str] = None,
        sha: Optional[str] = None,
    ) -> str:
        """Create or update a file directly in a GitHub repository with a commit message.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            path: File path inside the repository.
            message: Git commit message.
            content: Plain-text file content to commit.
            branch: Optional target branch name.
            sha: Optional blob SHA if updating an existing file (auto-resolved if omitted).
        """
        return await _invoke(
            "create_or_update_file",
            owner=owner,
            repo=repo,
            path=path,
            message=message,
            content=content,
            branch=branch,
            sha=sha,
        )

    @tool
    async def github_delete_file(
        owner: str,
        repo: str,
        path: str,
        message: str,
        branch: Optional[str] = None,
        sha: Optional[str] = None,
    ) -> str:
        """Delete a file from a GitHub repository with a commit message.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            path: File path inside the repository.
            message: Git commit message.
            branch: Optional target branch name.
            sha: Optional blob SHA (auto-resolved if omitted).
        """
        return await _invoke(
            "delete_file",
            owner=owner,
            repo=repo,
            path=path,
            message=message,
            branch=branch,
            sha=sha,
        )

    @tool
    async def github_search_code(
        query: str,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        per_page: int = 20,
        page: int = 1,
    ) -> str:
        """Search code across GitHub repositories.

        Args:
            query: Search query keywords/qualifiers.
            owner: Optional repository owner or organization filter.
            repo: Optional repository name filter.
            per_page: Max results (default: 20).
            page: Page number (default: 1).
        """
        return await _invoke("search_code", query=query, owner=owner, repo=repo, per_page=per_page, page=page)

    @tool
    async def github_search_commits(
        query: str,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        sort: Optional[str] = None,
        order: str = "desc",
        per_page: int = 20,
        page: int = 1,
    ) -> str:
        """Search commits across GitHub repositories.

        Args:
            query: Commit search query.
            owner: Optional owner/org filter.
            repo: Optional repository filter.
            sort: Optional sort ('author-date' or 'committer-date').
            order: Sort order ('desc' or 'asc').
            per_page: Max results (default: 20).
            page: Page number (default: 1).
        """
        return await _invoke(
            "search_commits",
            query=query,
            owner=owner,
            repo=repo,
            sort=sort,
            order=order,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_search_issues_and_prs(
        query: str,
        owner: Optional[str] = None,
        repo: Optional[str] = None,
        sort: Optional[str] = None,
        order: str = "desc",
        per_page: int = 20,
        page: int = 1,
    ) -> str:
        """Search Issues and Pull Requests across GitHub repositories.

        Args:
            query: Search query string.
            owner: Optional owner/org filter.
            repo: Optional repository filter.
            sort: Optional sort ('comments', 'reactions', 'created', 'updated').
            order: Sort order ('desc' or 'asc').
            per_page: Max results (default: 20).
            page: Page number (default: 1).
        """
        return await _invoke(
            "search_issues_and_prs",
            query=query,
            owner=owner,
            repo=repo,
            sort=sort,
            order=order,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_list_workflow_runs(
        owner: str,
        repo: str,
        branch: Optional[str] = None,
        event: Optional[str] = None,
        status: Optional[str] = None,
        actor: Optional[str] = None,
        workflow_id: Optional[str] = None,
        per_page: int = 20,
        page: int = 1,
    ) -> str:
        """List GitHub Actions workflow runs for a repository.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            branch: Optional branch filter.
            event: Optional trigger event filter ('push', 'pull_request', 'workflow_dispatch').
            status: Optional status filter ('completed', 'in_progress', 'queued', 'failure', 'success').
            actor: Optional triggering user login.
            workflow_id: Optional workflow file name (e.g. 'ci.yml') or ID.
            per_page: Max runs to return (default: 20).
            page: Page number (default: 1).
        """
        return await _invoke(
            "list_workflow_runs",
            owner=owner,
            repo=repo,
            branch=branch,
            event=event,
            status=status,
            actor=actor,
            workflow_id=workflow_id,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_get_workflow_run(owner: str, repo: str, run_id: int) -> str:
        """Get details of a specific GitHub Actions workflow run.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            run_id: Workflow run ID.
        """
        return await _invoke("get_workflow_run", owner=owner, repo=repo, run_id=run_id)

    @tool
    async def github_list_workflow_run_jobs(
        owner: str,
        repo: str,
        run_id: int,
        filter_mode: str = "latest",
        per_page: int = 30,
        page: int = 1,
    ) -> str:
        """List jobs and steps for a specific GitHub Actions workflow run.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            run_id: Workflow run ID.
            filter_mode: Filter mode ('latest' or 'all').
            per_page: Max jobs to return (default: 30).
            page: Page number (default: 1).
        """
        return await _invoke(
            "list_workflow_run_jobs",
            owner=owner,
            repo=repo,
            run_id=run_id,
            filter_mode=filter_mode,
            per_page=per_page,
            page=page,
        )

    @tool
    async def github_trigger_workflow_dispatch(
        owner: str,
        repo: str,
        workflow_id: str,
        ref: str = "main",
        inputs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Trigger a `workflow_dispatch` event for a GitHub Actions workflow.

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            workflow_id: Workflow filename (e.g. 'deploy.yml') or ID.
            ref: Git branch or tag ref (default: 'main').
            inputs: Optional workflow_dispatch input parameters.
        """
        return await _invoke(
            "trigger_workflow_dispatch",
            owner=owner,
            repo=repo,
            workflow_id=workflow_id,
            ref=ref,
            inputs=inputs,
        )

    @tool
    async def github_rerun_workflow_run(
        owner: str,
        repo: str,
        run_id: int,
        failed_only: bool = False,
    ) -> str:
        """Re-run a GitHub Actions workflow run (or only its failed jobs).

        Args:
            owner: Repository owner or organization login.
            repo: Repository name.
            run_id: Workflow run ID.
            failed_only: If True, only re-run failed jobs.
        """
        return await _invoke(
            "rerun_workflow_run",
            owner=owner,
            repo=repo,
            run_id=run_id,
            failed_only=failed_only,
        )

    @tool
    async def github_list_ci_deployments(limit: int = 25) -> str:
        """List recent GitHub Actions CI/CD workflow_run deployment records (`cicd_deployment`).

        Args:
            limit: Maximum deployment records to return (default: 25).
        """
        return await _invoke("list_ci_deployments", limit=limit)

    return [
        github_list_repositories,
        github_rest_request,
        github_get_repository,
        github_list_commits,
        github_get_commit,
        github_compare_commits,
        github_list_repository_activity,
        github_list_branches,
        github_create_branch,
        github_delete_branch,
        github_list_pull_requests,
        github_get_pull_request,
        github_create_pull_request,
        github_update_pull_request,
        github_merge_pull_request,
        github_list_pull_request_files,
        github_list_pull_request_commits,
        github_create_pull_request_review,
        github_list_issues,
        github_get_issue,
        github_create_issue,
        github_update_issue,
        github_list_issue_comments,
        github_create_issue_comment,
        github_get_file_contents,
        github_create_or_update_file,
        github_delete_file,
        github_search_code,
        github_search_commits,
        github_search_issues_and_prs,
        github_list_workflow_runs,
        github_get_workflow_run,
        github_list_workflow_run_jobs,
        github_trigger_workflow_dispatch,
        github_rerun_workflow_run,
        github_list_ci_deployments,
    ]


def register_github_tools(
    registry: ToolRegistry,
    client: Optional[ValstormApiClient] = None,
    token: Optional[str] = None,
    env: str = "local",
) -> ToolRegistry:
    """Registers all GitHub integration tools into a ToolRegistry."""
    api_client = client or ValstormApiClient(token=token, env=env)
    tools = create_github_tools(client=api_client)
    for t in tools:
        registry.register(t)
    return registry
