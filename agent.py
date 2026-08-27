import os
import asyncio
from dotenv import load_dotenv
from github import Github, Auth

# Importy LlamaIndex
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import (
    FunctionAgent,
    AgentWorkflow,
    AgentOutput,
    ToolCall,
    ToolCallResult
)

# Ładowanie zmiennych środowiskowych
load_dotenv()

# ==========================================
# 1. KONFIGURACJA LLM (OpenAI)
# ==========================================
llm = OpenAI(
    model="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ==========================================
# 2. KONFIGURACJA GITHUB API
# ==========================================
# 1. Pobieramy TYLKO tekstowy klucz:
github_token = os.getenv("GITHUB_TOKEN")

# 2. Tworzymy obiekt klienta (korzystając z zalecanego Auth.Token):
if github_token:
    git = Github(auth=Auth.Token(github_token))
else:
    git = Github()

# 3. Pobieranie zmiennych repozytorium:
full_repo_name = os.getenv("REPOSITORY")
pr_number = os.getenv("PR_NUMBER")

# 4. Inicjalizacja obiektu 'repo':
if git is not None and full_repo_name:
    repo = git.get_repo(full_repo_name)
else:
    raise ValueError("Brak zmiennej REPOSITORY w środowisku!")
    
# ==========================================
# 3. DEFINICJE NARZĘDZI (TOOLS) - GitHub
# ==========================================
def get_pr_details(pr_number: int) -> dict:
    """Use this tool to get details about a pull request given its number."""
    pull_request = repo.get_pull(pr_number)
    commit_SHAs = [c.sha for c in pull_request.get_commits()]

    return {
        "author": pull_request.user.login,
        "title": pull_request.title,
        "body": pull_request.body,
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": commit_SHAs[-1] if commit_SHAs else None,
        "commit_shas": commit_SHAs
    }


def get_file_content(file_path: str) -> str:
    """Use this tool to fetch the contents of a file from the repository given its path."""
    return repo.get_contents(file_path).decoded_content.decode('utf-8')


def get_commit_details(commit_sha: str) -> list:
    """Use this tool to retrieve information about a commit given its SHA."""
    commit = repo.get_commit(commit_sha)
    changed_files = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


# --- NOWE NARZĘDZIE: Postowanie do GitHuba ---
def post_review_to_github(pr_number: int, comment: str) -> str:
    """Use this tool to post the final review comment to GitHub PR."""
    pull_request = repo.get_pull(pr_number)
    pull_request.create_review(body=comment)
    return f"Review successfully posted to GitHub PR #{pr_number}."


# ==========================================
# 4. ZARZĄDZANIE STANEM (STATE MANAGEMENT)
# ==========================================
async def add_context_to_state(ctx: Context, gathered_contexts: str) -> str:
    """Useful for adding the gathered context to the state."""
    if hasattr(ctx, "session") and hasattr(ctx.session, "state"):
        ctx.session.state["gathered_contexts"] = gathered_contexts
    elif hasattr(ctx, "data"):
        ctx.data["gathered_contexts"] = gathered_contexts
    return "Context added to state successfully."

async def add_comment_to_state(ctx: Context, review_comment: str) -> str:
    """Useful for adding the drafted review comment to the state."""
    if hasattr(ctx, "session") and hasattr(ctx.session, "state"):
        ctx.session.state["review_comment"] = review_comment
    elif hasattr(ctx, "data"):
        ctx.data["review_comment"] = review_comment
    return "Draft comment added to state successfully."

async def add_final_review_to_state(ctx: Context, final_review: str) -> str:
    """Useful for adding the final review to the state."""
    if hasattr(ctx, "session") and hasattr(ctx.session, "state"):
        ctx.session.state["final_review"] = final_review
    elif hasattr(ctx, "data"):
        ctx.data["final_review"] = final_review
    return "Final review added to state successfully."

# Konwersja wszystkich funkcji na narzędzia LlamaIndex
pr_details_tool = FunctionTool.from_defaults(get_pr_details)
file_tool = FunctionTool.from_defaults(get_file_content)
pr_commits_tool = FunctionTool.from_defaults(get_commit_details)
post_review_tool = FunctionTool.from_defaults(post_review_to_github)

context_state_tool = FunctionTool.from_defaults(add_context_to_state)
comment_state_tool = FunctionTool.from_defaults(add_comment_to_state)
final_review_state_tool = FunctionTool.from_defaults(add_final_review_to_state)

# ==========================================
# 5. TWORZENIE AGENTÓW I ORKIESTRACJA
# ==========================================

# --- Agent 1: ContextAgent ---
system_prompt_context = """You are the context gathering agent. You have strictly defined steps:
1. Fetch PR details using `get_pr_details`.
2. Fetch changed files using `get_commit_details`.
3. Fetch any requested files using `get_file_content`.
4. You MUST save the gathered context using the `add_context_to_state` tool.
5. After saving the context, you MUST use the `handoff` tool to pass control back to `CommentorAgent`.
CRITICAL: DO NOT generate a conversational response. ONLY use tools."""

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context from the GitHub repository and saves it to state.",
    tools=[pr_details_tool, file_tool, pr_commits_tool, context_state_tool],
    system_prompt=system_prompt_context,
    can_handoff_to=["CommentorAgent"]
)

# --- Agent 2: CommentorAgent ---
system_prompt_commentor = """You are the commentor agent. Follow these exact steps:
1. If you don't have PR context yet, use the `handoff` tool to pass control to `ContextAgent`.
2. Once you have the context, draft a ~200-300 word review in markdown (mention tests, endpoints, code improvements, and address the author).
3. CRITICAL: YOU MUST NOT output the review directly to the user!
4. You MUST first save your draft using the `add_comment_to_state` tool.
5. Immediately after saving, you MUST use the `handoff` tool to pass control to `ReviewAndPostingAgent`.
DO NOT answer directly, always use the tools!"""

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment.",
    tools=[comment_state_tool],
    system_prompt=system_prompt_commentor,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

# --- Agent 3: ReviewAndPostingAgent ---
system_prompt_review_posting = """You are the Review and Posting agent.
1. If there is no drafted review in the state, use the `handoff` tool to pass control to `CommentorAgent`.
2. Once CommentorAgent hands off the draft to you, verify it.
3. Save the final review using the `add_final_review_to_state` tool.
4. Finally, post the review to GitHub using the `post_review_to_github` tool.
CRITICAL: Do not output any text until the review is successfully posted via the tool."""

review_and_posting_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the drafted PR comment, ensures it meets criteria, and posts it to GitHub.",
    tools=[post_review_tool, final_review_state_tool],
    system_prompt=system_prompt_review_posting,
    can_handoff_to=["CommentorAgent"]
)

# --- Workflow (Orkiestracja) ---
workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_posting_agent],
    root_agent=review_and_posting_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review": "",
    },
)

# ==========================================
# 6. GŁÓWNA PĘTLA WYKONAWCZA (STREAMING WORKFLOW)
# ==========================================
async def main():
    # print("Workflow gotowe! Wpisz zapytanie:")
    query = "Write a review for PR: " + pr_number
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format())
    current_agent = None

    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"\nCurrent agent: {current_agent}")

        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\n\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])

        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")

        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        if git:
            git.close()