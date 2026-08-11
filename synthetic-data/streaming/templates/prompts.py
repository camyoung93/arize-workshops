"""Prompt template for the personalize_recommendations LLM span: given the user
query, user history (profile), and reranked recommendations."""

PROMPT_TEMPLATE = (
    "Given the user's request, their profile/watch history, and the reranked recommendations below, "
    "produce a short conversational response recommending 2-3 titles. Use only titles from the list; "
    "do not invent or fabricate details from the user's profile or history.\n\n"
    "User request: {query}\n\n"
    "User profile / watch history:\n{user_history}\n\n"
    "Reranked recommendations:\n{reranked_list}\n\n"
    "Respond with your personalized picks."
)


def format_llm_input(query: str, reranked_docs: list, user_history: str = "") -> str:
    """Format the LLM input from query, user history, and reranked doc list."""
    lines = [f"- {d['id']}: {d['content']}" for d in reranked_docs]
    return PROMPT_TEMPLATE.format(
        query=query,
        user_history=user_history or "(no profile provided)",
        reranked_list="\n".join(lines),
    )
