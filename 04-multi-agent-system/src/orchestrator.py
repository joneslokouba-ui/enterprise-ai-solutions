from src.agents import research_agent, analysis_agent, summary_agent


def run_multi_agent(topic):
    """Orchestrates all three agents working together."""

    # Agent 1: Research
    research = research_agent(topic)

    # Agent 2: Analysis
    analysis = analysis_agent(research)

    # Agent 3: Summary
    summary = summary_agent(analysis)

    return {
        "research": research,
        "analysis": analysis,
        "summary": summary
    }