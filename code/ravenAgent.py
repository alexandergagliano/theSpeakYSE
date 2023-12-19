from typing import Any, Callable, List, Dict, Optional, Sequence

from langchain_core.pydantic_v1 import Field, root_validator
from langchain.agents.agent import Agent

from ravenAPI import *

class RavenAgent(Agent):
    """Agent for the MRKL chain."""

    output_parser: RavenOutputParser = Field(default_factory=RavenOutputParser)

    @classmethod
    def _get_default_output_parser(cls, **kwargs: Any) -> RavenOutputParser:
        return RavenOutputParser()

    @property
    def _agent_type(self) -> str:
        """Return Identifier of agent type."""
        return AgentType.ZERO_SHOT_REACT_DESCRIPTION

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "Observation: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Thought:"

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        template: str = RAVEN_PROMPT,
        input_variables: Optional[List[str]] = None,
    ) -> RavenPromptTemplate:
        """Create prompt in the style of the zero shot agent.
        Args:
            tools: List of tools the agent will have access to, used to format the
                prompt.
            prefix: String to put before the list of tools.
            suffix: String to put after the list of tools.
            input_variables: List of input variables the final prompt will expect.

        Returns:
            A PromptTemplate with the template assembled from the pieces here.
        """
        if input_variables is None:
            input_variables = ["input", "agent_scratchpad"]
        return RavenPromptTemplate(template=template, input_variables=input_variables)
    
    @root_validator()
    def validate_prompt(cls, values: Dict) -> Dict:
        """Validate that prompt matches format."""
        print(values)
        return values
