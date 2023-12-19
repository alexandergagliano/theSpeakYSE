from typing import List, Literal, Union
from langchain.prompts import StringPromptTemplate
from langchain.agents import AgentOutputParser
from langchain.tools import BaseTool

# TODO: incorporate these in prompt
from langchain_community.agent_toolkits.sql.prompt import (
    SQL_FUNCTIONS_SUFFIX,
    SQL_PREFIX,
    SQL_SUFFIX,
)

from langchain.schema import AgentAction, AgentFinish, OutputParserException

RAVEN_PROMPT = """
{raven_tools}
User Query: {input}

Please pick one or multiple functions from the above options that best answers the user query and fill in the appropriate arguments. Then, return the output of the evaluated function.
<human_end>"""

# Set up a prompt template
class RavenPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[BaseTool]

    def format(self, **kwargs) -> str:
        prompt = "<human>:\n"
        for tool in self.tools:
            try:
                func_signature, func_docstring = tool.description.split(" - ", 1)
            except:
                func_signature = tool.name
                func_docstring = tool.description
            prompt += f'\nOPTION:\n<func_start>def {func_signature}<func_end>\n<docstring_start>\n"""\n{func_docstring}\n"""\n<docstring_end>\n'
        kwargs["raven_tools"] = prompt
        return self.template.format(**kwargs).replace("{{", "{").replace("}}", "}")
    
class RavenOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        #if "Initial Answer:" in llm_output:
        return AgentFinish(
            return_values={
                "output": llm_output.strip()
                #.split("\n")[1]
                #.replace("Initial Answer: ", "")
                #.strip()
            },
            log=llm_output,
        )
        #else:
        #    raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")

