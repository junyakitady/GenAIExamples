from typing import Any
import os
import streamlit as st
import langchain
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.chains.llm import LLMChain
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import vertexai
import bigframes
import bigframes.pandas as bpd

# init Vertex AI
PROJECT_ID = os.environ.get("PROJECT_ID")
REGION = "us-central1"
BQ_LOCATION = os.environ.get("BQ_LOCATION")
BQ_DATASET = os.environ.get("BQ_DATASET")
BQ_TABLE = os.environ.get("BQ_TABLE")
vertexai.init(project=PROJECT_ID, location=REGION)

# layoout page
st.title("Ask BigQuery")
st.text("2023年度 公示地価情報を保持するテーブルに自然言語で質問できます。")
# setup bigframes and PandasAgent
bpd.options.bigquery.project = PROJECT_ID
bpd.options.bigquery.location = BQ_LOCATION
table = PROJECT_ID+"."+BQ_DATASET+"."+BQ_TABLE
bdf = bpd.read_gbq(table)
st.dataframe(bdf.head(5))

with st.form('my_form'):
    prompt = st.text_area(label='Prompt', value="""東京都杉並区の中で地価を高い順に5個教えてください。
以下のフォーマットで表形式で回答してください。
[最寄り駅] [住所] [地価]""")
    submitted = st.form_submit_button('Submit')

# Text model instance integrated with LangChain
llm = VertexAI(model_name="gemini-pro", max_output_tokens=1024, temperature=0)

AGENT_TEMPLATE = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`.
You should use the tools below to answer the question posed of you:

python_repl_ast: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be "python_repl_ast"
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question


This is the result of `print(df.head())`:
{df_head}
You can only use above columns for query.

Begin!
Question: {input}
{agent_scratchpad}
"""


def hack_create_pandas_dataframe_agent(  # hack Langchain as bigframes.dataframe.DataFrame is not instance of pd.DataFrame
    llm: BaseLanguageModel,
    df: Any,
    verbose: bool = False,
) -> AgentExecutor:

    tools = [PythonAstREPLTool(locals={"df": df})]
    prompt = PromptTemplate(template=AGENT_TEMPLATE, input_variables=["input", "agent_scratchpad", "df_head"])
    partial_prompt = prompt.partial()
    partial_prompt = partial_prompt.partial(df_head=str(df.head(5).to_markdown()))
    llm_chain = LLMChain(llm=llm, prompt=partial_prompt)
    tool_names = [tool.name for tool in tools]
    agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools=tool_names)
    new_agent = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=verbose, return_intermediate_steps=verbose, max_iterations=5)
    return new_agent


# agent = create_pandas_dataframe_agent(llm, bdf, verbose=True)
pandasagent = hack_create_pandas_dataframe_agent(llm, bdf, verbose=True)

if submitted:
    response = pandasagent.invoke(prompt)
    st.text("Output")
    st.markdown(response["output"])
    st.text("Check pandas")
    command = response["intermediate_steps"][len(response["intermediate_steps"])-1][0].tool_input
    st.text(command)
    tool = PythonAstREPLTool(locals={"df": bdf})
    tempdf = tool.invoke(command)
    if isinstance(tempdf, bigframes.dataframe.DataFrame):
        st.dataframe(tempdf)
    else:
        st.write(tempdf)
