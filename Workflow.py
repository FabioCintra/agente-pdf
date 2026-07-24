from typing import TypedDict,List
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langgraph.graph import START,END,StateGraph
from langchain_core.messages import HumanMessage,SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from EmbeddingsManage import search_best_chunk_for_context
from prompts import *

#modelo de llm a ser usado
llm = ChatOpenAI(
    model="qwen3:8b",
    base_url="http://localhost:11434/",
    api_key="ollama",
    max_tokens=1200,
    temperature=0.1
)

#criando memorySaver
memory = MemorySaver()

#states
class InputState(TypedDict):
    question: str

class IntenalState(TypedDict):
    best_chunks: List[str]
    question: str

class OutputState(TypedDict):
    answer: str

class AnswerModel(BaseModel):
    answer: str = Field(
        description="Resposta da questao, objetiva e sem enrolacao!"
    )

#Nodes
def get_chunks(state:InputState):
    question = state["question"]
    chunks = search_best_chunk_for_context(question)

    return {
        "best_chunks": chunks,
        "question": question,
    }

def create_answer(state: IntenalState) -> OutputState:
    system_prompt = """
        Voce eh um agente responsavel por responder perguntas do usuario.
        
        OBSERVACAO:
        1 - Utilizer somente o contexto fornecido pelo usuario para responder a pergunta. Nao invente informacao!
    """
    prompt = PROMPT_ANSWER.format(context=state["best_chunks"], question=state["question"]);

    result: AnswerModel = llm.invoke([SystemMessage(content=system_prompt)] + [HumanMessage(content=prompt)])

    return {
        "answer": result.answer,
    }

def create_workflow() :
    builder = StateGraph(InputState)

    builder.add_node("get_chunks", get_chunks)
    builder.add_node("create_answer", create_answer)

    builder.add_edge(START, "get_chunks")
    builder.add_edge("get_chunks", "create_answer")
    builder.add_edge("create_answer", END)

    return builder.compile(checkpointer=memory)


