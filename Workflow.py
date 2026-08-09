from typing import TypedDict,List,Literal

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel,Field
from langchain_openai import ChatOpenAI
from langgraph.graph import START,END,StateGraph,MessagesState
from langchain_core.messages import HumanMessage,SystemMessage,AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama

from EmbeddingsManage import search_best_chunk_for_context
from prompts import *

#criando memorySaver
memory = MemorySaver()

#states
class InputState(TypedDict):
    question: str

class IntenalState(MessagesState):
    route: Literal["get_chunks", "simple_answer"]
    best_chunks: List[str]

class OutputState(MessagesState):
    pass

class AnswerModel(BaseModel):
    answer: str = Field(
        description="Resposta da questao, objetiva e sem enrolacao!"
    )

class RouteModel(BaseModel):
    route: Literal["get_chunks", "simple_answer"]

#modelo de llm a ser usado
llm = ChatOllama(
    model="qwen3:8b",
    base_url="http://localhost:11434",
    temperature=0.1,
)
llm_with_structure = llm.with_structured_output(AnswerModel, method="json_schema", include_raw=False)
router_llm = llm.with_structured_output(RouteModel, method="json_schema", include_raw=False)

#Nodes
def first_node(state: InputState) -> IntenalState:
    return {
        "messages": [HumanMessage(content=state["question"])],
    }

def route_node(state: IntenalState) :
    recent_messages = state["messages"][-4:]

    router_prompt = HumanMessage(
        content="""
            Escolha exatamente uma rota:

            - simple_answer:
              somente para saudações, agradecimentos e conversa casual.

            - get_chunks:
              para perguntas que pedem informação, definição,
              explicação, comparação ou conteúdo dos documentos.

            Em caso de dúvida, escolha get_chunks.
            """
    )

    decision: RouteModel = router_llm.invoke(
        [
            SystemMessage(
                content="Você é responsável por classificar a pergunta do usuário."
            ),
            *recent_messages,
            router_prompt
        ]
    )

    return  decision.route


def simple_answer(state: IntenalState) -> OutputState:
    system_prompt = """
            Voce eh um agente responsavel por responder perguntas do usuario.

            OBSERVACAO:
            1 - Utilizer somente o contexto fornecido pelo usuario para responder a pergunta. Nao invente informacao!
        """

    recent_messages = state["messages"][-6:]

    result: AnswerModel = llm_with_structure.invoke(
        [SystemMessage(content=system_prompt)] + [*recent_messages],
    )

    return {
        "messages": [AIMessage(content=result.answer)],
    }

def get_chunks(state:IntenalState):
    recent_messages = "\n\n".join(msg.content for msg in state["messages"][-10:])

    chunks = search_best_chunk_for_context(recent_messages)

    return {
        "best_chunks": chunks,
    }

def create_answer(state: IntenalState) -> OutputState:
    system_prompt = f"""
        Voce eh um agente responsavel por responder perguntas do usuario.
        
        OBSERVACAO:
        1 - Utilizer somente o contexto fornecido pelo usuario para responder a pergunta. Nao invente informacao!
        
        Contexto:
        {state["best_chunks"]}
    """
    recent_messages = state["messages"][-6:]

    result: AnswerModel = llm_with_structure.invoke(
        [
            SystemMessage(content=system_prompt),
            *recent_messages
        ]
    )
    return {
        "messages": [AIMessage(content=result.answer)],
    }

def create_workflow() -> CompiledStateGraph:
    builder = StateGraph(
        IntenalState,
        input_schema= InputState,
        output_schema= OutputState
    )

    builder.add_node("first_node", first_node)
    builder.add_node("get_chunks", get_chunks)
    builder.add_node("create_answer", create_answer)
    builder.add_node("simple_answer", simple_answer)


    builder.add_edge(START, "first_node")
    builder.add_conditional_edges("first_node", route_node)
    builder.add_edge("simple_answer", END)
    builder.add_edge("get_chunks", "create_answer")
    builder.add_edge("create_answer", END)

    return builder.compile(checkpointer=memory)


