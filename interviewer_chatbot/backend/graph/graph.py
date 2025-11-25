from langgraph.graph import StateGraph, END
from graph.state import InterviewState
from graph.nodes import (
    setup_node,
    get_answer_node,
    evaluate_question_node,
    generate_question_node,
    final_evaluation_node,
    display_results_node,
    retrieval_decision_node,
    retrieval_node,
    tavily_search_node,
)
from utils.logger import setup_logger
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from config.settings import settings

logger = setup_logger(__name__)

SETUP_NODE = "setup"
GET_ANSWER_NODE = "get_answer"
RETRIEVAL_DECISION_NODE = "retrieval_decision"
RETRIEVAL_NODE = "retrieval"
TAVILY_SEARCH_NODE = "tavily_search"
GENERATE_QUESTION_NODE = "generate_question"
EVALUATE_QUESTION_NODE = "evaluate_question"
FINAL_EVALUATION_NODE = "final_evaluation"
DISPLAY_RESULTS_NODE = "display_results"


def should_retrieve(state: InterviewState) -> str:
    """
    Decide whether to use internal RAG retrieval or Tavily search based on the state.

    Args:
        state (InterviewState): Current interview state containing flags and context.

    Returns:
        str: The name of the next node to execute, either RETRIEVAL_NODE or TAVILY_SEARCH_NODE.
    """
    return RETRIEVAL_NODE if state["needs_retrieval"] else TAVILY_SEARCH_NODE


def should_continue(state: InterviewState) -> str:
    """
    Determine whether the interview should continue, end, or move to evaluation.

    Args:
        state (InterviewState): Current state of the interview.

    Returns:
        str: The name of the next node.
            - END if waiting for user input.
            - EVALUATE_QUESTION_NODE if max steps reached.
            - GET_ANSWER_NODE otherwise.
    """
    if state.get("waiting_for_user", False):
        return END
    if state.get("step", 0) >= state.get("max_steps", 10):
        return EVALUATE_QUESTION_NODE
    return GET_ANSWER_NODE


def should_generate_question(state: InterviewState) -> str:
    """
    Decide whether to generate the next question or perform the final evaluation.

    Args:
        state (InterviewState): Current interview state.

    Returns:
        str: GENERATE_QUESTION_NODE if interview continues,
             FINAL_EVALUATION_NODE if interview ends.
    """
    return (
        GENERATE_QUESTION_NODE
        if state["step"] < state["max_steps"]
        else FINAL_EVALUATION_NODE
    )


def should_start_or_wait(state: InterviewState) -> str:
    """
    Determine if the interview should start immediately or wait for user input.

    Args:
        state (InterviewState): Current interview state.

    Returns:
        str: END if waiting for user input, GET_ANSWER_NODE otherwise.
    """
    return END if state.get("waiting_for_user", False) else GET_ANSWER_NODE


def get_postgres_checkpointer():
    """
    Create PostgreSQL checkpointer with fallback to SQLite and then Memory.

    Returns:
        Checkpoint saver instance (PostgresSaver, SqliteSaver, or MemorySaver)
    """
    try:

        conn = psycopg.connect(settings.database_url)
        checkpointer = PostgresSaver(conn)

        checkpointer.setup()

        logger.info("✅ Using PostgreSQL checkpointer")
        return checkpointer

    except ImportError as e:
        logger.warning("⚠️ PostgreSQL checkpoint package not available: %s", e)
    except psycopg.OperationalError as e:
        logger.warning("⚠️ Cannot connect to PostgreSQL database: %s", e)
        logger.warning("⚠️ Make sure PostgreSQL is running and database exists")
    except Exception as e:
        logger.warning("⚠️ PostgreSQL setup failed: %s", e)

    try:
        import sqlite3

        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        checkpointer = SqliteSaver(conn)
        logger.info("✅ Using SQLite checkpointer (PostgreSQL fallback)")
        return checkpointer
    except Exception as e:
        logger.warning("⚠️ SQLite saver failed: %s", e)

    logger.info("✅ Using in-memory checkpointer (final fallback)")
    return MemorySaver()


def create_interview_graph() -> StateGraph:
    """
    Create and compile the interview state graph with PostgreSQL checkpoints.

    The graph orchestrates the multi-step interview process, including:
        - Setup and initialization.
        - Retrieving or searching context (via RAG or Tavily).
        - Generating and evaluating interview questions.
        - Producing final evaluations and displaying results.

    The flow includes checkpointing support using PostgreSQL (preferred),
    with fallbacks to SQLite and in-memory storage.

    Returns:
        StateGraph: Compiled interview graph ready for execution.
    """
    logger.info("Initializing interview graph with RAG + Tavily search flow...")
    builder = StateGraph(InterviewState)

    builder.add_node(SETUP_NODE, setup_node)
    builder.add_node(GET_ANSWER_NODE, get_answer_node)
    builder.add_node(RETRIEVAL_DECISION_NODE, retrieval_decision_node)
    builder.add_node(RETRIEVAL_NODE, retrieval_node)
    builder.add_node(TAVILY_SEARCH_NODE, tavily_search_node)
    builder.add_node(GENERATE_QUESTION_NODE, generate_question_node)
    builder.add_node(EVALUATE_QUESTION_NODE, evaluate_question_node)
    builder.add_node(FINAL_EVALUATION_NODE, final_evaluation_node)
    builder.add_node(DISPLAY_RESULTS_NODE, display_results_node)

    builder.set_entry_point(SETUP_NODE)

    builder.add_conditional_edges(
        SETUP_NODE,
        should_start_or_wait,
        {GET_ANSWER_NODE: GET_ANSWER_NODE, END: END},
    )

    builder.add_edge(GET_ANSWER_NODE, RETRIEVAL_DECISION_NODE)

    builder.add_conditional_edges(
        RETRIEVAL_DECISION_NODE,
        should_retrieve,
        {RETRIEVAL_NODE: RETRIEVAL_NODE, TAVILY_SEARCH_NODE: TAVILY_SEARCH_NODE},
    )

    builder.add_edge(RETRIEVAL_NODE, GENERATE_QUESTION_NODE)
    builder.add_edge(TAVILY_SEARCH_NODE, GENERATE_QUESTION_NODE)

    builder.add_conditional_edges(
        GENERATE_QUESTION_NODE,
        should_continue,
        {
            GET_ANSWER_NODE: GET_ANSWER_NODE,
            EVALUATE_QUESTION_NODE: EVALUATE_QUESTION_NODE,
            FINAL_EVALUATION_NODE: FINAL_EVALUATION_NODE,
            END: END,
        },
    )

    builder.add_edge(EVALUATE_QUESTION_NODE, FINAL_EVALUATION_NODE)
    builder.add_edge(FINAL_EVALUATION_NODE, DISPLAY_RESULTS_NODE)
    builder.add_edge(DISPLAY_RESULTS_NODE, END)

    checkpointer = get_postgres_checkpointer()

    logger.info("✅ Interview graph successfully compiled with PostgreSQL checkpoints.")
    return builder.compile(checkpointer=checkpointer)


compiled_graph = create_interview_graph()
