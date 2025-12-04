import os
import asyncio
import logging
from typing import TypedDict
from dotenv import load_dotenv

from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    WorkerOptions,
    cli,
    RoomInputOptions,
)
from livekit.plugins import elevenlabs, silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins.langchain import LLMAdapter
from langgraph.graph import StateGraph, END
from services.gemini_client import gemini_client
from contextlib import asynccontextmanager

logger = logging.getLogger("voice-agent")
load_dotenv()


class State(TypedDict):
    """Workflow state containing user input and generated response."""

    user_input: str
    response: str


def echo_node(state: State):
    """
    LangGraph node that generates a response using Gemini.

    Args:
        state: Current workflow state containing user_input.

    Returns:
        Updated state dictionary with a response field.
    """
    try:
        response = gemini_client.generate_content(state["user_input"])
    except Exception as e:
        logger.exception("Error in echo_node")
        response = f"An internal error occurred: {e}"
    return {"response": response}


def create_workflow():
    """
    Creates and compiles the LangGraph workflow.

    Returns:
        A compiled LangGraph workflow object.
    """
    try:
        graph = StateGraph(State)
        graph.add_node("echo", echo_node)
        graph.set_entry_point("echo")
        graph.add_edge("echo", END)
        return graph.compile()
    except Exception as e:
        logger.exception("Failed to create workflow")
        raise


class LangGraphLLM(LLMAdapter):
    """
    LiveKit LLMAdapter wrapper that runs a LangGraph workflow for each chat turn.
    """

    def __init__(self, workflow):
        """
        Args:
            workflow: A compiled LangGraph workflow.
        """
        super().__init__(graph=workflow)
        self._workflow = workflow

    @asynccontextmanager
    async def chat(self, chat_ctx, **kwargs):
        """
        Asynchronous chat generator that runs the LangGraph workflow in a thread.

        Args:
            chat_ctx: LiveKit chat context.
            **kwargs: Additional options.

        Yields:
            An asynchronous generator producing the model response.
        """
        try:
            last_msg = chat_ctx.items[-1]
            user_text = getattr(last_msg, "content", None) or str(last_msg)

            result = await asyncio.to_thread(
                lambda: self._workflow.invoke({"user_input": user_text})
            )
            response = result.get("response", str(result))
        except Exception as e:
            logger.exception("Error inside LLM chat workflow")
            response = f"An internal error occurred: {e}"

        async def generator():
            try:
                if isinstance(response, list):
                    yield " ".join(str(r) for r in response)
                else:
                    yield str(response)
            except Exception as e:
                logger.exception("Error in response generator")
                yield f"An internal error occurred while generating response: {e}"

        yield generator()


class VoiceAgent(Agent):
    """Voice-based conversational agent capable of producing filler responses."""

    def __init__(self):
        """Initializes the agent with base instructions."""
        super().__init__(instructions="You are a helpful assistant.")

    async def on_user_turn_completed(self, turn_ctx, new_message):
        """
        Sends a filler response after the user finishes speaking.

        Args:
            turn_ctx: Turn context from LiveKit.
            new_message: Message object representing the latest user input.
        """

        async def filler_gen():
            yield "Hmm... let me think..."

        try:
            await self.session.say(filler_gen(), add_to_chat_ctx=False)
        except Exception as e:
            logger.exception("Failed to send filler response")


async def entrypoint(ctx: JobContext):
    """
    Main entrypoint for LiveKit worker. Initializes the session and starts the agent.

    Args:
        ctx: JobContext provided by LiveKit runtime.
    """
    try:
        await ctx.connect()
    except Exception as e:
        logger.exception("Failed to connect to context")
        raise

    try:
        session = AgentSession(
            llm=LangGraphLLM(create_workflow()),
            stt="assemblyai/universal-streaming:en",
            tts=elevenlabs.TTS(
                model="eleven_v2_flash",
                voice_id="CwhRBWXzGAHq8TQ4Fs17",
                api_key=os.getenv("ELEVENLABS_API_KEY"),
            ),
            vad=silero.VAD.load(),
            turn_detection=MultilingualModel(),
            preemptive_generation=False,
        )
    except Exception as e:
        logger.exception("Failed to initialize AgentSession")
        raise

    try:
        await session.start(
            agent=VoiceAgent(),
            room=ctx.room,
            room_input_options=RoomInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        )
    except Exception as e:
        logger.exception("Failed to start session")
        raise

    logger.info(f"Agent started in room: {ctx.room.name}")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
