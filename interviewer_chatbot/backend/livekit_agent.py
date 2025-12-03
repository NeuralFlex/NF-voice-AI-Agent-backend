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

logger = logging.getLogger("voice-agent")
load_dotenv()


# -------------------------
# Define LangGraph workflow
# -------------------------
class State(TypedDict):
    user_input: str
    response: str


def echo_node(state: State):
    """Simple echo node: returns user input as response"""
    return {"response": state["user_input"]}


def create_workflow():
    graph = StateGraph(State)
    graph.add_node("echo", echo_node)
    graph.set_entry_point("echo")
    graph.add_edge("echo", END)
    return graph.compile()


# -------------------------
# LangGraph LLM Adapter
# -------------------------
from contextlib import asynccontextmanager


class LangGraphLLM(LLMAdapter):
    """Wrap a LangGraph workflow for LiveKit."""

    def __init__(self, workflow):
        super().__init__(graph=workflow)
        self._workflow = workflow

    @asynccontextmanager
    async def chat(self, chat_ctx, **kwargs):
        last_msg = chat_ctx.items[-1]
        user_text = getattr(last_msg, "content", None) or str(last_msg)

        # Run workflow in a thread
        result = await asyncio.to_thread(
            lambda: self._workflow.invoke({"user_input": user_text})
        )
        response = result.get("response", str(result))

        # Always yield a proper async generator of strings
        async def generator():
            if isinstance(response, list):
                yield " ".join(str(r) for r in response)
            else:
                yield str(response)

        yield generator()


# -------------------------
# Agent Implementation
# -------------------------
class VoiceAgent(Agent):
    """Agent that can produce filler and main responses"""

    def __init__(self):
        super().__init__(instructions="You are a helpful assistant.")
        self._workflow_llm = LangGraphLLM(create_workflow())

    async def on_user_turn_completed(self, turn_ctx, new_message):
        # -----------------
        # Filler response
        # -----------------
        async def filler_gen():
            yield "Hmm... let me think..."

        await self.session.say(filler_gen(), add_to_chat_ctx=False)

        # -----------------
        # Main response
        # -----------------
        async with self._workflow_llm.chat(chat_ctx=turn_ctx) as gen:
            async for chunk in gen:  # chunk is a string
                # Wrap string into proper async generator
                async def chunk_gen(text):
                    yield text

                await self.session.say(chunk_gen(chunk), add_to_chat_ctx=True)


# -------------------------
# Entrypoint
# -------------------------
async def entrypoint(ctx: JobContext):
    await ctx.connect()

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

    await session.start(
        agent=VoiceAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    logger.info(f"Agent started in room: {ctx.room.name}")


# -------------------------
# Run CLI
# -------------------------
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
