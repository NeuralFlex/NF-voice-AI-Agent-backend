from dotenv import load_dotenv

_ = load_dotenv(override=True)
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    WorkerOptions,
    cli,
    RoomInputOptions,
    inference,
)
import os
from livekit.plugins import elevenlabs, noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import os
from run_interview import create_workflow
from livekit.plugins import langchain


load_dotenv()


class Assistant(Agent):
    pass


async def entrypoint(ctx: JobContext):
    # Step 1: Connect to the room/job FIRST
    await ctx.connect()

    # Step 2: Create and start the session
    session = AgentSession(
        llm=langchain.LLMAdapter(graph=create_workflow()),
        # stt="deepgram/nova-2",  # Or keep AssemblyAI if you fixed the key
        stt="assemblyai/universal-streaming:en",
        tts=elevenlabs.TTS(
            model="eleven_flash_v2_5",
            voice_id="yj30vwTGJxSHezdAGsv9",
            api_key=os.getenv("ELEVENLABS_API_KEY"),
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=False,
    )
    await session.start(
        agent=Agent(instructions="You are a helpful assistant."),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
