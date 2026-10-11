import asyncio
import json
import os
from dotenv import load_dotenv
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, PipelineParams
from pipecat.pipeline.runner import WorkerRunner
from pipecat.transports.daily.transport import DailyParams, DailyTransport
# Pipecat 1.12.0 Service & Transport Imports
from pipecat.services.edge.tts import EdgeTTSService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.services.groq.stt import GroqSTTService

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


def load_agent_context(filepath: str = "condensed_agent_context.json") -> str:
    """Loads condensed JSON and formats it for the LLM system prompt."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Context file '{filepath}' not found. Run content structring.py first!"
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return json.dumps(data, indent=2, ensure_ascii=False)


async def main():
    # 1. Load context generated from CV and Job Description
    context_str = load_agent_context()

    # 2. Setup Speech-to-Text via Groq Whisper
    stt = GroqSTTService(api_key=GROQ_API_KEY, model="whisper-large-v3-turbo")

    # 3. Formulate the Voice Interviewer Persona Prompt
    system_prompt = f"""
أنت محاور تقني متخصص تجري مقابلة عمل تجريبية باللغة العربية.

سياق المتقدم والوظيفة الشاغرة (JSON):
{context_str}

قواعد وإرشادات المقابلة الصوتية:
1. تحدث باللغة العربية الفصحى أو بلهجة عربية بيضاء واضحة وبسيطة.
2. اطرح سؤالاً واحداً فقط في كل مرة، وانتظر إجابة المتقدم.
3. ابدأ بالترحيب بالمتقدم بالاسم المذكور في الملف، واطلب منه تقديم نبذة مختصرة عن خلفيته المهنية.
4. ركّز على مناقشة المشاريع المذكورة، والتحقق من المهارات المطلوبة للوظيفة، والتركيز على نقاط الفجوة (Gap Analysis).
5. اجعل ردودك موجزة ومباشرة (جملتان إلى 3 جمل كحد أقصى) لتناسب المحادثة الصوتية التفاعلية.
"""

    # 4. Setup LLM Service
    llm = GroqLLMService(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-20b",
    )

    # 5. Setup Text-to-Speech (Edge-TTS Saudi Male Voice)
    tts = EdgeTTSService(voice="ar-SA-HamedNeural")

    # 6. Setup Daily WebRTC Transport
    daily_room_url = os.getenv("DAILY_ROOM_URL", "https://your-subdomain.daily.co/your-room")
    
    transport = DailyTransport(
        room_url=daily_room_url,
        token=None,
        bot_name="المحاور الذكي",
        params=DailyParams(audio_out_enabled=True, audio_in_enabled=True),
    )

    # 7. Create context aggregator
    context_aggregator = llm.create_context_aggregator(llm.create_initial_context())

    # 8. Construct Pipecat Frame Processing Pipeline
    pipeline = Pipeline(
        [
            transport.input(),              # Audio input from user
            stt,                            # STT processing
            context_aggregator.user(),      # Aggregates transcribed user speech
            llm,                            # LLM generates assistant turn
            tts,                            # Synthesizes speech from LLM text
            transport.output(),             # Streams audio back to user
            context_aggregator.assistant(), # Aggregates assistant response to context
        ]
    )

    task = PipelineWorker(
        pipeline,
        PipelineParams(
            allow_interruptions=True,
            system_prompt=system_prompt,
        ),
    )

    runner = WorkerRunner()
    print("--- Voice Agent Running. Connecting to Daily Room... ---")
    await runner.run(task)


if __name__ == "__main__":
    asyncio.run(main())