import chainlit as cl
import httpx
import json

FASTAPI_URL = "http://localhost:8000"

@cl.on_chat_start
async def on_chat_start():
    # Initialize conversation history per session
    cl.user_session.set("history", [])
    
    await cl.Message(
        content=(
            " **Welcome to GitLab Handbook HR Assistant!**\n\n"
            "I can answer questions about GitLab's HR policies, including:\n"
            "- Anti-harassment policy\n"
            "- Time off and leave types\n"
            "- Onboarding process\n"
            "- Promotions and transfers\n"
            "- Offboarding\n\n"
            "Ask me anything about GitLab's people policies!"
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()

    # Retrieve history from session
    history = cl.user_session.get("history", [])

    try:
        response = httpx.post(
            f"{FASTAPI_URL}/chat/stream",
            json={
                "question": message.content,
                "history": history
            },
            timeout=300.0
        )
        response.raise_for_status()

        sources = []
        num_chunks = 0
        full_answer = ""

        for line in response.text.split("\n"):
            line = line.strip()
            if line.startswith("data: "):
                data = json.loads(line[6:])

                if data["type"] == "token":
                    await msg.stream_token(data["content"])
                    full_answer += data["content"]

                elif data["type"] == "sources":
                    sources = data["sources"]
                    num_chunks = data["num_chunks"]

        # Update history
        history.append({"role": "user", "content": message.content})
        history.append({"role": "assistant", "content": full_answer})
        
        # save history
        if len(history) > 10:
            history = history[-10:]
        cl.user_session.set("history", history)

        # Append sources
        if sources:
            sources_clean = [s.replace("\\", "/").replace("documents/", "") for s in sources]
            sources_text = "\n".join([f"- `{s}`" for s in sources_clean])
            msg.content += f"\n\n---\n **Sources** ({num_chunks} chunks):\n{sources_text}"
            await msg.update()

    except httpx.TimeoutException:
        msg.content = " Request timed out. Please try again."
        await msg.update()
    except Exception as e:
        msg.content = f" Error: {str(e)}"
        await msg.update()