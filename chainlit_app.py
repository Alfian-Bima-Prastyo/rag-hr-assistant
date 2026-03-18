# chainlit_app.py
import chainlit as cl
import httpx

FASTAPI_URL = "http://localhost:8000"

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "👋 **Welcome to GitLab Handbook HR Assistant!**\n\n"
            "I can answer questions about GitLab's HR policies, including:\n"
            "- 📋 Anti-harassment policy\n"
            "- 🏖️ Time off and leave types\n"
            "- 🚀 Onboarding process\n"
            "- 📈 Promotions and transfers\n"
            "- 🚪 Offboarding\n\n"
            "Ask me anything about GitLab's people policies!"
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    # Tampilkan loading indicator
    msg = cl.Message(content="")
    await msg.send()

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{FASTAPI_URL}/chat",
                json={"question": message.content}
            )
            response.raise_for_status()
            data = response.json()

        answer = data["answer"]
        sources = data["sources"]
        num_chunks = data["num_chunks"]

        # Format sources
        if sources:
            sources_clean = [s.replace("\\", "/").replace("documents/", "") for s in sources]
            sources_text = "\n".join([f"- `{s}`" for s in sources_clean])
            full_response = f"{answer}\n\n---\n📚 **Sources** ({num_chunks} chunks):\n{sources_text}"
        else:
            full_response = answer

        msg.content = full_response
        await msg.update()

    except httpx.TimeoutException:
        msg.content = "⏱️ Request timed out. The LLM is taking too long, please try again."
        await msg.update()
    except Exception as e:
        msg.content = f"❌ Error: {str(e)}"
        await msg.update()