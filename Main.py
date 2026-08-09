import asyncio
from Workflow import create_workflow
from EmbeddingsManage import index_pdf

CONFIG = {
    "configurable": {
        "thread_id": 1
    }
}

async def main():
    #index_pdf()

    while True:
        question = input("Prompt: ")
        agent = create_workflow()

        async for update in agent.astream(
                {"question": question},
                config=CONFIG,
                stream_mode="updates",
        ):
            for node_name in ("simple_answer", "create_answer"):
                if node_name not in update:
                    continue

                messages = update[node_name].get("messages", [])

                if messages:
                    print(messages[-1].content)



if __name__ == "__main__":
    asyncio.run( main())