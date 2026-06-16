import asyncio
import websockets

async def test():
    try:
        async with websockets.connect("ws://localhost:5000/ws") as ws:
            print("WebSocket connected!")
            msg = await asyncio.wait_for(ws.recv(), timeout=5)
            print("Got data! Length:", len(msg))
            print("Working!")
    except Exception as e:
        print("Error:", e)

asyncio.run(test())