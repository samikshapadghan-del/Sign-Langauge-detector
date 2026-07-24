"""Small end-to-end WebSocket smoke test for a running server."""

import asyncio
import json

import websockets


async def receive_state(socket):
    return json.loads(await asyncio.wait_for(socket.recv(), timeout=10))


async def wait_for(socket, predicate, attempts=30):
    for _ in range(attempts):
        state = await receive_state(socket)
        if predicate(state):
            return state
    raise AssertionError("Expected WebSocket state was not received.")


async def main():
    async with websockets.connect("ws://127.0.0.1:5000/ws", max_size=2**22) as socket:
        initial = await wait_for(
            socket,
            lambda item: item.get("frame") and item.get("camera_status") == "ready",
            attempts=80,
        )
        assert initial.get("frame"), "No camera frame received"
        assert initial.get("camera_status") == "ready"

        await socket.send(json.dumps({"action": "clear"}))
        await wait_for(socket, lambda state: state.get("sentence") == "")

        await socket.send(json.dumps({"action": "append_text", "value": "hel"}))
        state = await wait_for(socket, lambda item: item.get("sentence") == "HEL ")
        assert state["accepted"] == "HEL"

        await socket.send(json.dumps({"action": "set_threshold", "value": 0.81}))
        await socket.send(json.dumps({"action": "set_stability", "value": 18}))
        state = await wait_for(
            socket,
            lambda item: item.get("threshold") == 0.81 and item.get("stable_target") == 18,
        )
        assert state["frame"]

        await socket.send(json.dumps({"action": "pause"}))
        await wait_for(socket, lambda item: item.get("paused") is True)
        await socket.send(json.dumps({"action": "resume"}))
        await wait_for(socket, lambda item: item.get("paused") is False)

        await socket.send(json.dumps({"action": "clear"}))
        await wait_for(socket, lambda state: state.get("sentence") == "")
        print("HTTP/WebSocket camera and action smoke test: OK")


if __name__ == "__main__":
    asyncio.run(main())
