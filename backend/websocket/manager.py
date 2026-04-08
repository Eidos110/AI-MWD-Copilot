"""WebSocket manager for real-time streaming with binary protocol support."""

import logging
import json
import struct
from typing import List, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

try:
    import msgpack

    MSGPACK_AVAILABLE = True
except ImportError:
    MSGPACK_AVAILABLE = False
    msgpack = None

logger = logging.getLogger(__name__)
router = APIRouter()

MESSAGE_HEADER = {
    "PING": 0x01,
    "PONG": 0x02,
    "SUBSCRIBE": 0x10,
    "SUBSCRIBED": 0x11,
    "PREDICTION": 0x20,
    "ANOMALY": 0x21,
    "DATA": 0x30,
    "ERROR": 0xFF,
}


class ConnectionManager:
    """Manages WebSocket connections for real-time data streaming."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscriptions: dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.subscriptions[websocket] = {}
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.subscriptions:
            del self.subscriptions[websocket]
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    def _pack_binary(self, msg_type: int, data: dict) -> bytes:
        """Pack message to binary format using MessagePack."""
        if MSGPACK_AVAILABLE:
            packed = msgpack.packb({"type": msg_type, "data": data})
            return struct.pack("!I", len(packed)) + packed
        else:
            json_str = json.dumps({"type": msg_type, "data": data})
            return struct.pack("!I", len(json_str)) + json_str.encode("utf-8")

    def _unpack_binary(self, data: bytes) -> Optional[dict]:
        """Unpack binary message."""
        if MSGPACK_AVAILABLE and data[0] & 0x80:
            return msgpack.unpackb(data[4:], raw=False)
        try:
            return json.loads(data.decode("utf-8"))
        except:
            return None

    async def send_binary(self, msg_type: int, data: dict, websocket: WebSocket):
        """Send binary message."""
        try:
            binary_data = self._pack_binary(msg_type, data)
            await websocket.send_bytes(binary_data)
        except Exception as e:
            logger.error(f"Failed to send binary message: {e}")

    async def send_json(self, message: dict, websocket: WebSocket):
        """Send JSON message (fallback)."""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Failed to send JSON message: {e}")

    async def broadcast(self, message: dict):
        """Broadcast JSON message to all connections."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

    async def broadcast_binary(self, msg_type: int, data: dict):
        """Broadcast binary message to all connections."""
        binary_data = self._pack_binary(msg_type, data)
        for connection in self.active_connections:
            try:
                await connection.send_bytes(binary_data)
            except Exception:
                pass

    async def broadcast_prediction(self, depth: float, predictions: dict):
        """Broadcast a new prediction to all connected clients."""
        if MSGPACK_AVAILABLE:
            await self.broadcast_binary(
                MESSAGE_HEADER["PREDICTION"],
                {"depth": depth, "predictions": predictions},
            )
        else:
            await self.broadcast(
                {"type": "prediction", "depth": depth, "data": predictions}
            )

    async def broadcast_anomaly(self, alert: dict):
        """Broadcast an anomaly alert to all connected clients."""
        if MSGPACK_AVAILABLE:
            await self.broadcast_binary(MESSAGE_HEADER["ANOMALY"], alert)
        else:
            await self.broadcast({"type": "anomaly_alert", **alert})

    def get_subscription(self, websocket: WebSocket) -> dict:
        """Get subscription settings for a connection."""
        return self.subscriptions.get(websocket, {})


manager = ConnectionManager()


@router.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint with binary protocol support."""
    await manager.connect(websocket)
    use_binary = MSGPACK_AVAILABLE

    try:
        while True:
            try:
                data = await websocket.receive()

                if "text" in data:
                    message = json.loads(data["text"])
                elif "bytes" in data:
                    use_binary = True
                    message = manager._unpack_binary(data["bytes"])
                    if message is None:
                        continue
                else:
                    continue

                msg_type = message.get("type")

                if msg_type == "ping":
                    if use_binary:
                        await manager.send_binary(MESSAGE_HEADER["PONG"], {}, websocket)
                    else:
                        await manager.send_json({"type": "pong"}, websocket)

                elif msg_type == "subscribe":
                    depth_range = message.get("depth_range", {})
                    manager.subscriptions[websocket] = depth_range
                    if use_binary:
                        await manager.send_binary(
                            MESSAGE_HEADER["SUBSCRIBED"],
                            {"depth_range": depth_range},
                            websocket,
                        )
                    else:
                        await manager.send_json(
                            {"type": "subscribed", "depth_range": depth_range},
                            websocket,
                        )

                elif msg_type == "binary":
                    use_binary = True

            except json.JSONDecodeError:
                logger.warning("Invalid JSON received")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

    except WebSocketDisconnect:
        manager.disconnect(websocket)
