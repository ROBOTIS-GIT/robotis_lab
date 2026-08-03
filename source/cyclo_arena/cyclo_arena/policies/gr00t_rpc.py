# Copyright 2026 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Seongwoo Kim

"""Minimal client for GR00T's native ZeroMQ policy protocol."""

from __future__ import annotations

import argparse
import io
from typing import Any, Sequence

import msgpack
import numpy as np
import zmq


class MsgSerializer:
    """Encode the subset of GR00T's MessagePack protocol used by Cyclo."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MsgSerializer._encode)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MsgSerializer._decode)

    @staticmethod
    def _encode(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            output = io.BytesIO()
            np.save(output, value, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        raise TypeError(f"Unsupported GR00T RPC value: {type(value)!r}")

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, dict) and "__ModalityConfig_class__" in value:
            return value["as_json"]
        if isinstance(value, dict) and "__ndarray_class__" in value:
            return np.load(io.BytesIO(value["as_npy"]), allow_pickle=False)
        return value


class Gr00tPolicyClient:
    """Call a GR00T policy server without importing Isaac Sim modules."""

    def __init__(self, host: str, port: int, timeout_ms: int):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        """Call one server endpoint and return its decoded response."""
        assert self.socket is not None, "GR00T client is closed"
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(MsgSerializer.to_bytes(request))
            response = MsgSerializer.from_bytes(self.socket.recv())
        except zmq.Again as error:
            self._connect()
            raise TimeoutError(
                f"GR00T server {self.host}:{self.port} timed out during {endpoint!r}"
            ) from error
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def ping(self) -> bool:
        """Return whether the remote server reports ready."""
        response = self.call("ping")
        return isinstance(response, dict) and response.get("status") == "ok"

    def get_action(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Request one action chunk for an observation."""
        response = self.call(
            "get_action", {"observation": observation, "options": None}
        )
        action, info = response
        return action, info

    def get_modality_config(self) -> dict[str, dict[str, Any]]:
        """Return the checkpoint's decoded observation and action schema."""
        response = self.call("get_modality_config")
        assert isinstance(response, dict), "Invalid GR00T modality configuration"
        assert all(isinstance(value, dict) for value in response.values()), (
            "GR00T modality configuration entries must be mappings"
        )
        return response

    def reset(self) -> None:
        """Reset the remote policy state."""
        self.call("reset", {"options": None})

    def kill_server(self) -> None:
        """Ask the remote server to exit."""
        self.call("kill")

    def close(self) -> None:
        """Close the client transport."""
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()


def main(argv: Sequence[str] | None = None) -> int:
    """Ping a GR00T server for launcher health checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout-ms", type=int, default=1000)
    args = parser.parse_args(argv)
    client = Gr00tPolicyClient(args.host, args.port, args.timeout_ms)
    try:
        return 0 if client.ping() else 1
    except (RuntimeError, TimeoutError):
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
