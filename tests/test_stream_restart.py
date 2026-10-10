import asyncio
from typing import Any

from f8pysdk.specs import F8RuntimeNode
from f8pymppose.service_node import MediaPipePoseServiceNode
from f8pymppose.video_input import FrameContext, LatestVideoInput


class RestartedVideoInput(LatestVideoInput):
    def __init__(self) -> None:
        super().__init__()
        self.frames = iter([FrameContext(frame_id=i, ts_ms=i, width=1, height=1, pitch=4,
                                         payload=b"1234", stream_epoch="b" * 32) for i in (1, 2, 4)])

    def is_open_for(self, *, stream_key: str) -> bool:
        return True

    def read_frame(self) -> FrameContext | None:
        frame = next(self.frames, None)
        if frame is None:
            raise asyncio.CancelledError
        return frame


class RecordingPoseNode(MediaPipePoseServiceNode):
    def __init__(self) -> None:
        super().__init__(node_id="pose", node=F8RuntimeNode(nodeId="pose", serviceId="pose", serviceClass="f8.mppose"),
                         initial_state={})
        self._video_input = RestartedVideoInput()
        self._stream_identity = ("video", "a" * 32)
        self._last_processed_frame_id = 1000
        self._last_infer_frame_id = 1000
        self._infer_every_n = 3
        self.resets = 0
        self.processed: list[int] = []

    async def _ensure_config_loaded(self) -> None:
        return

    async def _ensure_pose_runtime(self) -> None:
        return

    async def _reset_pose_runtime(self) -> None:
        self.resets += 1

    def _resolve_video_stream_key(self) -> str:
        return "video"

    async def _process_frame(self, frame: FrameContext, *, np_module: Any) -> None:
        self.processed.append(frame.frame_id)
        self._last_infer_frame_id = frame.frame_id


def test_restarted_stream_resets_runtime_and_inference_sampling() -> None:
    node = RecordingPoseNode()
    try:
        asyncio.run(node._loop())
    except asyncio.CancelledError:
        pass
    assert node.resets == 1
    assert node.processed == [1, 4]
