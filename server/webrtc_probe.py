#
# Headless WebRTC probe — reproduces a browser connecting to the local proctor.
# Uploads a tiny exam, opens a SmallWebRTC connection (silent mic), and prints
# SSE events + ICE state so we can see whether the bot starts or errors.
#
import asyncio
import json
import sys

import os

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack

BASE = os.getenv("PROBE_BASE", "http://localhost:7860")
PDF = (
    b"%PDF-1.4\n"  # placeholder; we use a real generated one below
)


async def make_pdf() -> bytes:
    import fitz

    doc = fitz.open()
    p = doc.new_page()
    p.insert_text((72, 90), "Biology: photosynthesis happens in chloroplasts. "
                  "Mitosis makes two identical cells. Respiration makes ATP in mitochondria.")
    data = doc.tobytes()
    doc.close()
    return data


async def wait_ice(pc):
    if pc.iceGatheringState == "complete":
        return
    fut = asyncio.get_event_loop().create_future()

    @pc.on("icegatheringstatechange")
    def _():
        if pc.iceGatheringState == "complete" and not fut.done():
            fut.set_result(None)

    try:
        await asyncio.wait_for(fut, timeout=4)
    except asyncio.TimeoutError:
        pass


async def main():
    async with aiohttp.ClientSession() as http:
        pdf = await make_pdf()
        form = aiohttp.FormData()
        form.add_field("file", pdf, filename="e.pdf", content_type="application/pdf")
        async with http.post(f"{BASE}/upload-exam", data=form) as r:
            up = await r.json()
        sid = up["session_id"]
        print(f"[probe] session={sid} title={up['title']} nq={up['num_questions']}")

        # SSE listener
        async def sse():
            async with http.get(f"{BASE}/events/{sid}") as r:
                async for raw in r.content:
                    line = raw.decode(errors="ignore").strip()
                    if line.startswith("data:"):
                        ev = json.loads(line[5:])
                        t = ev.get("type")
                        if t in ("interim",):
                            continue
                        extra = ev.get("question") or ev.get("text") or ev.get("combined_score") or ""
                        print(f"[sse] {t}: {str(extra)[:80]}")

        sse_task = asyncio.create_task(sse())

        pc = RTCPeerConnection()
        pc.addTrack(AudioStreamTrack())  # silent mic

        @pc.on("connectionstatechange")
        async def _cs():
            print(f"[pc] connectionState={pc.connectionState}")

        @pc.on("iceconnectionstatechange")
        async def _ice():
            print(f"[pc] iceConnectionState={pc.iceConnectionState}")

        @pc.on("track")
        def _track(track):
            print(f"[pc] received track: {track.kind}")

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        await wait_ice(pc)
        async with http.post(
            f"{BASE}/api/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type,
                  "request_data": {"session_id": sid}},
        ) as r:
            print(f"[probe] /api/offer -> {r.status}")
            ans = await r.json()
        await pc.setRemoteDescription(RTCSessionDescription(sdp=ans["sdp"], type=ans["type"]))
        print("[probe] remote description set; waiting 35s for bot to greet + ask Q1...")

        await asyncio.sleep(35)
        await pc.close()
        sse_task.cancel()
        print("[probe] done")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
