"""
Air Defense C2 – FastAPI Application
======================================
REST endpoints + WebSocket for real-time state broadcast.
"""
from __future__ import annotations
import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.state import ScenarioState
from backend.simulation import SimulationLoop
from backend.models import ThreatClass, EffectorType
from backend.engine.optimizer import optimise

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------
state = ScenarioState()
sim   = SimulationLoop(state)


@asynccontextmanager
async def lifespan(app: FastAPI):
    sim.start()
    # pre-seed with a couple of threats so the map isn't empty on load
    for tc, pri in [(ThreatClass.DRONE, 6), (ThreatClass.CRUISE, 8), (ThreatClass.FIGHTER, 9)]:
        state.spawn_threat(threat_class=tc, priority=pri)
    yield
    sim.stop()


app = FastAPI(title="Air Defense C2", version="1.0", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")


@app.get("/")
async def root():
    return FileResponse(
        "frontend/index.html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---------------------------------------------------------------------------
# WebSocket – real-time state broadcast
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        self.connections.discard(ws) if hasattr(self.connections, 'discard') else None
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, data: dict):
        payload = json.dumps(data)
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            snap = state.snapshot()
            await websocket.send_text(json.dumps(snap))
            await asyncio.sleep(1.0)   # push update every second
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ---------------------------------------------------------------------------
# REST – Scenario control
# ---------------------------------------------------------------------------

@app.get("/api/state")
async def get_state():
    return state.snapshot()


class SpawnThreatRequest(BaseModel):
    threat_class: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    priority: Optional[int] = None
    target_idx: Optional[int] = None


@app.post("/api/threats/spawn")
async def spawn_threat(req: SpawnThreatRequest):
    tc = ThreatClass(req.threat_class) if req.threat_class else None
    threat = state.spawn_threat(
        threat_class=tc,
        lat=req.lat,
        lon=req.lon,
        priority=req.priority,
        target_idx=req.target_idx,
    )
    return threat.to_dict()


@app.delete("/api/threats/{threat_id}")
async def remove_threat(threat_id: str):
    state.remove_threat(threat_id)
    return {"status": "removed"}


# ---------------------------------------------------------------------------
# REST – Engagement control
# ---------------------------------------------------------------------------

@app.post("/api/engage/optimise")
async def run_optimiser():
    """Trigger one optimisation cycle and return recommended orders (no auto-execute)."""
    threats   = state.get_threats()
    effectors = state.get_effectors()
    orders, warnings = optimise(threats, effectors)
    return {
        "orders": [o.to_dict() for o in orders],
        "warnings": warnings,
        "timestamp": time.time(),
    }


class ManualEngageRequest(BaseModel):
    effector_id: str
    threat_id: str


class AutoEngageRequest(BaseModel):
    threat_id: str


@app.post("/api/engage/auto")
async def auto_engage(req: AutoEngageRequest):
    """
    Auto-select the best effector for a threat using the optimizer's scoring,
    with cost-efficiency weighting based on threat priority.
    """
    from backend.models import EngagementOrder, haversine
    from backend.engine.optimizer import composite_score, build_rationale

    with state._lock:
        thr = state.threats.get(req.threat_id)
        all_effectors = list(state.effectors.values())

    if not thr:
        raise HTTPException(404, f"Threat {req.threat_id} not found")

    # Score all ready effectors against this threat
    candidates = []
    for eff in all_effectors:
        if not eff.can_engage():
            continue
        score = composite_score(eff, thr)
        if score > 0:
            candidates.append((score, eff))

    if not candidates:
        raise HTTPException(409, "No effector in range or all depleted – cannot engage.")

    # Pick the highest-scoring candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    score, eff = candidates[0]

    dist = haversine(eff.lat, eff.lon, thr.lat, thr.lon)
    pk   = eff.effective_pk(thr)
    order = EngagementOrder(
        effector_id=eff.id,
        threat_id=thr.id,
        effector_type=eff.effector_type.value,
        effector_label=eff.label,
        threat_label=thr.label,
        threat_class=thr.threat_class.value,
        pk=pk,
        distance_km=dist,
        time_to_intercept_s=dist / max(thr.speed_kmh, 1) * 3600 * 0.5,
        rationale=f"AUTO-SELECT | {build_rationale(eff, thr, score)}",
        timestamp=time.time(),
        outcome="pending",
    )
    state.execute_order(order)
    sim._pending_resolutions.append((state.tick + 3, order))

    # ── Double-tap doctrine ──────────────────────────────────────────────────
    # For P9-10 threats (e.g. ballistic missiles) where PK < 0.80, automatically
    # queue a second independent effector to guarantee a kill-chain backup.
    double_tap_label = None
    if thr.priority >= 9 and pk < 0.80 and len(candidates) > 1:
        second_score, second_eff = candidates[1]
        if second_score > 0.15:
            second_dist = haversine(second_eff.lat, second_eff.lon, thr.lat, thr.lon)
            second_pk   = second_eff.effective_pk(thr)
            second_order = EngagementOrder(
                effector_id=second_eff.id,
                threat_id=thr.id,
                effector_type=second_eff.effector_type.value,
                effector_label=second_eff.label,
                threat_label=thr.label,
                threat_class=thr.threat_class.value,
                pk=second_pk,
                distance_km=second_dist,
                time_to_intercept_s=second_dist / max(thr.speed_kmh, 1) * 3600 * 0.5,
                rationale=f"DOUBLE-TAP | {build_rationale(second_eff, thr, second_score)}",
                timestamp=time.time(),
                outcome="pending",
            )
            state.execute_order(second_order)
            sim._pending_resolutions.append((state.tick + 4, second_order))
            double_tap_label = second_eff.label

    return {
        **order.to_dict(),
        "selected_effector": eff.label,
        "base_id": eff.base_id,
        "score": round(score, 3),
        "double_tap": double_tap_label,
    }


@app.post("/api/engage/manual")
async def manual_engage(req: ManualEngageRequest):
    """Operator-directed engagement override."""
    from backend.models import EngagementOrder, haversine
    import random

    with state._lock:
        eff = state.effectors.get(req.effector_id)
        thr = state.threats.get(req.threat_id)

    if not eff:
        raise HTTPException(404, f"Effector {req.effector_id} not found")
    if not thr:
        raise HTTPException(404, f"Threat {req.threat_id} not found")
    if not eff.can_engage():
        raise HTTPException(409, f"Effector {eff.label} is not ready ({eff.status.value})")

    dist = haversine(eff.lat, eff.lon, thr.lat, thr.lon)
    pk   = eff.pk(thr)
    order = EngagementOrder(
        effector_id=eff.id,
        threat_id=thr.id,
        effector_type=eff.effector_type.value,
        effector_label=eff.label,
        threat_label=thr.label,
        threat_class=thr.threat_class.value,
        pk=pk,
        distance_km=dist,
        time_to_intercept_s=dist / max(thr.speed_kmh, 1) * 3600 * 0.5,
        rationale=f"MANUAL OVERRIDE | PK={pk:.0%} | dist={dist:.0f}km",
        timestamp=time.time(),
        outcome="pending",
    )
    state.execute_order(order)
    # schedule resolution
    sim._pending_resolutions.append((state.tick + 3, order))
    return order.to_dict()


# ---------------------------------------------------------------------------
# REST – Asset management
# ---------------------------------------------------------------------------

@app.get("/api/bases")
async def list_bases():
    return [b.to_dict() for b in state.bases.values()]


@app.get("/api/effectors")
async def list_effectors():
    return [e.to_dict() for e in state.effectors.values()]


@app.post("/api/effectors/{effector_id}/reload")
async def force_reload(effector_id: str):
    """Force-reload an effector (operator action)."""
    with state._lock:
        eff = state.effectors.get(effector_id)
        if not eff:
            raise HTTPException(404, "Effector not found")
        eff.ammo = eff.max_ammo
        eff.status = __import__('backend.models', fromlist=['AssetStatus']).AssetStatus.READY
        eff.reload_ticks_remaining = 0
    return eff.to_dict()


# ---------------------------------------------------------------------------
# REST – Simulation control
# ---------------------------------------------------------------------------

@app.post("/api/sim/start")
async def sim_start():
    sim.start()
    return {"running": True}


@app.post("/api/sim/stop")
async def sim_stop():
    sim.stop()
    return {"running": False}


@app.post("/api/sim/reset")
async def sim_reset():
    global state, sim
    sim.stop()
    # Re-initialise state in-place
    state = ScenarioState()
    sim   = SimulationLoop(state)
    sim.start()
    for tc, pri in [(ThreatClass.DRONE, 6), (ThreatClass.CRUISE, 8)]:
        state.spawn_threat(threat_class=tc, priority=pri)
    return {"status": "reset"}


@app.get("/api/sim/status")
async def sim_status():
    return {
        "running": state.running,
        "tick": state.tick,
        "threat_count": len(state.threats),
        "effector_count": len(state.effectors),
    }
