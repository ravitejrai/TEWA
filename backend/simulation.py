"""
Simulation Loop
================
Runs in a background thread, advancing the tactical picture every tick.
Periodically spawns new threats and applies AI engagement recommendations.
"""
from __future__ import annotations
import random
import time
import threading

from backend.state import ScenarioState
from backend.models import ThreatClass, ThreatStatus
from backend.engine.optimizer import optimise

# Simulation configuration
TICK_INTERVAL_S   = 3.0   # real-world seconds per simulation tick
AUTO_ENGAGE       = True   # automatically execute AI recommendations
THREAT_SPAWN_PROB = 0.08   # probability each tick that a new threat appears
MAX_THREATS       = 12     # cap for simulation playability


class SimulationLoop:
    def __init__(self, state: ScenarioState):
        self.state = state
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # pending resolution: list of (resolve_at_tick, order)
        self._pending_resolutions: list[tuple[int, object]] = []

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.state.running = True

    def stop(self):
        self._stop_event.set()
        self.state.running = False

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception as exc:
                self.state._log(f"SIM ERROR: {exc}", "ERROR")
            time.sleep(TICK_INTERVAL_S)

    def _tick(self):
        state = self.state

        # 1. Advance physics
        state.tick_state()

        # Cancel any pending engagements voided by ballistic MaRV evasion
        marv_cancelled = state.consume_marv_cancellations()
        if marv_cancelled:
            cancelled_set = set(marv_cancelled)
            self._pending_resolutions = [
                (t, o) for t, o in self._pending_resolutions
                if o.threat_id not in cancelled_set
            ]

        # 2. Kalman filter: predict → noisy measurement → update for all active threats
        state.tick_kalman()

        # 2. Clean up neutralized/escaped threats
        to_remove = [
            tid for tid, t in state.threats.items()
            if t.status in (ThreatStatus.NEUTRALIZED, ThreatStatus.ESCAPED)
        ]
        for tid in to_remove:
            state.remove_threat(tid)

        # 3. Resolve pending engagements
        current_tick = state.tick
        still_pending = []
        for resolve_tick, order in self._pending_resolutions:
            if current_tick >= resolve_tick:
                state.resolve_engagement(order)
            else:
                still_pending.append((resolve_tick, order))
        self._pending_resolutions = still_pending

        # 4. Maybe spawn a new threat
        active_count = sum(
            1 for t in state.threats.values()
            if t.status == ThreatStatus.ACTIVE
        )
        if active_count < MAX_THREATS and random.random() < THREAT_SPAWN_PROB:
            # Weight threat classes – drones and loitering munitions most common
            tc = random.choices(
                [ThreatClass.DRONE, ThreatClass.CRUISE, ThreatClass.FIGHTER,
                 ThreatClass.BALLISTIC, ThreatClass.LOITERING_MUNITION],
                weights=[35, 25, 15, 10, 15],
                k=1
            )[0]
            # Ballistic missiles are always P10 – highest possible threat
            if tc == ThreatClass.BALLISTIC:
                priority = 10
            elif tc == ThreatClass.FIGHTER:
                priority = random.randint(7, 9)
            else:
                priority = random.randint(3, 8)
            state.spawn_threat(threat_class=tc, priority=priority)

        # 5. Run optimizer and auto-engage
        if AUTO_ENGAGE:
            self._auto_engage()

    def _auto_engage(self):
        threats    = self.state.get_threats()
        effectors  = self.state.get_effectors()
        orders, warnings = optimise(threats, effectors)

        for order in orders:
            self.state.execute_order(order)
            # Schedule resolution 2-5 ticks later
            resolve_after = random.randint(2, 5)
            self._pending_resolutions.append(
                (self.state.tick + resolve_after, order)
            )

        # Append warnings to log
        for w in warnings:
            self.state._log(w["msg"], w["level"])
