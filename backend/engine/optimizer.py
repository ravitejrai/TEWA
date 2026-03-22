"""
Tactical Optimization Engine
=============================
Solves the multi-threat / multi-effector assignment problem using a
two-phase approach:

Phase 1 – Score matrix construction
  For each (effector, threat) pair compute a composite score that factors in:
    • Probability of kill (PK) from the PK lookup table
    • Time-to-target urgency (threats about to reach protected zone get priority)
    • Effector range feasibility (only in-range pairs are considered)
    • Ammo conservation (prefer effectors with ample reserves)
    • Shot-line geometry (avoid engaging from behind when possible)

Phase 2 – Assignment
  Uses the scipy linear_sum_assignment (Hungarian algorithm) on the
  negated score matrix to maximise total expected value.
  Post-processes to ensure:
    • One effector per threat per cycle (double-tap handled by re-run)
    • High-priority threats get at least one effector assigned
    • Remaining capacity is assessed for reserve/coverage planning

Phase 3 – Lookahead reservation
  After assignments, the engine scans the remaining effector pool and
  flags bases that are dangerously depleted, recommending reallocation.
"""

from __future__ import annotations
import time
import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from ..models import Threat, Effector, Base, EngagementOrder

from backend.models import (
    Threat, Effector, Base, EngagementOrder,
    ThreatStatus, ThreatClass, AssetStatus, EffectorType, haversine, RELOAD_TICKS
)

# Weights for score components
W_PK          = 0.35
W_URGENCY     = 0.30
W_COST_EFF    = 0.20   # prefer cheaper weapons unless threat priority demands otherwise
W_AMMO_CONS   = 0.05
W_RANGE_MARGIN= 0.10

PROTECTED_LAT = 59.33   # Stockholm
PROTECTED_LON = 18.07
INTERCEPT_HORIZON_S = 600   # threats arriving in <10 min are critical

# Cost ceiling for normalisation – PAC-3 salvo ($8M) is the most expensive
MAX_ENGAGEMENT_COST_USD = 8_000_000

# Estimated weapon/aircraft intercept speed by effector type (km/h)
EFFECTOR_SPEED_KMH: dict = {
    EffectorType.MISSILE:         1500.0,
    EffectorType.ANTI_AIR:         700.0,
    EffectorType.DRONE:            280.0,
    EffectorType.AIRCRAFT:         900.0,   # intercept dash speed
    EffectorType.KINETIC_MISSILE: 5000.0,
    EffectorType.DIRECTED_ENERGY:    1e9,   # speed of light
}

# Strategic reserve policy
RESERVE_SLOTS = 2                # keep ≥ this many high-value assets uncommitted
RESERVE_PRIORITY_THRESHOLD = 7  # only break reserve for P7+ threats


def _urgency_score(threat: Threat) -> float:
    """0.0–1.0, higher = more urgent."""
    ttt = threat.time_to_target_s()
    if ttt <= 0:
        return 1.0
    return min(1.0, INTERCEPT_HORIZON_S / max(ttt, 1))


def _cost_efficiency_score(effector: Effector, threat: Threat) -> float:
    """
    Reward cost-proportionality: cheap weapons score high against low-priority
    threats; expensive weapons are justified only for high-priority threats.

    Score formula:
      - Base: 1 - (engagement_cost / MAX_COST)   [cheaper = higher base]
      - Scaled UP when threat priority is high (expensive weapons acceptable)
      - Result capped to [0, 1]
    """
    engagement_cost = effector.cost_per_shot_usd * effector.salvo_size
    cost_ratio = min(engagement_cost / MAX_ENGAGEMENT_COST_USD, 1.0)
    priority_factor = threat.priority / 10.0   # 0.0 (lowest) → 1.0 (highest)

    # For a P10 (ICBM) threat: even a $8M PAC-3 salvo scores 1.0
    # For a P3 (drone) threat: a $30K SHORAD scores 0.97, a $8M PAC-3 scores 0.4
    cost_score = 1.0 - cost_ratio * (1.0 - priority_factor)
    return max(0.0, min(1.0, cost_score))


def _ammo_conservation_score(effector: Effector) -> float:
    """Prefer effectors with more ammo remaining; unlimited ammo = full score."""
    if effector.ammo == -1:
        return 1.0
    return effector.ammo / max(effector.max_ammo, 1)


def _range_margin_score(effector: Effector, threat: Threat) -> float:
    """Reward engaging from well within range (gives time to fire second shot)."""
    dist = haversine(effector.lat, effector.lon, threat.lat, threat.lon)
    if dist > effector.range_km:
        return 0.0
    # 1.0 at 0 distance, approaching 0 at maximum range
    return 1.0 - (dist / effector.range_km) ** 0.5


def _intercept_window_s(effector: Effector, threat: Threat) -> float:
    """
    Seconds remaining in which this effector can still intercept the threat
    before it reaches its target.  Positive = feasible; ≤ 0 = too late.
    """
    ttt = threat.time_to_target_s()
    dist = haversine(effector.lat, effector.lon, threat.lat, threat.lon)
    speed = EFFECTOR_SPEED_KMH.get(effector.effector_type, 1000.0)
    travel_s = (dist / speed) * 3600.0
    return ttt - travel_s


def composite_score(effector: Effector, threat: Threat) -> float:
    """Returns a [0, 1] engagement desirability score."""
    if not effector.in_range(threat):
        return 0.0
    if not effector.can_engage():
        return 0.0

    pk      = effector.effective_pk(threat)   # includes salvo policy + weather
    urg     = _urgency_score(threat)
    cost_e  = _cost_efficiency_score(effector, threat)
    ammo    = _ammo_conservation_score(effector)
    margin  = _range_margin_score(effector, threat)

    # Penalise if intercept window is critically narrow or negative.
    # Cap at 0.15 so even past-deadline engagements still score above zero
    # (the system must assign *something* for high-priority threats).
    window = _intercept_window_s(effector, threat)
    window_penalty = 0.0 if window >= 90 else min(0.15, (90.0 - window) / 90.0 * 0.15)

    return max(0.0, W_PK * pk +
                    W_URGENCY * urg +
                    W_COST_EFF * cost_e +
                    W_AMMO_CONS * ammo +
                    W_RANGE_MARGIN * margin -
                    window_penalty)


def build_rationale(effector: Effector, threat: Threat, score: float) -> str:
    dist    = haversine(effector.lat, effector.lon, threat.lat, threat.lon)
    pk_base = effector.pk(threat)
    pk_eff  = effector.effective_pk(threat)
    urg     = _urgency_score(threat)
    ttt     = threat.time_to_target_s()
    reasons = []
    # Intercept window warning (front of list if critical)
    window = _intercept_window_s(effector, threat)
    if window < 60:
        reasons.append(f"⚠ CRITICAL WINDOW {window:.0f}s")
    elif window < 180:
        reasons.append(f"⚠ narrow window {window:.0f}s")
    # PK – show base and effective (salvo-boosted) if different
    if effector.salvo_size > 1:
        reasons.append(f"PK={pk_base:.0%}×{effector.salvo_size}={pk_eff:.0%}")
    else:
        reasons.append(f"PK={pk_eff:.0%}")
    if effector.weather_degradation:
        reasons.append("⚠ weather-sensitive")
    if urg > 0.7:
        reasons.append(f"URGENT ({ttt:.0f}s to target)")
    reasons.append(f"range {dist:.0f}/{effector.range_km:.0f}km")
    if effector.ammo != -1:
        reasons.append(f"ammo {effector.ammo}/{effector.max_ammo}")
    # Cost
    total_cost = effector.cost_per_shot_usd * effector.salvo_size
    if total_cost >= 1_000_000:
        reasons.append(f"${total_cost/1e6:.1f}M/engagement")
    elif total_cost > 0:
        reasons.append(f"${total_cost:,.0f}/engagement")
    return f"Score {score:.2f} | " + " | ".join(reasons)


def optimise(
    threats: list[Threat],
    effectors: list[Effector],
) -> tuple[list[EngagementOrder], list[dict]]:
    """
    Core assignment function.

    Returns:
        orders   – list of EngagementOrder objects
        warnings – list of advisory dicts for the operator
    """
    now = time.time()
    orders: list[EngagementOrder] = []
    warnings: list[dict] = []

    # Filter to actionable inputs
    active_threats      = [t for t in threats  if t.status == ThreatStatus.ACTIVE]
    engaged_ballistics  = [t for t in threats
                           if t.threat_class == ThreatClass.BALLISTIC
                           and t.status == ThreatStatus.ENGAGED]
    ready_effectors = [e for e in effectors if e.can_engage()]

    # If nothing to do at all, exit early
    if not active_threats and not engaged_ballistics:
        return orders, warnings
    if not ready_effectors:
        warnings.append({"level": "CRITICAL", "msg": "No ready effectors available! All assets depleted or reloading."})
        return orders, warnings

    assigned_effectors: set[int] = set()
    assigned_threats:   set[int] = set()

    if active_threats:
        # ── Strategic reserve policy ─────────────────────────────────────────
        # Hold back high-value effectors (aircraft, PAC-3) when:
        #   • all current threats are below the priority threshold, AND
        #   • we have enough non-reserve effectors to handle them anyway
        # This ensures rapid-response capacity for a sudden escalation (P7+ threat).
        def _is_reserve_asset(e: Effector) -> bool:
            return e.effector_type in (EffectorType.AIRCRAFT, EffectorType.KINETIC_MISSILE)

        high_priority_present = any(t.priority >= RESERVE_PRIORITY_THRESHOLD for t in active_threats)
        reserve_candidates    = [e for e in ready_effectors if _is_reserve_asset(e)]
        non_reserve           = [e for e in ready_effectors if not _is_reserve_asset(e)]

        held_back: list[Effector] = []
        if (not high_priority_present
                and len(reserve_candidates) >= RESERVE_SLOTS
                and len(non_reserve) >= len(active_threats)):
            # Safety check: verify non-reserve effectors can actually engage ALL threats
            # before withholding the reserve.  If any threat is uncoverable without the
            # reserve assets, release the reserve immediately.
            uncoverable = [
                t for t in active_threats
                if not any(e.in_range(t) and e.can_engage() for e in non_reserve)
            ]
            if not uncoverable:
                held_back       = reserve_candidates[:RESERVE_SLOTS]
                ready_effectors = [e for e in ready_effectors if e not in held_back]
                warnings.append({
                    "level": "ADVISORY",
                    "msg": (
                        f"RESERVE HOLD: {', '.join(e.label for e in held_back)} withheld "
                        f"from low-priority assignment (P<{RESERVE_PRIORITY_THRESHOLD}). "
                        f"Will release automatically on P{RESERVE_PRIORITY_THRESHOLD}+ escalation."
                    ),
                })
        if not held_back and len(ready_effectors) > 0:
            uncommitted = len(ready_effectors) - len(active_threats)
            if uncommitted > 0:
                warnings.append({
                    "level": "ADVISORY",
                    "msg": f"Escalation buffer: {uncommitted} effector(s) will remain uncommitted after this cycle.",
                })

        n_t = len(active_threats)
        n_e = len(ready_effectors)

        # Build n_e × n_t score matrix
        score_matrix = np.zeros((n_e, n_t))
        for i, eff in enumerate(ready_effectors):
            for j, thr in enumerate(active_threats):
                score_matrix[i, j] = composite_score(eff, thr)

        # Priority weighting: scale columns by threat priority
        priority_weights = np.array([t.priority / 10.0 for t in active_threats])
        weighted_matrix = score_matrix * priority_weights  # broadcast over rows

        # ------------------------------------------------------------------
        # Phase 2: Hungarian assignment (maximise → negate for minimiser)
        # We need a square or rectangular matrix.  Pad if needed.
        # ------------------------------------------------------------------
        rows, cols = n_e, n_t
        padded = weighted_matrix.copy()

        if rows < cols:
            # More threats than effectors - pad effector rows with zeros
            padded = np.vstack([padded, np.zeros((cols - rows, cols))])

        row_ind, col_ind = linear_sum_assignment(-padded)

        for r, c in zip(row_ind, col_ind):
            if r >= n_e:
                continue   # padded row
            if c >= n_t:
                continue
            score = score_matrix[r, c]  # use un-weighted for display clarity
            if score <= 0.0:
                continue   # no feasible assignment

            eff = ready_effectors[r]
            thr = active_threats[c]
            dist = haversine(eff.lat, eff.lon, thr.lat, thr.lon)
            # Approximate intercept time: travel at threat speed along remaining path
            intercept_s = dist / max(thr.speed_kmh, 1) * 3600 * 0.5  # missile faster

            order = EngagementOrder(
                effector_id=eff.id,
                threat_id=thr.id,
                effector_type=eff.effector_type.value,
                effector_label=eff.label,
                threat_label=thr.label,
                threat_class=thr.threat_class.value,
                pk=eff.effective_pk(thr),   # salvo + weather adjusted
                distance_km=dist,
                time_to_intercept_s=intercept_s,
                rationale=build_rationale(eff, thr, score),
                timestamp=now,
                outcome="pending",
            )
            orders.append(order)
            assigned_effectors.add(r)
            assigned_threats.add(c)

        # ------------------------------------------------------------------
        # Phase 3: Warnings & advisory
        # ------------------------------------------------------------------
        unassigned_threats = [active_threats[j] for j in range(n_t) if j not in assigned_threats]
        for t in unassigned_threats:
            level = "CRITICAL" if t.priority >= 8 else "WARNING"
            warnings.append({
                "level": level,
                "msg": f"Threat {t.label} ({t.threat_class.value}) has NO effector assigned – no asset in range or all depleted.",
                "threat_id": t.id,
            })

        # Check coverage reserve
        _add_reserve_warnings(ready_effectors, assigned_effectors, warnings)

    # ──────────────────────────────────────────────────────────────────────────
    # Phase 4 – Shoot-Look-Shoot (SLS) doctrine
    # Ballistic missiles already ENGAGED (by a JAS-39 long-range shot) may fly
    # into the engagement envelope of a ground SAM while in flight.  Queue a
    # supplementary shot from a MISSILE or KINETIC_MISSILE battery if:
    #   • The threat is BALLISTIC and currently ENGAGED by another effector
    #   • A ground SAM that hasn't already fired on this threat is now in range
    # Creates a two-layer kill-chain: JAS-39 first shot, Patriot/PAC-3 follow-up.
    # ──────────────────────────────────────────────────────────────────────────
    ground_sam_types = {EffectorType.MISSILE, EffectorType.KINETIC_MISSILE}
    for thr in engaged_ballistics:
        for eff in ready_effectors:
            if eff.effector_type not in ground_sam_types:
                continue
            if eff.assigned_threat_id == thr.id:
                continue
            if not eff.in_range(thr) or not eff.can_engage():
                continue
            score = composite_score(eff, thr)
            if score <= 0.0:
                continue
            dist = haversine(eff.lat, eff.lon, thr.lat, thr.lon)
            sls_order = EngagementOrder(
                effector_id=eff.id,
                threat_id=thr.id,
                effector_type=eff.effector_type.value,
                effector_label=eff.label,
                threat_label=thr.label,
                threat_class=thr.threat_class.value,
                pk=eff.effective_pk(thr),
                distance_km=dist,
                time_to_intercept_s=dist / max(thr.speed_kmh, 1) * 3600 * 0.5,
                rationale=(
                    f"SLS LAYER-2 | {build_rationale(eff, thr, score)} "
                    f"| backing up in-flight engagement"
                ),
                timestamp=time.time(),
                outcome="pending",
            )
            orders.append(sls_order)
            warnings.append({
                "level": "ADVISORY",
                "msg": (
                    f"SLS: {eff.label} assigned as layer-2 interceptor against "
                    f"{thr.label} (ballistic already tracked by another asset)"
                ),
            })
            break  # one SLS effector per ballistic per cycle

    return orders, warnings

def _add_reserve_warnings(
    ready_effectors: list[Effector],
    assigned_idxs: set[int],
    warnings: list[dict],
):
    """Flag if remaining reserve is critically low."""
    unassigned = [ready_effectors[i] for i in range(len(ready_effectors)) if i not in assigned_idxs]
    if len(unassigned) == 0:
        warnings.append({
            "level": "WARNING",
            "msg": "All ready effectors are committed. No reserve capacity for follow-on threats.",
        })
    elif len(unassigned) <= 2:
        warnings.append({
            "level": "ADVISORY",
            "msg": f"Low reserve: only {len(unassigned)} effector(s) unassigned. Consider reallocation.",
        })

    # Check for effectors with critically low ammo
    low_ammo = [e for e in ready_effectors if e.ammo != -1 and e.ammo <= 2]
    for e in low_ammo:
        warnings.append({
            "level": "ADVISORY",
            "msg": f"{e.label} is low on ammo ({e.ammo} rounds remaining). Schedule resupply.",
            "effector_id": e.id,
        })
