from backend.models import ThreatClass, EffectorType, Effector, Threat, AssetStatus
from backend.state import ScenarioState
from backend.engine.optimizer import optimise

# ── Unit test: PAC-3 MSE
pac3 = Effector(
    label='PAC-3 MSE Bty 1', effector_type=EffectorType.KINETIC_MISSILE,
    lat=68.49, lon=16.68, ammo=16, max_ammo=16, range_km=120,
    cost_per_shot_usd=4_000_000, min_range_km=5, max_altitude_m=30_480,
    salvo_size=2, weather_degradation=False
)
# ── Unit test: HELIOS-300
dew = Effector(
    label='HELIOS-300 #1', effector_type=EffectorType.DIRECTED_ENERGY,
    lat=68.49, lon=16.68, ammo=-1, max_ammo=-1, range_km=12,
    cost_per_shot_usd=15, min_range_km=0.1, max_altitude_m=6_096,
    salvo_size=1, cooldown_ticks=1, weather_degradation=True
)

ballistic = Threat(threat_class=ThreatClass.BALLISTIC, lat=69.5, lon=20.0, altitude_m=25000, speed_kmh=5000)
lm        = Threat(threat_class=ThreatClass.LOITERING_MUNITION, lat=68.6, lon=16.8, altitude_m=150, speed_kmh=180)
fighter   = Threat(threat_class=ThreatClass.FIGHTER, lat=68.5, lon=16.7, altitude_m=9000, speed_kmh=1800)

print("=== PAC-3 MSE vs BALLISTIC ===")
print(f"  base_pk      = {pac3.pk(ballistic):.0%}")
print(f"  salvo_pk×2   = {pac3.effective_pk(ballistic):.0%}  (1-(1-0.85)^2)")
print(f"  cost/engage  = ${pac3.cost_per_shot_usd * pac3.salvo_size:,}")
print(f"  in_range     = {pac3.in_range(ballistic)}")
print(f"  can_engage   = {pac3.can_engage()}")

print()
print("=== HELIOS-300 vs LOITERING MUNITION ===")
print(f"  base_pk      = {dew.pk(lm):.0%}")
print(f"  effective_pk = {dew.effective_pk(lm):.0%}  (base × 0.85 weather)")
print(f"  cost/shot    = ${dew.cost_per_shot_usd}")
print(f"  in_range     = {dew.in_range(lm)}")
print(f"  can_engage   = {dew.can_engage()}  (ammo=-1 = unlimited)")

print()
print("=== HELIOS altitude gate ===")
print(f"  fighter @ 9000m altitude: in_range = {dew.in_range(fighter)}  (max={dew.max_altitude_m}m ≈ 20000ft)")

print()
print("=== Full scenario optimizer test ===")
s = ScenarioState()
s.spawn_threat(threat_class=ThreatClass.LOITERING_MUNITION, priority=7)
s.spawn_threat(threat_class=ThreatClass.BALLISTIC, priority=10)
threats   = s.get_threats()
effectors = s.get_effectors()
orders, warnings = optimise(threats, effectors)
print(f"  Threats={len(threats)}, Effectors={len(effectors)}, Orders={len(orders)}, Warnings={len(warnings)}")
for o in orders:
    print(f"  {o.effector_label:25s} → {o.threat_label} [{o.threat_class:20s}]  PK={o.pk:.0%}  {o.rationale[:70]}")

# Verify new effector types appear in scenario
types = [e.effector_type.value for e in effectors]
print()
print("=== New effector types in scenario ===")
print(f"  kinetic_missile present: {'kinetic_missile' in types}")
print(f"  directed_energy present: {'directed_energy' in types}")
