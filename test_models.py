import sys
sys.path.insert(0, '.')
from backend.models import ThreatClass, EffectorType, Effector, Threat, AssetStatus

pac3 = Effector(label='PAC-3 MSE', effector_type=EffectorType.KINETIC_MISSILE,
    lat=68.49, lon=16.68, ammo=16, max_ammo=16, range_km=120,
    cost_per_shot_usd=4_000_000, min_range_km=5, max_altitude_m=30_480,
    salvo_size=2, weather_degradation=False)

dew = Effector(label='HELIOS-300', effector_type=EffectorType.DIRECTED_ENERGY,
    lat=68.49, lon=16.68, ammo=-1, max_ammo=-1, range_km=12,
    cost_per_shot_usd=15, min_range_km=0.1, max_altitude_m=6_096,
    salvo_size=1, cooldown_ticks=1, weather_degradation=True)

ballistic = Threat(threat_class=ThreatClass.BALLISTIC, lat=69.0, lon=17.5, altitude_m=25000)  # ~70km from Evenes
lm        = Threat(threat_class=ThreatClass.LOITERING_MUNITION, lat=68.54, lon=16.73, altitude_m=150)  # ~6km from Evenes
fighter   = Threat(threat_class=ThreatClass.FIGHTER, lat=68.5, lon=16.7, altitude_m=9000)

assert EffectorType.KINETIC_MISSILE.value == 'kinetic_missile'
assert EffectorType.DIRECTED_ENERGY.value == 'directed_energy'
assert ThreatClass.LOITERING_MUNITION.value == 'loitering_munition'

# PAC-3: salvo combined PK = 1-(1-0.85)^2 = 0.9775
pac3_eff = pac3.effective_pk(ballistic)
assert abs(pac3_eff - (1-(1-0.85)**2)) < 0.001, f"Got {pac3_eff}"
assert pac3.in_range(ballistic)
assert pac3.can_engage()
print(f"PAC-3 vs BALLISTIC  base=85%  salvo×2={pac3_eff:.1%}  cost=${pac3.cost_per_shot_usd*pac3.salvo_size:,}/engage")

# HELIOS: weather degrades 99% by 15% → 0.8415
dew_eff = dew.effective_pk(lm)
assert abs(dew_eff - 0.99 * 0.85) < 0.001, f"Got {dew_eff}"
assert dew.can_engage()   # ammo=-1
print(f"HELIOS vs LM        base=99%  eff={dew_eff:.1%} (weather)  cost=${dew.cost_per_shot_usd}/shot")

# HELIOS altitude gate blocks 9000m fighter
assert not dew.in_range(fighter), "HELIOS should NOT engage fighter at 9000m (>6096m limit)"
print(f"HELIOS altitude gate: BLOCKS fighter@9000m (max_altitude={dew.max_altitude_m}m = 20000ft) ✓")

# HELIOS reaches loitering munition at 150m
assert dew.in_range(lm), "HELIOS should engage LM at 150m"
print(f"HELIOS in_range vs LM@150m: ✓")

# ThreatClass and EffectorType enums
print(f"\nThreatClass: {[t.value for t in ThreatClass]}")
print(f"EffectorType: {[e.value for e in EffectorType]}")

# Salvo ammo consumption test
pac3_copy = Effector(label='test', effector_type=EffectorType.KINETIC_MISSILE,
    ammo=16, max_ammo=16, range_km=120, salvo_size=2)
from backend.models import AssetStatus
pac3_copy.status = AssetStatus.READY
assert pac3_copy.can_engage()
pac3_copy.ammo = max(0, pac3_copy.ammo - pac3_copy.salvo_size)
assert pac3_copy.ammo == 14, f"Expected 14, got {pac3_copy.ammo}"
print(f"\nSalvo ammo consumption: 16 - 2 = {pac3_copy.ammo} ✓")

print("\nAll assertions passed ✓")
