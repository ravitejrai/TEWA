import math

from backend.state import ScenarioState, SPAWN_ZONE
from backend.engine.optimizer import optimise
from backend.models import ThreatClass, ThreatStatus

def hav(lat1, lon1, lat2, lon2):
    R = 6371
    dl = math.radians(lat2 - lat1)
    dn = math.radians(lon2 - lon1)
    a = math.sin(dl/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dn/2)**2
    return R * 2 * math.asin(math.sqrt(a))

F7 = (58.43, 12.72)

print("=== New spawn zones near F7 Satenas ===")
for z in SPAWN_ZONE:
    d = hav(F7[0], F7[1], z["lat"], z["lon"])
    if d <= 350:
        flag = "<< PATRIOT IN RANGE" if d <= 250 else "<< PAC-3 range only" if d <= 120 else ""
        print(f"  {z['label']:<25} {d:.0f}km  {flag}")

print()
print("=== SLS doctrine test ===")
s = ScenarioState()

# Spawn a ballistic, then move it into Patriot range of F7 and mark ENGAGED
t = s.spawn_threat(threat_class=ThreatClass.BALLISTIC)
t.lat, t.lon = 57.8, 13.5   # ~220km from F7, within Patriot range
t.status = ThreatStatus.ENGAGED  # pretend JAS-39 already fired

threats = list(s.threats.values())
effs = list(s.effectors.values())
orders, warnings = optimise(threats, effs)

sls = [o for o in orders if "SLS" in (o.rationale or "")]
print(f"  SLS orders generated: {len(sls)}")
for o in sls:
    print(f"  -> {o.effector_label} -> {o.threat_label} PK={o.pk:.0%}")
    print(f"     {o.rationale[:90]}")

adv = [w for w in warnings if "SLS" in w["msg"]]
print(f"  SLS advisories: {len(adv)}")
for w in adv:
    print(f"  [{w['level']}] {w['msg'][:90]}")

assert len(sls) > 0, "No SLS orders generated - F7 SAMs still not engaging!"
print()
print("All checks PASSED")
