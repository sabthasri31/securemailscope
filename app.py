"""
SecureMailScope - SIMULATION backend (Flask + scikit-learn)

Run:
    pip install flask scikit-learn numpy
    python app.py
Open: http://127.0.0.1:5000

What is simulated: the PCAP parsing stage (we generate realistic SMTP/IMAP/POP3 +
STARTTLS + TLS sessions instead of reading packets).
What is real: rule engine, feature extraction, RandomForest risk classifier,
IsolationForest anomaly detector, scoring, prioritisation, reports, REST API.
To go real later: replace generate_sessions() with a Scapy/PyShark/tshark parser
that returns dicts with the same fields.
"""
import datetime as dt
import hashlib
import html
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from sklearn.ensemble import IsolationForest, RandomForestClassifier

BASE = Path(__file__).parent
app = Flask(__name__)
TODAY = dt.date.today()

# ------------------------------------------------------------------ catalogues
VERSIONS = ["SSLv3", "TLSv1.0", "TLSv1.1", "TLSv1.2", "TLSv1.3"]
VNUM = {v: i for i, v in enumerate(VERSIONS)}

CIPHERS = {
    # TLS 1.3
    "TLS_AES_256_GCM_SHA384": dict(kex="ECDHE (x25519)", fs=True, enc="AES-256-GCM", v13=True),
    "TLS_AES_128_GCM_SHA256": dict(kex="ECDHE (x25519)", fs=True, enc="AES-128-GCM", v13=True),
    "TLS_CHACHA20_POLY1305_SHA256": dict(kex="ECDHE (x25519)", fs=True, enc="ChaCha20-Poly1305", v13=True),
    # TLS 1.2 modern
    "ECDHE-RSA-AES256-GCM-SHA384": dict(kex="ECDHE (P-256)", fs=True, enc="AES-256-GCM"),
    "ECDHE-ECDSA-AES128-GCM-SHA256": dict(kex="ECDHE (P-256)", fs=True, enc="AES-128-GCM"),
    "ECDHE-RSA-CHACHA20-POLY1305": dict(kex="ECDHE (P-256)", fs=True, enc="ChaCha20-Poly1305"),
    # CBC / no-FS
    "ECDHE-RSA-AES128-SHA": dict(kex="ECDHE (P-256)", fs=True, enc="AES-128-CBC", cbc=True),
    "DHE-RSA-AES128-SHA": dict(kex="DHE (1024-bit)", fs=True, enc="AES-128-CBC", cbc=True),
    "AES256-SHA256": dict(kex="RSA (static)", fs=False, enc="AES-256-CBC", cbc=True),
    "AES128-SHA": dict(kex="RSA (static)", fs=False, enc="AES-128-CBC", cbc=True),
    # broken
    "DES-CBC3-SHA": dict(kex="RSA (static)", fs=False, enc="3DES-CBC", weak="3DES (Sweet32)"),
    "RC4-SHA": dict(kex="RSA (static)", fs=False, enc="RC4", weak="RC4 (biased keystream)"),
}
C13 = [c for c, m in CIPHERS.items() if m.get("v13")]
C12_GOOD = ["ECDHE-RSA-AES256-GCM-SHA384", "ECDHE-ECDSA-AES128-GCM-SHA256", "ECDHE-RSA-CHACHA20-POLY1305"]

TIERS = {
    "good": dict(vmax="TLSv1.3", vmin="TLSv1.2", ciphers=C13 + C12_GOOD),
    "ok": dict(vmax="TLSv1.2", vmin="TLSv1.2", ciphers=C12_GOOD + ["ECDHE-RSA-AES128-SHA", "AES256-SHA256"]),
    "legacy": dict(vmax="TLSv1.2", vmin="TLSv1.0",
                   ciphers=["ECDHE-RSA-AES128-SHA", "DHE-RSA-AES128-SHA", "AES128-SHA", "AES256-SHA256", "DES-CBC3-SHA"]),
    "bad": dict(vmax="TLSv1.1", vmin="SSLv3", ciphers=["AES128-SHA", "DES-CBC3-SHA", "RC4-SHA"]),
}
SERVERS = [
    ("mail.nic-gov.example", "good"), ("smtp.bank-secure.example", "good"),
    ("mx.university.example", "ok"), ("imap.university.example", "ok"), ("mail.startup.example", "ok"),
    ("mail.hospital.example", "legacy"), ("relay.mfgco.example", "legacy"),
    ("pop.oldisp.example", "bad"), ("smtp.legacy-erp.example", "bad"),
]
TIER_WEIGHT = {"good": 0.30, "ok": 0.35, "legacy": 0.22, "bad": 0.13}
PORTS = {  # (protocol, port, mode)
    "SMTP": [(25, "STARTTLS"), (587, "STARTTLS"), (465, "IMPLICIT")],
    "IMAP": [(143, "STARTTLS"), (993, "IMPLICIT")],
    "POP3": [(110, "STARTTLS"), (995, "IMPLICIT")],
}

# ------------------------------------------------------------------ recommendations
FIX = {
    "PLAINTEXT": "Disable cleartext listeners (25/143/110 without TLS). Require TLS for all client auth and relay.",
    "STARTTLS_UNUSED": "Enforce STARTTLS (Postfix: smtpd_tls_security_level=encrypt; Dovecot: ssl=required). Add MTA-STS / DANE.",
    "STARTTLS_STRIPPED": "STARTTLS missing from capabilities vs. baseline: possible stripping attack. Enforce TLS + MTA-STS, investigate the network path.",
    "OLD_TLS": "Disable SSLv3/TLS1.0/1.1. Set minimum protocol to TLS 1.2 (prefer 1.3).",
    "WEAK_CIPHER": "Remove RC4/3DES/export ciphers. Allow only AEAD suites (AES-GCM, ChaCha20-Poly1305).",
    "CBC_CIPHER": "Prefer AEAD suites over CBC-mode suites; drop CBC-SHA1 suites.",
    "NO_FS": "Use only ECDHE/DHE key exchange so past traffic cannot be decrypted if the key leaks.",
    "CERT_EXPIRED": "Renew the certificate now and automate renewal (ACME / cert-manager).",
    "CERT_EXPIRING": "Certificate expires within 30 days: renew and monitor expiry.",
    "SELF_SIGNED": "Replace self-signed certificate with one from a trusted CA.",
    "CHAIN_INVALID": "Serve the full intermediate chain; make sure the root is publicly trusted.",
    "HOST_MISMATCH": "Issue a certificate whose SAN matches the mail hostname clients use.",
    "WEAK_SIG": "Reissue with SHA-256+ signature (SHA-1 / MD5 are broken).",
    "WEAK_KEY": "Reissue with RSA >= 2048 (prefer 3072) or ECDSA P-256+.",
    "ANOMALY": "Investigate the session: unusual handshake timing, cipher list or alert pattern (scanner, MITM, or misbehaving client).",
}
GENERIC = {
    "PLAINTEXT": "Cleartext mail sessions", "STARTTLS_UNUSED": "STARTTLS offered but not used",
    "STARTTLS_STRIPPED": "Possible STARTTLS stripping", "OLD_TLS": "Deprecated TLS/SSL versions",
    "WEAK_CIPHER": "Weak ciphers (RC4/3DES)", "CBC_CIPHER": "CBC-mode ciphers", "NO_FS": "No forward secrecy",
    "CERT_EXPIRED": "Expired certificates", "CERT_EXPIRING": "Certificates expiring < 30 days",
    "SELF_SIGNED": "Self-signed certificates", "CHAIN_INVALID": "Untrusted certificate chain",
    "HOST_MISMATCH": "Certificate hostname mismatch", "WEAK_SIG": "Weak certificate signature (SHA-1)",
    "WEAK_KEY": "Weak RSA key length", "ANOMALY": "Anomalous TLS behaviour",
}
SEV_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}


# ------------------------------------------------------------------ session simulation
def make_servers(rng):
    servers = []
    for i, (host, tier) in enumerate(SERVERS):
        t = TIERS[tier]
        cert = dict(subject=host, san_match=True, chain_len=3, chain_valid=True, self_signed=False)
        if tier == "good":
            cert.update(issuer="DigiCert TLS RSA SHA256 2020 CA1", sig_alg="sha256WithRSAEncryption",
                        key_alg=rng.choice(["RSA", "ECDSA"]), days_left=rng.randint(120, 300))
            cert["key_bits"] = 3072 if cert["key_alg"] == "RSA" else 256
        elif tier == "ok":
            cert.update(issuer="Let's Encrypt R11", sig_alg="sha256WithRSAEncryption", key_alg="RSA",
                        key_bits=2048, days_left=rng.randint(10, 80))
        elif tier == "legacy":
            cert.update(issuer=rng.choice(["Internal Corp CA", "Let's Encrypt R11"]),
                        sig_alg="sha256WithRSAEncryption", key_alg="RSA", key_bits=2048,
                        days_left=rng.randint(4, 25), san_match=rng.random() > 0.4)
            if cert["issuer"] == "Internal Corp CA":
                cert["chain_valid"] = False
        else:
            cert.update(issuer=host, self_signed=True, chain_valid=False, chain_len=1, sig_alg="sha1WithRSAEncryption",
                        key_alg="RSA", key_bits=rng.choice([1024, 1024, 2048]), days_left=rng.randint(-500, -10),
                        san_match=False)
        cert["not_after"] = str(TODAY + dt.timedelta(days=cert["days_left"]))
        servers.append(dict(host=host, ip=f"10.20.{i + 1}.{rng.randint(10, 250)}", tier=tier,
                            vmax=t["vmax"], vmin=t["vmin"], ciphers=t["ciphers"], cert=cert,
                            starttls_enforced=tier in ("good", "ok")))
    return servers


def pick_cipher(rng, srv, version):
    pool = [c for c in srv["ciphers"] if bool(CIPHERS[c].get("v13")) == (version == "TLSv1.3")]
    if not pool:
        pool = ["ECDHE-RSA-AES256-GCM-SHA384"]
    return rng.choice(pool)


def one_session(rng, nprng, srv, sid, anomalous=False):
    proto = rng.choices(list(PORTS), weights=[0.5, 0.3, 0.2])[0]
    port, mode = rng.choice(PORTS[proto])
    starttls_offered = mode == "STARTTLS"
    stripped = used = False
    plaintext = False
    if mode == "STARTTLS":
        if not srv["starttls_enforced"]:
            r = rng.random()
            if r < 0.25:
                starttls_offered, plaintext = (rng.random() > 0.5), True     # never upgraded
            elif r < 0.33:
                stripped, plaintext = True, True                              # capability removed in transit
            else:
                used = True
        else:
            used = True
    if plaintext and starttls_offered and not stripped:
        pass  # offered but client never sent STARTTLS

    s = dict(id=sid, client=f"192.168.1.{rng.randint(10, 60)}", server=srv["host"], server_ip=srv["ip"],
             tier=srv["tier"], protocol=proto, port=port, mode=mode,
             starttls_offered=starttls_offered, starttls_used=used, starttls_stripped=stripped,
             plaintext=plaintext, tls=None, cert=None, cert_visible=False, behavior={})
    start = dt.datetime.combine(TODAY, dt.time(8, 0)) + dt.timedelta(seconds=rng.randint(0, 36000))
    s["timestamp"] = start.isoformat(timespec="seconds")

    if not plaintext:
        cmax = rng.choices(["TLSv1.3", "TLSv1.2", "TLSv1.0"], weights=[0.65, 0.25, 0.10])[0]
        if VNUM[cmax] < VNUM[srv["vmin"]]:
            cmax = srv["vmin"]
        ver = min(srv["vmax"], cmax, key=lambda v: VNUM[v])
        cip = pick_cipher(rng, srv, ver)
        m = CIPHERS[cip]
        s["tls"] = dict(version=ver, cipher=cip, kex=m["kex"], enc=m["enc"], fs=m["fs"],
                        weak=m.get("weak"), cbc=bool(m.get("cbc")),
                        client_max=cmax, sni=srv["host"])
        if ver != "TLSv1.3":   # TLS 1.3 encrypts the Certificate message -> not visible passively
            s["cert_visible"] = True
            s["cert"] = dict(srv["cert"])
            s["cert"]["days_left"] = srv["cert"]["days_left"]

    # behaviour metrics (used by the anomaly detector)
    if plaintext:
        s["behavior"] = dict(handshake_ms=0.0, bytes=float(nprng.lognormal(9, 1)), duration=float(rng.uniform(1, 60)),
                             offered=0, alerts=0, retrans=int(nprng.poisson(1)))
    elif anomalous:
        s["behavior"] = dict(handshake_ms=float(rng.uniform(900, 4000)), bytes=float(nprng.lognormal(6, 1)),
                             duration=float(rng.uniform(0.2, 4)), offered=rng.randint(45, 70),
                             alerts=rng.randint(3, 7), retrans=rng.randint(15, 40))
    else:
        s["behavior"] = dict(handshake_ms=float(nprng.lognormal(4.4, 0.4)), bytes=float(nprng.lognormal(10, 1.2)),
                             duration=float(rng.uniform(1, 120)), offered=rng.randint(12, 25),
                             alerts=int(rng.random() < 0.04), retrans=int(nprng.poisson(1.2)))
    s["_gt_anomaly"] = anomalous
    s["transcript"] = build_transcript(s)
    return s


def generate_sessions(seed=7, n=90):
    rng, nprng = random.Random(seed), np.random.default_rng(seed)
    servers = make_servers(rng)
    weights = [TIER_WEIGHT[s["tier"]] for s in servers]
    sessions = []
    for i in range(n):
        srv = rng.choices(servers, weights=weights)[0]
        sessions.append(one_session(rng, nprng, srv, f"S-{i + 1:04d}", anomalous=rng.random() < 0.05))
    return servers, sessions


def build_transcript(s):
    t, ev = 0, []

    def add(d, layer, text, dt_ms=None):
        nonlocal t
        t += dt_ms if dt_ms is not None else 8
        ev.append(dict(t=t, dir=d, layer=layer, text=text))

    p, host = s["protocol"], s["server"]
    cap_tls = s["starttls_offered"] and not s["starttls_stripped"]
    if s["mode"] == "STARTTLS":
        if p == "SMTP":
            add("S>C", "SMTP", f"220 {host} ESMTP ready")
            add("C>S", "SMTP", "EHLO client.local")
            add("S>C", "SMTP", "250-" + host + (" | 250-STARTTLS" if cap_tls else " | (no STARTTLS in capability list)") + " | 250 8BITMIME")
            if s["starttls_used"]:
                add("C>S", "SMTP", "STARTTLS"); add("S>C", "SMTP", "220 2.0.0 Ready to start TLS")
        elif p == "IMAP":
            add("S>C", "IMAP", "* OK IMAP4rev1 ready"); add("C>S", "IMAP", "a1 CAPABILITY")
            add("S>C", "IMAP", "* CAPABILITY IMAP4rev1 " + ("STARTTLS " if cap_tls else "") + "AUTH=PLAIN")
            if s["starttls_used"]:
                add("C>S", "IMAP", "a2 STARTTLS"); add("S>C", "IMAP", "a2 OK Begin TLS negotiation now")
        else:
            add("S>C", "POP3", "+OK POP3 ready"); add("C>S", "POP3", "CAPA")
            add("S>C", "POP3", "+OK " + ("STLS " if cap_tls else "") + "USER")
            if s["starttls_used"]:
                add("C>S", "POP3", "STLS"); add("S>C", "POP3", "+OK Begin TLS negotiation")
        if s["plaintext"]:
            add("C>S", p, "AUTH PLAIN <base64 credentials VISIBLE IN CLEARTEXT>", 20)
            return ev
    else:
        add("C>S", "TCP", f"SYN -> {s['port']} (implicit TLS port)")

    tls = s["tls"]
    add("C>S", "TLS", f"ClientHello  max={tls['client_max']}  ciphers_offered={s['behavior']['offered']}  SNI={tls['sni']}", 12)
    add("S>C", "TLS", f"ServerHello  version={tls['version']}  cipher={tls['cipher']}", 25)
    if s["cert_visible"]:
        c = s["cert"]
        add("S>C", "TLS", f"Certificate  CN={c['subject']}  issuer={c['issuer']}  sig={c['sig_alg']}  key={c['key_alg']}-{c['key_bits']}")
        add("S>C", "TLS", f"ServerKeyExchange  {tls['kex']}")
    else:
        add("S>C", "TLS", "EncryptedExtensions / Certificate  [encrypted in TLS 1.3 - not visible passively]")
    add("C>S", "TLS", "Finished"); add("S>C", "TLS", "Finished")
    add("C>S", p, "[application data - encrypted]", 5)
    return ev


# ------------------------------------------------------------------ rule engine
def assess_rules(s):
    f = []

    def add(fid, sev, title, detail, pts):
        f.append(dict(id=fid, severity=sev, title=title, detail=detail, points=pts, fix=FIX[fid]))

    if s["plaintext"] and s["starttls_stripped"]:
        add("STARTTLS_STRIPPED", "critical", "Possible STARTTLS stripping", "Server normally advertises STARTTLS; it is missing in this session and traffic stayed in cleartext.", 55)
    elif s["plaintext"] and s["starttls_offered"]:
        add("STARTTLS_UNUSED", "high", "STARTTLS offered but never used", "Server offered STARTTLS, session continued in cleartext.", 45)
    elif s["plaintext"]:
        add("PLAINTEXT", "critical", "Cleartext mail session", "No TLS at all; credentials and content readable by any observer.", 55)
    tls = s["tls"]
    if tls:
        v = tls["version"]
        if v in ("SSLv3", "TLSv1.0", "TLSv1.1"):
            add("OLD_TLS", "critical" if v != "TLSv1.1" else "high", f"Deprecated protocol {v}", "Deprecated by RFC 8996 / vulnerable to BEAST, POODLE-class attacks.", {"SSLv3": 50, "TLSv1.0": 35, "TLSv1.1": 28}[v])
        if tls["weak"]:
            add("WEAK_CIPHER", "critical", f"Weak cipher {tls['cipher']}", f"Uses {tls['weak']}.", 35)
        elif tls["cbc"]:
            add("CBC_CIPHER", "low", f"CBC-mode cipher {tls['cipher']}", "Non-AEAD cipher; susceptible to padding-oracle class issues.", 8)
        if not tls["fs"]:
            add("NO_FS", "medium", "No forward secrecy", f"Key exchange {tls['kex']} does not provide forward secrecy.", 15)
    if s["cert_visible"]:
        c = s["cert"]
        if c["days_left"] < 0:
            add("CERT_EXPIRED", "high", "Certificate expired", f"Expired {-c['days_left']} days ago ({c['not_after']}).", 30)
        elif c["days_left"] < 30:
            add("CERT_EXPIRING", "medium", "Certificate expiring soon", f"{c['days_left']} days left ({c['not_after']}).", 8)
        if c["self_signed"]:
            add("SELF_SIGNED", "high", "Self-signed certificate", "Clients cannot verify server identity.", 18)
        elif not c["chain_valid"]:
            add("CHAIN_INVALID", "medium", "Certificate chain not trusted", f"Issuer '{c['issuer']}' does not chain to a public root.", 15)
        if not c["san_match"]:
            add("HOST_MISMATCH", "medium", "Hostname mismatch", "Certificate SAN/CN does not match the server name.", 12)
        if c["sig_alg"].startswith(("sha1", "md5")):
            add("WEAK_SIG", "high", f"Weak signature {c['sig_alg']}", "SHA-1/MD5 signatures allow forgery.", 20)
        if c["key_alg"] == "RSA" and c["key_bits"] < 2048:
            add("WEAK_KEY", "high", f"Weak RSA key ({c['key_bits']} bit)", "Below the 2048-bit minimum.", 25)
    return f


def features(s):
    tls, c = s["tls"], s["cert"] or {}
    return [
        int(s["plaintext"]), int(s["starttls_used"]), int(s["starttls_stripped"] or (s["starttls_offered"] and s["plaintext"])),
        VNUM[tls["version"]] if tls else -1,
        int(bool(tls and tls["fs"])), int(bool(tls and tls["weak"])), int(bool(tls and tls["cbc"])),
        int(s["cert_visible"]),
        (c.get("key_bits", 0) / 4096) if s["cert_visible"] else 0,
        (max(-400, min(400, c.get("days_left", 400))) / 400) if s["cert_visible"] else 1,
        int(c.get("self_signed", False)), int(bool(c) and not c.get("chain_valid", True)),
        int(bool(c) and c.get("sig_alg", "").startswith(("sha1", "md5"))), int(bool(c) and not c.get("san_match", True)),
    ]


def behavior_vec(s):
    b = s["behavior"]
    return [b["handshake_ms"], math.log10(b["bytes"] + 1), b["duration"], b["offered"], b["alerts"], b["retrans"]]


LEVELS = ["low", "medium", "high", "critical"]
CENTER = np.array([10, 35, 58, 85])


def level_of(score):
    return "critical" if score >= 70 else "high" if score >= 45 else "medium" if score >= 20 else "low"


# ------------------------------------------------------------------ ML models (trained once on simulated corpus)
class Models:
    def __init__(self):
        _, train = generate_sessions(seed=1234, n=2500)
        X, y = [], []
        for s in train:
            score = min(100, sum(f["points"] for f in assess_rules(s)))
            X.append(features(s)); y.append(LEVELS.index(level_of(score)))
        self.rf = RandomForestClassifier(n_estimators=150, max_depth=9, random_state=0).fit(X, y)
        self.classes = list(self.rf.classes_)
        normal = [behavior_vec(s) for s in train if s["tls"] and not s["_gt_anomaly"]]
        self.iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=0).fit(normal)

    def risk(self, s):
        proba = self.rf.predict_proba([features(s)])[0]
        full = np.zeros(4); full[self.classes] = proba
        return float(full @ CENTER), LEVELS[int(full.argmax())], float(full.max())

    def anomaly(self, s):
        if not s["tls"]:
            return 0.0, False
        v = [behavior_vec(s)]
        score = float(np.clip(50 - self.iso.decision_function(v)[0] * 250, 0, 100))
        return score, bool(self.iso.predict(v)[0] == -1)


MODELS = Models()


def analyse(s):
    findings = assess_rules(s)
    rule = min(100, sum(f["points"] for f in findings))
    ai_score, ai_level, conf = MODELS.risk(s)
    a_score, is_anom = MODELS.anomaly(s)
    if is_anom:
        b = s["behavior"]
        findings.append(dict(id="ANOMALY", severity="medium", title="Anomalous TLS behaviour",
                             detail=f"Handshake {b['handshake_ms']:.0f} ms, {b['offered']} ciphers offered, {b['alerts']} alerts, {b['retrans']} retransmits - outside normal profile.",
                             points=10, fix=FIX["ANOMALY"]))
    risk = round(0.65 * rule + 0.35 * ai_score)
    findings.sort(key=lambda f: (-SEV_ORDER[f["severity"]], -f["points"]))
    s.update(findings=findings, rule_score=rule, ai_score=round(ai_score), ai_level=ai_level,
             ai_confidence=round(conf, 2), anomaly_score=round(a_score), anomaly=is_anom,
             risk_score=risk, risk_level=level_of(risk), priority=round(risk + 0.3 * a_score))
    return s


# ------------------------------------------------------------------ state + summary
STATE = {}


def grade(score):
    return "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"


def build_state(seed=7, n=90, source="simulation"):
    servers, sessions = generate_sessions(seed, n)
    for s in sessions:
        analyse(s)
    sessions.sort(key=lambda s: -s["priority"])
    risks = [s["risk_score"] for s in sessions]
    posture = max(0, round(100 - (0.7 * np.mean(risks) + 0.3 * np.percentile(risks, 90))))
    agg = defaultdict(lambda: dict(count=0, severity="low", title="", fix="", servers=set()))
    for s in sessions:
        for f in s["findings"]:
            a = agg[f["id"]]; a["count"] += 1; a["title"] = GENERIC[f["id"]]
            a["fix"] = f["fix"]; a["servers"].add(s["server"])
            if SEV_ORDER[f["severity"]] > SEV_ORDER[a["severity"]]: a["severity"] = f["severity"]
    top = sorted(([k, v] for k, v in agg.items()), key=lambda kv: (-SEV_ORDER[kv[1]["severity"]], -kv[1]["count"]))
    srv = defaultdict(list)
    for s in sessions: srv[s["server"]].append(s["risk_score"])
    tls = [s for s in sessions if s["tls"]]
    injected = [s for s in sessions if s["_gt_anomaly"]]
    STATE.update(
        source=source, seed=seed, servers=servers, sessions=sessions,
        summary=dict(
            total=len(sessions), posture_score=int(posture), grade=grade(posture),
            by_protocol=dict(Counter(s["protocol"] for s in sessions)),
            by_level=dict(Counter(s["risk_level"] for s in sessions)),
            by_version=dict(Counter(s["tls"]["version"] if s["tls"] else "Cleartext" for s in sessions)),
            by_mode=dict(Counter("Cleartext" if s["plaintext"] else s["mode"] for s in sessions)),
            fs_pct=round(100 * sum(s["tls"]["fs"] for s in tls) / max(1, len(tls))),
            starttls=dict(offered=sum(s["starttls_offered"] for s in sessions),
                          upgraded=sum(s["starttls_used"] for s in sessions),
                          stripped=sum(s["starttls_stripped"] for s in sessions)),
            anomalies=sum(s["anomaly"] for s in sessions),
            cert_issues=sum(1 for s in sessions if s["cert_visible"] and any(f["id"] in ("CERT_EXPIRED", "CERT_EXPIRING", "SELF_SIGNED", "CHAIN_INVALID", "HOST_MISMATCH", "WEAK_SIG", "WEAK_KEY") for f in s["findings"])),
            tls13_hidden_certs=sum(1 for s in tls if not s["cert_visible"]),
            servers=sorted([dict(host=h, avg_risk=round(float(np.mean(v))), max_risk=max(v), sessions=len(v)) for h, v in srv.items()], key=lambda x: -x["avg_risk"]),
            top_findings=[dict(id=k, count=v["count"], severity=v["severity"], title=v["title"], fix=v["fix"], servers=sorted(v["servers"])) for k, v in top],
            sim_eval=dict(injected_anomalies=len(injected), caught=sum(1 for s in injected if s["anomaly"]), false_alarms=sum(1 for s in sessions if s["anomaly"] and not s["_gt_anomaly"])),
        ))


def public(s, full=False):
    d = {k: v for k, v in s.items() if not k.startswith("_") and (full or k != "transcript")}
    return d


build_state()


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/api/summary")
def api_summary():
    return jsonify(dict(source=STATE["source"], seed=STATE["seed"], **STATE["summary"]))


@app.get("/api/sessions")
def api_sessions():
    lvl, proto, q = request.args.get("level"), request.args.get("protocol"), (request.args.get("q") or "").lower()
    out = [s for s in STATE["sessions"]
           if (not lvl or s["risk_level"] == lvl) and (not proto or s["protocol"] == proto)
           and (not q or q in s["server"].lower() or q in s["client"] or q in s["id"].lower())]
    return jsonify([public(s) for s in out])


@app.get("/api/sessions/<sid>")
def api_session(sid):
    for s in STATE["sessions"]:
        if s["id"] == sid:
            return jsonify(public(s, full=True))
    return jsonify(error="not found"), 404


@app.post("/api/simulate")
def api_simulate():
    body = request.get_json(silent=True) or {}
    build_state(int(body.get("seed", random.randint(1, 10 ** 6))), min(300, int(body.get("n", 90))), "simulation")
    return jsonify(ok=True)


@app.post("/api/upload")
def api_upload():
    """Accepts a .pcap/.pcapng, checks the magic bytes, then seeds the SIMULATED analysis from its hash."""
    f = request.files.get("file")
    if not f:
        return jsonify(error="no file"), 400
    data = f.read()
    magic = data[:4]
    valid = magic in (b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x4d\x3c\xb2\xa1", b"\x0a\x0d\x0d\x0a")
    seed = int(hashlib.sha256(data).hexdigest()[:8], 16)
    n = max(40, min(150, len(data) // 2048))
    build_state(seed, n, f"simulated analysis of '{f.filename}'")
    return jsonify(ok=True, pcap_magic_valid=valid, bytes=len(data), sessions=n,
                   note="Demo mode: packets are not parsed; results are simulated deterministically from the file hash.")


@app.get("/api/report.json")
def report_json():
    payload = dict(generated=str(TODAY), source=STATE["source"], summary=STATE["summary"],
                   sessions=[public(s, full=True) for s in STATE["sessions"]])
    return Response(json.dumps(payload, indent=2), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=securemailscope_report.json"})


@app.get("/api/report.html")
def report_html():
    S, e = STATE["summary"], html.escape
    rows = "".join(f"<tr><td>{e(f['severity'])}</td><td>{e(f['title'])}</td><td>{f['count']}</td><td>{e(f['fix'])}</td></tr>" for f in S["top_findings"])
    srv = "".join(f"<tr><td>{e(x['host'])}</td><td>{x['avg_risk']}</td><td>{x['max_risk']}</td><td>{x['sessions']}</td></tr>" for x in S["servers"])
    top = "".join(f"<tr><td>{s['id']}</td><td>{e(s['server'])}</td><td>{s['protocol']}/{s['port']}</td><td>{e(s['tls']['version'] if s['tls'] else 'cleartext')}</td><td>{s['risk_score']} ({s['risk_level']})</td></tr>" for s in STATE["sessions"][:15])
    page = f"""<!doctype html><meta charset=utf-8><title>SecureMailScope report</title>
<style>body{{font:14px/1.5 system-ui;max-width:900px;margin:32px auto;padding:0 16px;color:#14232e}}table{{border-collapse:collapse;width:100%;margin:12px 0 28px}}td,th{{border:1px solid #cfd8dc;padding:6px 8px;text-align:left;vertical-align:top}}th{{background:#eef2f5}}</style>
<h1>SecureMailScope - cryptographic posture report</h1>
<p>Generated {TODAY} | Source: {e(STATE['source'])} | Sessions: {S['total']}</p>
<h2>Posture: grade {S['grade']} ({S['posture_score']}/100)</h2>
<p>Forward secrecy: {S['fs_pct']}% | STARTTLS upgraded {S['starttls']['upgraded']}/{S['starttls']['offered']} | anomalies: {S['anomalies']}</p>
<h2>Prioritised findings</h2><table><tr><th>Severity</th><th>Finding</th><th>Sessions</th><th>Recommendation</th></tr>{rows}</table>
<h2>Servers by risk</h2><table><tr><th>Host</th><th>Avg risk</th><th>Max risk</th><th>Sessions</th></tr>{srv}</table>
<h2>Top 15 sessions to review</h2><table><tr><th>ID</th><th>Server</th><th>Proto/Port</th><th>TLS</th><th>Risk</th></tr>{top}</table>"""
    return Response(page, mimetype="text/html", headers={"Content-Disposition": "attachment; filename=securemailscope_report.html"})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
