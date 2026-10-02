"""Message integrity and replay protection for edge -> fog -> cloud messages.

Each message carries (device_id, counter, timestamp, payload) and an HMAC-SHA256 tag computed
with a per-device key provisioned at enrolment (RFC 2104). The fog verifies the tag in constant
time, rejects timestamps outside a freshness window and counters that are not strictly
increasing (replay). Confidentiality in transit is assumed to be provided by TLS and is not
re-implemented here.

`benchmark()` measures the real CPU cost of signing and verifying messages of the sizes used
in the framework (routine record, alert, full image).
"""
import hashlib
import hmac
import json
import os
import struct
import time


class Signer:
    def __init__(self, device_id: int, key: bytes):
        self.device_id, self.key, self.counter = device_id, key, 0

    def sign(self, payload: bytes, now=None):
        self.counter += 1
        ts = time.time() if now is None else now
        header = struct.pack(">IQd", self.device_id, self.counter, ts)
        tag = hmac.new(self.key, header + payload, hashlib.sha256).digest()
        return header + payload + tag


class Verifier:
    def __init__(self, keys: dict, window_s: float = 30.0):
        self.keys, self.window, self.last = keys, window_s, {}

    def verify(self, msg: bytes, now=None):
        header, payload, tag = msg[:20], msg[20:-32], msg[-32:]
        dev, ctr, ts = struct.unpack(">IQd", header)
        key = self.keys.get(dev)
        if key is None:
            return False, "unknown device"
        if not hmac.compare_digest(tag, hmac.new(key, header + payload, hashlib.sha256).digest()):
            return False, "bad tag"
        now = time.time() if now is None else now
        if abs(now - ts) > self.window:
            return False, "stale"
        if ctr <= self.last.get(dev, 0):
            return False, "replay"
        self.last[dev] = ctr
        return True, "ok"


def benchmark(sizes=(600, 16_000, 80_000), n=2000, seed=0):
    rnd = os.urandom
    key = rnd(32)
    out = {}
    for s in sizes:
        payload = rnd(s)
        sg, vf = Signer(1, key), Verifier({1: key})
        t0 = time.perf_counter()
        msgs = [sg.sign(payload) for _ in range(n)]
        t1 = time.perf_counter()
        ok = all(vf.verify(m)[0] for m in msgs)
        t2 = time.perf_counter()
        out[s] = {"sign_us": 1e6 * (t1 - t0) / n, "verify_us": 1e6 * (t2 - t1) / n, "all_valid": ok,
                  "overhead_bytes": 20 + 32}
    # tamper / replay checks
    sg, vf = Signer(1, key), Verifier({1: key})
    m = sg.sign(b"x" * 600)
    tampered = m[:30] + bytes([m[30] ^ 1]) + m[31:]
    out["tamper_rejected"] = vf.verify(tampered)[1] == "bad tag"
    vf.verify(m)
    out["replay_rejected"] = vf.verify(m)[1] == "replay"
    return out


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=1))
