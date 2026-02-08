#!/usr/bin/env python3
"""Quick Redis connectivity diagnostic."""
import os, redis, sys
from dotenv import load_dotenv

load_dotenv("/Users/bhoumiksangle/Desktop/Nirmala/central/.env")

host = os.getenv("REDIS_HOST", "localhost")
port = int(os.getenv("REDIS_PORT", "6379"))
user = os.getenv("REDIS_USERNAME", "")
pw = os.getenv("REDIS_PASSWORD", "")

print(f"REDIS_HOST = {host}", flush=True)
print(f"REDIS_PORT = {port}", flush=True)
print(f"REDIS_USER = {user}", flush=True)
print(f"REDIS_PASS = {pw[:6]}..." if pw else "REDIS_PASS = (empty)", flush=True)

# Cloud Redis
print("\n--- Cloud Redis ---", flush=True)
try:
    r = redis.Redis(host=host, port=port, username=user, password=pw,
                    db=0, decode_responses=True,
                    socket_timeout=5, socket_connect_timeout=5)
    r.ping()
    print(f"CONNECTED to {host}:{port}", flush=True)
except Exception as e:
    print(f"FAILED: {e}", flush=True)

# Local Redis
print("\n--- Local Redis ---", flush=True)
try:
    r2 = redis.Redis(host="localhost", port=6379, db=0,
                     decode_responses=True, socket_timeout=3)
    r2.ping()
    keys = r2.keys("*")
    print(f"CONNECTED to localhost:6379  ({len(keys)} keys)", flush=True)
    for k in sorted(keys)[:10]:
        print(f"  {k}", flush=True)
    if len(keys) > 10:
        print(f"  ... and {len(keys)-10} more", flush=True)
except Exception as e2:
    print(f"FAILED: {e2}", flush=True)
