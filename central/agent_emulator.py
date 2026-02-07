import requests
import json
import uuid
import random
import time
from datetime import datetime

BASE_URL = "http://localhost:8000/intent"
AGENTS = ["bank_01", "bank_02", "bank_03", "fund_A", "fund_B"]
TARGETS = ["bank_01", "bank_02", "bank_03", "ccp_central", "exchange_main"]

def generate_uuid():
    return str(uuid.uuid4())

def get_random_agent():
    return random.choice(AGENTS)

def get_random_target(exclude=None):
    t = random.choice(TARGETS)
    while t == exclude:
        t = random.choice(TARGETS)
    return t

def generate_lending_intent(tick):
    # Matches "bank lending to bank VIA CCP"
    agent = get_random_agent()
    borrower = get_random_target(exclude=agent)
    return {
        "intent_id": generate_uuid(),
        "tick": tick,
        "agent_id": agent,
        "action_type": "provide_credit",
        "payload": {
            "target": borrower,
            "amount": random.randint(50000, 200000)
        },
        "visibility": "private"
    }

def generate_otc_intent(tick):
    agent = get_random_agent()
    final_dest = get_random_target(exclude=agent)
    return {
        "intent_id": generate_uuid(),
        "tick": tick,
        "agent_id": agent,
        "action_type": "route_otc_proposal",
        "payload": {
            "router": "ccp_central",
            "final_destination": final_dest,
            "encrypted_content": {
                "type": "otc_loan_offer",
                "amount": random.randint(50000, 200000),
                "interest_rate": 0.05
            }
        },
        "visibility": "private"
    }

def generate_repay_intent(tick):
    agent = get_random_agent()
    # Ideally should target someone we owe, but for emulation random is okay
    target = get_random_target(exclude=agent)
    return {
        "intent_id": generate_uuid(),
        "tick": tick,
        "agent_id": agent,
        "action_type": "repay_interbank_loan",
        "payload": {
            "target": target,
            "amount": random.randint(10000, 100000),
            "interest": random.randint(100, 5000)
        },
        "visibility": "private"
    }

def generate_fire_sale_intent(tick):
    agent = get_random_agent()
    return {
        "intent_id": generate_uuid(),
        "tick": tick,
        "agent_id": agent,
        "action_type": "fire_sale_asset",
        "payload": {
            "target": "exchange_main",
            "asset_type": "liquid_bond",
            "amount": random.randint(1000, 10000)
        },
        "visibility": "public"
    }

def generate_margin_call_pay_intent(tick):
    agent = get_random_agent()
    return {
        "intent_id": generate_uuid(),
        "tick": tick,
        "agent_id": agent,
        "action_type": "pay_margin_call",
        "payload": {
            "target": "ccp_central",
            "amount": random.randint(5000, 25000)
        },
        "visibility": "private"
    }

def generate_default_intent(tick):
    agent = get_random_agent()
    return {
        "intent_id": generate_uuid(),
        "tick": tick,
        "agent_id": agent,
        "action_type": "declare_default",
        "payload": {
            "target": "ccp_central",
            "uncovered_amount": random.randint(100000, 500000)
        },
        "visibility": "public"
    }

def generate_deposit_default_fund(tick):
    agent = get_random_agent()
    return {
        "intent_id": generate_uuid(),
        "tick": tick,
        "agent_id": agent,
        "action_type": "deposit_default_fund",
        "payload": {
            "target": "ccp_central",
            "amount": random.randint(100000, 500000)
        },
        "visibility": "public"
    }

GENERATORS = [
    generate_lending_intent,
    generate_otc_intent,
    generate_repay_intent,
    generate_fire_sale_intent,
    generate_margin_call_pay_intent,
    generate_deposit_default_fund
]

def main():
    tick = 0
    print(f"Starting Agent Emulator sending to {BASE_URL}...")
    print("Live Graph Dashboard: http://localhost:8000/")
    
    try:
        while True:
            tick += 1
            
            # 5% chance of default, else random other action
            if random.random() < 0.05:
                intent_func = generate_default_intent
            else:
                intent_func = random.choice(GENERATORS)
                
            intent = intent_func(tick)
            
            print(f"\n[Tick {tick}] Sending {intent['action_type']} from {intent['agent_id']}...")
            
            try:
                response = requests.post(BASE_URL, json=intent)
                if response.status_code == 200:
                    print(f"  -> Success. ID: {response.json().get('intent_id')}")
                    stats = response.json().get('network_stats')
                    if stats:
                        # Print summary of banks
                        print(f"  -> CCP Default Fund: {stats.get('ccp_default_fund'):,.2f}")
                        print(f"  -> Exchange Vol: {stats.get('exchange_vol'):.2f}")
                        if 'banks_stats' in stats:
                            print("  -> Banks Overview:")
                            for b_id, b_data in list(stats['banks_stats'].items())[:3]: # Show top 3
                                print(f"     {b_id}: Cash={b_data.get('cash'):,.0f} NetWorth={b_data.get('net_worth'):,.0f}")
                else:
                    print(f"  -> Failed: {response.status_code} - {response.text}")
            except requests.exceptions.RequestException as e:
                print(f"  -> Connection Error: {e}")
                
            time.sleep(2) # Wait 2 seconds between intents
            
    except KeyboardInterrupt:
        print("\nStopping Agent Emulator.")

if __name__ == "__main__":
    main()
