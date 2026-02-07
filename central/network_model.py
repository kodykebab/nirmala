import networkx as nx
from models import AgentIntent
from typing import Dict, Any

class FinancialNetwork:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.initialize_infrastructure()

    def initialize_infrastructure(self):
        # Initialize Central Counterparty (CCP)
        self.graph.add_node("ccp_central", 
                            type="ccp", 
                            label="CCP Central",
                            default_fund=100000.0,
                            base_rate=0.05,
                            current_margin_rate=0.05,
                            safe_limit=1000000.0,
                            panic_mode=False,
                            total_exposure=0.0)
        
        # Initialize Exchange
        self.graph.add_node("exchange_main", 
                            type="exchange", 
                            label="Exchange Main",
                            total_sells=0.0,
                            market_volatility=10.0, # Initial vol
                            liquid_price=100.0,
                            toxic_price=100.0)

    def _ensure_bank(self, agent_id):
        # Check if node exists. If it exists but lacks 'type' or 'pure_cash', it was likely implicitly created by an edge.
        node = self.graph.nodes.get(agent_id)
        if node is None or node.get("type") != "bank":
            # Initialize or Update
            self.graph.add_node(agent_id, 
                                type="bank",
                                label=agent_id,
                                status="active",
                                # Initial State (preserve existing if any, though unlikely for implicit nodes)
                                pure_cash=1000000.0,
                                liabilities=0.0,
                                default_fund_contribution=0.0,
                                liquid_bond_count=1000,
                                toxic_count=1000,
                                credit_exposure_value=0.0,
                                margin_call_due=0.0,
                                # Derived
                                total_assets=0.0,
                                net_worth=0.0,
                                fire_sale_amount=0.0)

    def _update_exchange_prices(self):
        exch = self.graph.nodes["exchange_main"]
        total_sells = exch.get("total_sells", 0.0)
        
        # Formula: Liquid_Price = 100 - (Total_Sells * 0.1)
        # Note: Scaling factor might need adjustment if sells are huge.
        exch["liquid_price"] = max(0, 100.0 - (total_sells * 0.001)) # Reduced factor for demo stability
        
        # Formula: Toxic_Price = 100 - (Total_Sells * 0.5)
        exch["toxic_price"] = max(0, 100.0 - (total_sells * 0.005))

    def _update_ccp_rates(self):
        ccp = self.graph.nodes["ccp_central"]
        exch = self.graph.nodes["exchange_main"]
        
        # Formula: Current_Margin_Rate = Base_Rate + (Exchange.Market_Volatility * 0.01)
        ccp["current_margin_rate"] = ccp["base_rate"] + (exch["market_volatility"] * 0.01)
        
        # Safe Limit = Default_Fund * 10
        ccp["safe_limit"] = ccp["default_fund"] * 10
        
    def _update_bank_metrics(self, agent_id):
        bank = self.graph.nodes[agent_id]
        if bank.get("type") != "bank": return

        exch = self.graph.nodes["exchange_main"]
        ccp = self.graph.nodes["ccp_central"]
        
        # Liquid Value
        liquid_value = bank["liquid_bond_count"] * exch["liquid_price"]
        
        # Toxic Value
        toxic_value = bank["toxic_count"] * exch["toxic_price"]
        
        # Credit Exposure Value (Sum of money lent out)
        # We track this via edges: (Bank)-[CREDIT]->(Other)
        # Exposure is the amount this bank HAS LENT.
        exposure = 0.0
        # If no out_edges, result is empty.
        if agent_id in self.graph:
             for _, _, data in self.graph.out_edges(agent_id, data=True):
                if data.get("type") == "credit_exposure":
                    exposure += data.get("amount", 0.0)
        bank["credit_exposure_value"] = exposure
        
        # Total Assets = Pure_Cash + Liquid_Value + Toxic_Value + Credit_Exposure_Value
        bank["total_assets"] = bank["pure_cash"] + liquid_value + toxic_value + exposure
        
        # Net Worth
        bank["net_worth"] = bank["total_assets"] - bank["liabilities"]
        
        # Fire Sale Amount
        if bank["pure_cash"] < 0:
            bank["fire_sale_amount"] = abs(bank["pure_cash"])
        else:
            bank["fire_sale_amount"] = 0.0

    def process_intent(self, intent: AgentIntent):
        agent_id = intent.agent_id
        if agent_id.startswith("bank"):
            self._ensure_bank(agent_id)
        
        action = intent.action_type.lower() # Case insensitive
        payload = intent.payload
        
        # Nodes
        bank_node = self.graph.nodes.get(agent_id)
        ccp_node = self.graph.nodes["ccp_central"]
        exch_node = self.graph.nodes["exchange_main"]
        
        if action == "deposit_default_fund":
            # agent -> ccp
            amount = payload.get("amount", 0)
            target = payload.get("target") # "ccp_central"
            
            # Ensure bank exists and refresh reference
            self._ensure_bank(agent_id)
            bank_node = self.graph.nodes[agent_id]
            
            bank_node["pure_cash"] -= amount
            bank_node["default_fund_contribution"] += amount
            ccp_node["default_fund"] += amount
            
            # Update Edge for visualization (Bank -> CCP)
            self.graph.add_edge(agent_id, "ccp_central", type="deposit", amount=bank_node["default_fund_contribution"])
            
        elif action == "route_otc_proposal" or action == "provide_credit":
            # treat them similarly for accounting
            # Bank A -> Bank B
            target_id = payload.get("final_destination") or payload.get("target")
            
            # For OTC 'content' might be inside encrypted_content
            content = payload.get("encrypted_content", {})
            amount = payload.get("amount") or content.get("amount", 0)
            
            if target_id and amount > 0:
                self._ensure_bank(target_id)
                target_node = self.graph.nodes.get(target_id)
                self._ensure_bank(agent_id)
                bank_node = self.graph.nodes[agent_id]
                
                # Update Sender
                bank_node["pure_cash"] -= amount
                # bank_node["credit_exposure_value"] += amount # Calculated in update_metrics
                
                # Update Receiver
                target_node["pure_cash"] += amount
                target_node["liabilities"] += amount
                
                # Update/Create Edge (Sender -> Receiver) indicating Credit Exposure
                # Check if edge exists
                current_amount = 0
                if self.graph.has_edge(agent_id, target_id):
                    current_amount = self.graph[agent_id][target_id].get("amount", 0)
                
                # Overwrite or add? Usually accumulative exposure
                self.graph.add_edge(agent_id, target_id, type="credit_exposure", amount=current_amount + amount)

        elif action == "repay_interbank_loan":
            # Bank B (Borrower) -> Bank A (Lender)
            target_id = payload.get("target")
            amount = payload.get("amount", 0)
            interest = payload.get("interest", 0)
            
            if target_id:
                self._ensure_bank(target_id)
                lender_node = self.graph.nodes[target_id]
                self._ensure_bank(agent_id)
                bank_node = self.graph.nodes[agent_id]
                
                # Borrower (Sender)
                total_pay = amount + interest
                bank_node["pure_cash"] -= total_pay
                bank_node["liabilities"] -= amount
                
                # Lender (Target)
                lender_node["pure_cash"] += total_pay
                
                # reduce exposure on edge (Lender -> Borrower)
                # Note: Edge direction for usage is Lender -> Borrower (Exposure)
                # So we update edge (target_id, agent_id)
                if self.graph.has_edge(target_id, agent_id):
                    current = self.graph[target_id][agent_id].get("amount", 0)
                    self.graph.add_edge(target_id, agent_id, type="credit_exposure", amount=max(0, current - amount))

        elif action == "fire_sale_asset" or action == "sell_asset_standard":
            # Bank -> Exchange
            amount = payload.get("amount", 0)
            asset_type = payload.get("asset_type", "liquid_bond")
            
            self._ensure_bank(agent_id)
            bank_node = self.graph.nodes[agent_id]

            # Update counts
            # Assuming price logic applied per item
            if "liquid" in asset_type:
                bank_node["liquid_bond_count"] -= amount
                price = exch_node["liquid_price"]
            else:
                bank_node["toxic_count"] -= amount
                price = exch_node["toxic_price"]
            
            # Cash proceeds
            bank_node["pure_cash"] += (amount * price)
                
            # Exchange updates
            exch_node["total_sells"] += amount
            
            # Edge??
            self.graph.add_edge(agent_id, "exchange_main", type="selling", amount=amount)

        elif action == "issue_margin_call":
            # CCP -> Agent
            target_id = payload.get("target")
            req_amount = payload.get("required_amount", 0)
            
            if target_id:
                self._ensure_bank(target_id)
                self.graph.nodes[target_id]["margin_call_due"] = req_amount
                # Edge CCP -> Agent
                self.graph.add_edge("ccp_central", target_id, type="margin_call", amount=req_amount)

        elif action == "pay_margin_call":
            # Bank -> CCP
            amount = payload.get("amount", 0)
            target = payload.get("target")

            self._ensure_bank(agent_id)
            bank_node = self.graph.nodes[agent_id]

            bank_node["pure_cash"] -= amount
            bank_node["liabilities"] -= amount 
            bank_node["margin_call_due"] = max(0, bank_node.get("margin_call_due", 0) - amount)
                
            ccp_node["default_fund"] += amount
            
            self.graph.add_edge(agent_id, "ccp_central", type="margin_payment", amount=amount)

        elif action == "update_market_data":
            # Exchange -> Broadcast
            new_vol = payload.get("new_volatility")
            if new_vol is not None:
                exch_node["market_volatility"] = new_vol

        elif action == "declare_default":
            # Bank -> CCP
            self._ensure_bank(agent_id)
            bank_node = self.graph.nodes[agent_id]
            
            bank_node["status"] = "defaulted"
            bank_node["net_worth"] = 0
            bank_node["fire_sale_amount"] = 0
                
            uncovered = payload.get("uncovered_amount", 0)
            ccp_node["default_fund"] = max(0, ccp_node["default_fund"] - uncovered)


        # --- GLOBAL UPDATES ---
        self._update_exchange_prices()
        self._update_ccp_rates()
        
        # update all banks
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "bank":
                self._update_bank_metrics(n)

    def get_network_stats(self):
        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "ccp_default_fund": self.graph.nodes["ccp_central"].get("default_fund"),
            "exchange_vol": self.graph.nodes["exchange_main"].get("market_volatility"),
            "banks_stats": {}
        }
        for n, d in self.graph.nodes(data=True):
            if d.get("type") == "bank":
                stats["banks_stats"][n] = {
                    "cash": d.get("pure_cash"),
                    "liabilities": d.get("liabilities"),
                    "net_worth": d.get("net_worth")
                }
        return stats

    def get_graph_data(self):
        return nx.node_link_data(self.graph)

financial_network = FinancialNetwork()
