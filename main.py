import logging
import random
import uvicorn
import numpy as np
import time
from fastapi import FastAPI
from pydantic import BaseModel

#api
from fastapi.middleware.cors import CORSMiddleware

# Solana Libraries
from solana.rpc.api import Client
from solders.pubkey import Pubkey # type: ignore
from solana.rpc.types import TokenAccountOpts
from solana.rpc.api import Client
from solders.pubkey import Pubkey

# ML Libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Simple in-memory cache: { "address": { "data": result, "timestamp": 12345 } }
CACHE = {}
CACHE_DURATION = 300 # 5 Minutes

# Known Safe Addresses (Exchanges, Bridges)
WHITELIST = [
    "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1", # Example: Raydium Authority
    "mvines9iiHiQTysrwkJjGf2gb9Ex9jXJX8ns3qwf2kN", # Example: Devnet Faucet
    # You can add Binance/Coinbase hot wallets here if you find them
]

# --- CONFIGURATION ---
SOLANA_RPC_URL = "https://api.mainnet-beta.solana.com"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SolanaRiskDetector")

# Official Mainnet Mint Addresses
USDC_MINT = Pubkey.from_string("EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")
USDT_MINT = Pubkey.from_string("Es9vMFrzcCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB")

# --- DATA STRUCTURES ---

class AnalysisRequest(BaseModel):
    address: str

class AnalysisResponse(BaseModel):
    address: str
    risk_score: int
    risk_level: str
    details: dict

# --- 1. DUMMY DATA GENERATOR & ML MODEL ---

class RiskModel:
    """
    Handles the AI portion of the risk detection.
    Since we don't have a massive labeled dataset of real scams for Devnet,
    we train a model on synthetic data every time the server starts.
    """
    def __init__(self):
        self.model = None
        self.is_trained = False

    def generate_synthetic_data(self, n_samples=1000):
        """
        Generates dummy wallet features for training.
        """
        logger.info("Generating synthetic training data...")
        
        data = []
        labels = [] # 0 = Safe, 1 = Scam

        for _ in range(n_samples):
            is_scam = random.choice([True, False])
            
            if is_scam:
                # Scams often have: Low age, high fail ratio, specific tx patterns
                age = random.randint(0, 30) 
                tx_count = random.randint(0, 50)
                avg_val = random.uniform(0.1, 100.0)
                fail_ratio = random.uniform(0.3, 0.9) 
                suspicious = random.randint(1, 10)
                labels.append(1)
            else:
                # Legit users: Older wallets, steady tx count, low fail ratio
                age = random.randint(60, 1000)
                tx_count = random.randint(50, 5000)
                avg_val = random.uniform(0.01, 5.0)
                fail_ratio = random.uniform(0.0, 0.2)
                suspicious = 0
                labels.append(0)

            data.append([age, tx_count, avg_val, fail_ratio, suspicious])

        return np.array(data), np.array(labels)

    def train(self):
        """Trains a RandomForestClassifier on the synthetic data."""
        X, y = self.generate_synthetic_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        score = self.model.score(X_test, y_test)
        self.is_trained = True
        logger.info(f"Model trained successfully. Accuracy: {score:.2f}")

    def predict_proba(self, features):
        """Returns the probability (0.0 to 1.0) of being a scam."""
        if not self.is_trained:
            raise Exception("Model not trained yet.")
        return self.model.predict_proba([features])[0][1]

# Initialize and train model globally
risk_ai = RiskModel()

# --- 2. SOLANA BLOCKCHAIN SERVICE ---

class SolanaService:
    def __init__(self):
        self.client = Client(SOLANA_RPC_URL)
        logger.info(f"Connected to Solana RPC: {SOLANA_RPC_URL}")

    def get_token_balance(self, owner_pubkey: Pubkey, mint_pubkey: Pubkey):
        """Helper to fetch specific token balance (USDC/USDT)"""
        try:
            # 1. Find the Associated Token Account for this wallet & mint
            opts = TokenAccountOpts(mint=mint_pubkey)
            response = self.client.get_token_accounts_by_owner(owner_pubkey, opts)
            
            # 2. If they have an account, get the balance
            if response.value:
                token_account = response.value[0].pubkey
                balance_resp = self.client.get_token_account_balance(token_account)
                return balance_resp.value.ui_amount if balance_resp.value else 0.0
            return 0.0
        except Exception:
            # If RPC fails or rate limits, assume 0 to keep app running
            return 0.0

    def get_wallet_features(self, address: str):
        pubkey = Pubkey.from_string(address)
        
        # 1. Get SOL Balance
        balance_resp = self.client.get_balance(pubkey)
        lamports = balance_resp.value if balance_resp.value else 0
        sol_balance = lamports / 1e9

        # 2. Get Stablecoin Balances (NEW!)
        usdc_balance = self.get_token_balance(pubkey, USDC_MINT)
        usdt_balance = self.get_token_balance(pubkey, USDT_MINT)
        
        # 3. Get Transactions (History)
        sigs_resp = self.client.get_signatures_for_address(pubkey, limit=20)
        signatures = sigs_resp.value if sigs_resp.value else []
        tx_count_estimate = len(signatures)
        
        # 4. Calculate Failure Ratio
        failed_txs = 0
        for tx in signatures:
            if tx.err is not None:
                failed_txs += 1
        real_fail_ratio = (failed_txs / tx_count_estimate) if tx_count_estimate > 0 else 0.0

        # Features for AI
        simulated_age_days = tx_count_estimate * 2 
        
        # Use TOTAL Value (SOL + Stablecoins) as the feature proxy for "wealth"
        total_value_proxy = sol_balance + (usdc_balance / 150) + (usdt_balance / 150) # Normalize price roughly

        features = [simulated_age_days, tx_count_estimate * 10, total_value_proxy, real_fail_ratio, 0]
        
        return {
            "features": features,
            "raw_data": {
                "sol_balance": sol_balance,
                "usdc_balance": usdc_balance,  # Send to logic
                "usdt_balance": usdt_balance,  # Send to logic
                "recent_tx_count": tx_count_estimate
            }
        }

solana_service = SolanaService()

# --- 3. HYBRID RISK ENGINE ---

def calculate_hybrid_risk(address: str, is_program=False):
    """
    Combines Rule-Based Logic + AI Prediction.
    """
    
    # 1. Whitelist Check (Instant Safe)
    if address in WHITELIST:
        return {
            "score": 0,
            "level": "Safe",
            "breakdown": {
                "info": "This is a known verified address (Whitelist).",
                "balance": "N/A"
            }
        }
        
    try:
        # Step A: Fetch Data
        data = solana_service.get_wallet_features(address)
        features = data["features"]
        raw = data["raw_data"]
        
        # Step B: AI Prediction (0.0 to 1.0)
        ai_risk_prob = risk_ai.predict_proba(features)
        ai_score = ai_risk_prob * 100

        # Step C: Rule-Based Adjustments
        rule_score = 0
        reasons = []

        # --- Rule 1: "Empty Wallet" Check (Upgraded) ---
        # We only flag if SOL is low AND Stablecoins are low
        total_stablecoins = raw.get("usdc_balance", 0) + raw.get("usdt_balance", 0)
        
        if raw["sol_balance"] < 0.01 and total_stablecoins < 1.0:
            rule_score += 20
            reasons.append("Wallet is empty (Low SOL & No Stablecoins).")
        elif raw["sol_balance"] < 0.01 and total_stablecoins > 10.0:
            # Good Sign: Low SOL but high Stablecoins -> Legit User
            rule_score -= 10 
            reasons.append("Trust Bonus: Holds Stablecoins.")
        
        if raw["recent_tx_count"] == 0:
            rule_score += 30
            reasons.append("No recent transaction history.")
        elif raw["recent_tx_count"] < 3:
            rule_score += 10
            reasons.append("Low transaction history.")

        if is_program:
            rule_score += 15 
            reasons.append("Program unverified (simulated).")

        # Step D: Weighted Average (AI 60%, Rules 40%)
        final_score = (ai_score * 0.6) + (rule_score * 0.4)
        final_score = min(100, max(0, final_score))
        
        if final_score < 30: level = "Low"
        elif final_score < 70: level = "Medium"
        else: level = "High"

        return {
            "score": int(final_score),
            "level": level,
            "breakdown": {
                "ai_probability": f"{ai_score:.2f}%",
                "rule_penalty": rule_score,
                "reasons": reasons,
                # UPDATE THIS LINE TO INCLUDE USDT:
                "balance": f"{raw['sol_balance']:.3f} SOL | {raw.get('usdc_balance', 0):.2f} USDC | {raw.get('usdt_balance', 0):.2f} USDT"
            }
        }

    except Exception as e:
        logger.error(f"Error analyzing {address}: {str(e)}")
        return {
            "score": 0,
            "level": "Unknown",
            "breakdown": {"error": str(e)}
        }

def get_wallet_features(self, address: str):
        pubkey = Pubkey.from_string(address)
        
        # 1. Get Balance
        balance_resp = self.client.get_balance(pubkey)
        lamports = balance_resp.value if balance_resp.value else 0
        sol_balance = lamports / 1e9

        # 2. Get Transactions (Fetch more to be accurate, e.g., 20)
        sigs_resp = self.client.get_signatures_for_address(pubkey, limit=20)
        signatures = sigs_resp.value if sigs_resp.value else []
        
        tx_count_estimate = len(signatures)
        
        # --- NEW LOGIC: Calculate REAL Failure Ratio ---
        failed_txs = 0
        for tx in signatures:
            if tx.err is not None:  # If 'err' exists, the tx failed
                failed_txs += 1
        
        # Avoid division by zero
        real_fail_ratio = (failed_txs / tx_count_estimate) if tx_count_estimate > 0 else 0.0
        # -----------------------------------------------

        # 3. Time-based metrics (Simple estimation)
        # If we have 20 txs, we assume it's somewhat active.
        simulated_age_days = tx_count_estimate * 2 
        
        suspicious_interactions = 0

        # Update features with REAL fail ratio
        features = [simulated_age_days, tx_count_estimate * 10, sol_balance, real_fail_ratio, suspicious_interactions]
        
        return {
            "features": features,
            "raw_data": {
                "sol_balance": sol_balance,
                "recent_tx_count": tx_count_estimate,
                "fail_ratio": real_fail_ratio # Pass this to the frontend details
            }
        }

# --- 4. FASTAPI APP ---

app = FastAPI(title="Solana Web3 Risk Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (good for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    risk_ai.train()

@app.get("/")
def home():
    return {"message": "Solana Risk Detector API is running."}

@app.post("/analyze_wallet", response_model=AnalysisResponse)
async def analyze_wallet(request: AnalysisRequest):
    current_time = time.time()
    
    # CHECK CACHE
    if request.address in CACHE:
        cached_entry = CACHE[request.address]
        if current_time - cached_entry["timestamp"] < CACHE_DURATION:
            return AnalysisResponse(
                address=request.address,
                risk_score=cached_entry["data"]["score"],
                risk_level=cached_entry["data"]["level"],
                details=cached_entry["data"]["breakdown"]
            )

    # If not in cache, calculate it
    result = calculate_hybrid_risk(request.address, is_program=False)
    
    # SAVE TO CACHE
    CACHE[request.address] = {
        "data": result,
        "timestamp": current_time
    }
    
    return AnalysisResponse(
        address=request.address,
        risk_score=result["score"],
        risk_level=result["level"],
        details=result["breakdown"]
    )
@app.post("/analyze_program", response_model=AnalysisResponse)
async def analyze_program(request: AnalysisRequest):
    result = calculate_hybrid_risk(request.address, is_program=True)
    return AnalysisResponse(
        address=request.address,
        risk_score=result["score"],
        risk_level=result["level"],
        details=result["breakdown"]
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)