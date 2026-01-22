"use client";
import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const analyzeWallet = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      // Connect to your Python Backend
      const response = await axios.post("http://127.0.0.1:8000/analyze_wallet", {
        address: address,
      });
      setResult(response.data);
    } catch (err) {
      setError("Failed to fetch data. Ensure backend is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-white flex flex-col items-center p-10 font-sans">
      {/* HEADER */}
      <h1 className="text-5xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600">
        SolPhish AI
      </h1>
      <p className="text-slate-400 mb-10">Solana Scam & Phishing Detector</p>

      {/* SEARCH BOX */}
      <div className="w-full max-w-2xl flex gap-3 mb-10">
        <input
          type="text"
          placeholder="Paste Solana Wallet Address..."
          className="flex-1 p-4 rounded-xl bg-slate-900 border border-slate-700 focus:border-purple-500 outline-none transition"
          value={address}
          onChange={(e) => setAddress(e.target.value)}
        />
        <button
          onClick={analyzeWallet}
          disabled={loading || !address}
          className="px-8 py-4 bg-purple-600 hover:bg-purple-700 rounded-xl font-bold transition disabled:opacity-50"
        >
          {loading ? "Scanning..." : "Analyze"}
        </button>
      </div>

      {/* ERROR MESSAGE */}
      {error && <div className="text-red-400 mb-5">{error}</div>}

      {/* RESULTS CARD */}
      {result && (
        <div className="w-full max-w-2xl bg-slate-900 border border-slate-800 rounded-2xl p-8 shadow-2xl">
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-2xl font-semibold">Risk Analysis</h2>
            <span
              className={`px-4 py-1 rounded-full text-sm font-bold ${
                result.risk_level === "High"
                  ? "bg-red-900 text-red-200"
                  : result.risk_level === "Medium"
                  ? "bg-yellow-900 text-yellow-200"
                  : "bg-green-900 text-green-200"
              }`}
            >
              {result.risk_level} RISK
            </span>
          </div>

          <div className="grid grid-cols-2 gap-6 mb-6">
            <div className="bg-slate-950 p-5 rounded-xl border border-slate-800">
              <p className="text-slate-500 text-sm mb-1">Risk Score</p>
              <p className="text-4xl font-bold">{result.risk_score}/100</p>
            </div>
            <div className="bg-slate-950 p-5 rounded-xl border border-slate-800">
              <p className="text-slate-500 text-sm mb-1">AI Probability</p>
              <p className="text-4xl font-bold text-purple-400">
                {result.details.ai_probability}
              </p>
            </div>
          </div>

          <div className="space-y-3">
            <div className="flex justify-between p-3 bg-slate-950 rounded-lg">
              <span className="text-slate-400">Wallet Balance</span>
              <span className="font-mono">{result.details.balance}</span>
            </div>
            
            {result.details.reasons.length > 0 ? (
              <div className="mt-4">
                <p className="text-red-400 text-sm mb-2">Flagged Factors:</p>
                <ul className="list-disc list-inside text-slate-300 space-y-1">
                  {result.details.reasons.map((reason, i) => (
                    <li key={i}>{reason}</li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="mt-4 p-3 bg-green-900/20 border border-green-900 rounded-lg text-green-400 text-center text-sm">
                ✅ No suspicious rule violations found.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}