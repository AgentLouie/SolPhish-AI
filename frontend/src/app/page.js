"use client";
import { useState } from "react";
import axios from "axios";
import { Shield, AlertTriangle, CheckCircle, Search, Activity, Lock } from "lucide-react";

export default function Home() {
  const [address, setAddress] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const analyzeWallet = async () => {
    if (!address) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze_wallet", {
        address: address,
      });
      setResult(response.data);
    } catch (err) {
      setError("Server offline or invalid address. Make sure Backend is running.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Helper to determine color based on risk level
  const getRiskColor = (level) => {
    if (level === "High") return "text-red-400 border-red-500/50 bg-red-900/20";
    if (level === "Medium") return "text-yellow-400 border-yellow-500/50 bg-yellow-900/20";
    return "text-green-400 border-green-500/50 bg-green-900/20";
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-white font-sans selection:bg-purple-500/30">
      
      {/* BACKGROUND GLOW EFFECTS */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute top-[-10%] left-[20%] w-[500px] h-[500px] bg-purple-600/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-[-10%] right-[20%] w-[500px] h-[500px] bg-blue-600/10 rounded-full blur-[120px]" />
      </div>

      {/* NAVBAR */}
      <nav className="w-full p-6 flex justify-center border-b border-white/5 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-4xl w-full flex items-center gap-3">
          <Shield className="w-8 h-8 text-purple-500" />
          <h1 className="text-xl font-bold tracking-wider">
            SOLPHISH <span className="text-purple-500">AI</span>
          </h1>
        </div>
      </nav>

      <main className="max-w-4xl mx-auto p-6 mt-10 flex flex-col items-center">
        
        {/* HERO SECTION */}
        <div className="text-center mb-12">
          <h1 className="text-5xl md:text-6xl font-extrabold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-white via-slate-300 to-slate-500">
            Analyze Any Wallet.
          </h1>
          <p className="text-slate-400 text-lg">
            AI-powered fraud detection for the Solana Blockchain.
          </p>
        </div>

        {/* SEARCH BAR */}
        <div className="w-full max-w-2xl relative mb-12 group">
          <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-600 to-blue-600 rounded-2xl blur opacity-30 group-hover:opacity-75 transition duration-500"></div>
          <div className="relative flex bg-black rounded-2xl p-2 items-center border border-white/10">
            <Search className="w-6 h-6 text-slate-500 ml-4" />
            <input
              type="text"
              placeholder="Paste Solana Address (e.g. 5YNm...)"
              className="w-full bg-transparent border-none outline-none text-white px-4 py-3 text-lg placeholder:text-slate-600"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && analyzeWallet()}
            />
            <button
              onClick={analyzeWallet}
              disabled={loading || !address}
              className="bg-purple-600 hover:bg-purple-500 text-white px-8 py-3 rounded-xl font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-900/20"
            >
              {loading ? (
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              ) : (
                "Scan"
              )}
            </button>
          </div>
        </div>

        {/* ERROR MESSAGE */}
        {error && (
          <div className="w-full max-w-2xl p-4 mb-6 bg-red-900/20 border border-red-500/50 rounded-xl flex items-center gap-3 text-red-200 animate-fadeIn">
            <AlertTriangle className="w-5 h-5 text-red-500" />
            {error}
          </div>
        )}

        {/* RESULTS DASHBOARD */}
        {result && (
          <div className="w-full max-w-2xl animate-slideUp">
            
            {/* STATUS HEADER */}
            <div className={`p-1 rounded-t-3xl bg-gradient-to-r ${result.risk_level === 'High' ? 'from-red-600 to-orange-600' : result.risk_level === 'Medium' ? 'from-yellow-500 to-orange-500' : 'from-green-500 to-emerald-500'}`}></div>
            <div className="bg-[#111] border border-white/5 rounded-b-3xl p-8 shadow-2xl relative overflow-hidden">
              
              {/* Header Row */}
              <div className="flex justify-between items-start mb-8">
                <div>
                  <p className="text-slate-400 text-sm font-mono mb-1">TARGET WALLET</p>
                  <p className="font-mono text-white/80 text-sm break-all">{result.address}</p>
                </div>
                <div className={`px-4 py-2 rounded-lg border font-bold flex items-center gap-2 ${getRiskColor(result.risk_level)}`}>
                  {result.risk_level === 'Safe' || result.risk_level === 'Low' ? <CheckCircle className="w-4 h-4"/> : <AlertTriangle className="w-4 h-4"/>}
                  {result.risk_level.toUpperCase()}
                </div>
              </div>

              {/* Main Metrics Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
                
                {/* Risk Score Card */}
                <div className="bg-black/40 p-6 rounded-2xl border border-white/5 relative group">
                  <div className="flex justify-between items-center mb-2">
                    <h3 className="text-slate-400 text-sm font-medium">Risk Score</h3>
                    <Activity className="w-4 h-4 text-purple-500" />
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className={`text-4xl font-bold ${result.risk_score > 50 ? 'text-red-400' : 'text-white'}`}>
                      {result.risk_score}
                    </span>
                    <span className="text-slate-600">/ 100</span>
                  </div>
                  
                  {/* Progress Bar */}
                  <div className="w-full h-2 bg-slate-800 rounded-full mt-4 overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-1000 ${result.risk_score > 70 ? 'bg-red-500' : result.risk_score > 30 ? 'bg-yellow-500' : 'bg-green-500'}`}
                      style={{ width: `${result.risk_score}%` }}
                    />
                  </div>
                </div>

                {/* AI Probability Card */}
                <div className="bg-black/40 p-6 rounded-2xl border border-white/5">
                  <div className="flex justify-between items-center mb-2">
                    <h3 className="text-slate-400 text-sm font-medium">AI Confidence</h3>
                    <Lock className="w-4 h-4 text-blue-500" />
                  </div>
                  <div className="flex items-baseline gap-2">
                    <span className="text-4xl font-bold text-blue-400">
                      {result.details.ai_probability}
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">Machine Learning Prediction</p>
                </div>
              </div>

              {/* Breakdown Section */}
              <div className="space-y-4">
                <div className="bg-white/5 p-4 rounded-xl flex justify-between items-center">
                  <span className="text-slate-400 text-sm">Total Assets</span>
                  <span className="font-mono text-sm">{result.details.balance}</span>
                </div>

                <div className="bg-black/20 p-5 rounded-xl border border-white/5">
                  <p className="text-slate-400 text-sm mb-3">Risk Factors</p>
                  {result.details.reasons.length > 0 ? (
                    <ul className="space-y-2">
                      {result.details.reasons.map((reason, i) => (
                        <li key={i} className="flex items-start gap-2 text-sm text-red-300">
                          <span className="text-red-500 mt-0.5">•</span>
                          {reason}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <div className="flex items-center gap-2 text-green-400 text-sm">
                      <CheckCircle className="w-4 h-4" />
                      <span>No suspicious activity detected.</span>
                    </div>
                  )}
                </div>
              </div>

            </div>
          </div>
        )}
      </main>
    </div>
  );
}