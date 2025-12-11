import React, { useState, useRef, useEffect } from 'react';
import { GoogleGenAI, Type } from "@google/genai";
import { ChatMessage, LogEntry } from '../types';
import { BrainCircuit, Send, Loader2, MapPin, ShieldAlert, Globe, Crosshair, BarChart2, Users, Radio, Terminal, CornerDownRight } from 'lucide-react';
import DataVisualizer from './DataVisualizer';

interface IntelChatProps {
  addLog: (source: string, message: string, type?: LogEntry['type']) => void;
}

const IntelChat: React.FC<IntelChatProps> = ({ addLog }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<'flash' | 'web_ops' | 'geo_int' | 'strategy' | 'analytics'>('flash');
  
  // Collaborative Identity
  const [identity] = useState(() => `OP-${Math.floor(Math.random() * 9000) + 1000}`);
  const channelRef = useRef<BroadcastChannel | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Initialize Collaborative Mesh
  useEffect(() => {
    const channel = new BroadcastChannel('meli_mesh_network');
    channelRef.current = channel;

    channel.onmessage = (event) => {
      const { type, payload } = event.data;
      if (type === 'SYNC_MESSAGE') {
        setMessages(prev => {
          // Avoid duplicates
          if (prev.some(m => m.id === payload.id)) return prev;
          return [...prev, payload];
        });
      } else if (type === 'SYSTEM_COMMAND' && payload.target === 'INTEL') {
          // Handle Remote Search Command
          if (payload.action === 'search' && payload.query) {
              setMode('web_ops');
              setInput(payload.query);
              // Slight delay to allow state update before execute
              setTimeout(() => {
                   executeCommand(payload.query, 'web_ops');
              }, 100);
          }
      }
    };

    addLog('NET', `Joined mesh network as ${identity}`, 'success');

    return () => {
      channel.close();
    };
  }, [identity]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const broadcastMessage = (msg: ChatMessage) => {
    channelRef.current?.postMessage({ type: 'SYNC_MESSAGE', payload: msg });
  };

  const executeCommand = async (textInput: string, currentMode: string) => {
    if (!textInput.trim()) return;
    
    const userMsg: ChatMessage = {
      id: Date.now().toString() + Math.random(),
      role: 'user',
      content: textInput,
      timestamp: Date.now(),
      authorId: identity
    };

    // Update local and broadcast
    setMessages(prev => [...prev, userMsg]);
    broadcastMessage(userMsg);
    
    setInput('');
    setLoading(true);

    try {
      // Check for API key selection for environments that require it
      const win = window as any;
      if (win.aistudio && await win.aistudio.hasSelectedApiKey() === false) {
           addLog('INTEL', 'Security Key required. Requesting access...', 'warning');
           await win.aistudio.openSelectKey();
      }

      const apiKey = process.env.API_KEY;
      if (!apiKey) {
          throw new Error("System Identity Token (API_KEY) missing from environment.");
      }

      const ai = new GoogleGenAI({ apiKey });
      let response;
      let modelId = 'gemini-2.5-flash'; 

      addLog('INTEL', `Executing directive: [${currentMode.toUpperCase()}]`, 'info');

      const baseSystemInstruction = "You are MELI, a high-performance intelligence interface. Respond with precision, using tactical terminology. Format responses for quick scanning. Prioritize actionable intelligence.";

      if (currentMode === 'strategy') {
        // Thinking Mode - Complex Reasoning
        modelId = 'gemini-3-pro-preview';
        response = await ai.models.generateContent({
            model: modelId,
            contents: userMsg.content,
            config: {
                thinkingConfig: { thinkingBudget: 32768 },
                systemInstruction: baseSystemInstruction + " Engage strategic reasoning. Analyze second-order effects."
            }
        });
      } else if (currentMode === 'web_ops') {
        // Search Grounding
        modelId = 'gemini-2.5-flash';
        response = await ai.models.generateContent({
            model: modelId,
            contents: userMsg.content,
            config: {
                tools: [{ googleSearch: {} }],
                systemInstruction: baseSystemInstruction + " Prioritize Open Source Intelligence (OSINT). Provide sources."
            }
        });
      } else if (currentMode === 'geo_int') {
         // Maps Grounding
         modelId = 'gemini-2.5-flash';
         let location = { latitude: 37.7749, longitude: -122.4194 }; 
         try {
             const pos = await new Promise<GeolocationPosition>((resolve, reject) => 
                 navigator.geolocation.getCurrentPosition(resolve, reject, {timeout: 5000})
             );
             location = { latitude: pos.coords.latitude, longitude: pos.coords.longitude };
         } catch (e) {
             addLog('INTEL', 'Geo-lock failed, using triangulation default.', 'warning');
         }

         response = await ai.models.generateContent({
             model: modelId,
             contents: userMsg.content,
             config: {
                 tools: [{ googleMaps: {} }],
                 toolConfig: {
                    retrievalConfig: {
                        latLng: location
                    }
                 },
                 systemInstruction: baseSystemInstruction + " Provide precise geospatial coordinates and location data."
             }
         });
      } else if (currentMode === 'analytics') {
          // Analytics Mode - JSON Output
          modelId = 'gemini-2.5-flash';
          response = await ai.models.generateContent({
              model: modelId,
              contents: userMsg.content,
              config: {
                  systemInstruction: baseSystemInstruction + " Return strictly JSON data matching the schema for visualization.",
                  responseMimeType: "application/json",
                  responseSchema: {
                      type: Type.OBJECT,
                      properties: {
                          summary: { type: Type.STRING, description: "A brief textual summary of the analysis." },
                          chartTitle: { type: Type.STRING, description: "Title for the chart." },
                          chartType: { type: Type.STRING, enum: ["line", "bar", "area"] },
                          data: {
                              type: Type.ARRAY,
                              items: {
                                  type: Type.OBJECT,
                                  properties: {
                                      label: { type: Type.STRING },
                                      value: { type: Type.NUMBER }
                                  }
                              }
                          }
                      }
                  }
              }
          });
      } else {
         // Standard Low Latency
         modelId = 'gemini-2.5-flash-lite-latest'; 
         response = await ai.models.generateContent({
             model: modelId,
             contents: userMsg.content,
             config: {
                 systemInstruction: baseSystemInstruction
             }
         });
      }

      let text = response.text || "NO DATA PACKET RECEIVED.";
      let analyticsData = null;
      const groundingMetadata = response.candidates?.[0]?.groundingMetadata;

      // Parse JSON for analytics mode
      if (currentMode === 'analytics' && response.text) {
          try {
              analyticsData = JSON.parse(response.text);
              text = analyticsData.summary; // Use summary as the chat text
          } catch (e) {
              console.error("Failed to parse JSON", e);
              text = "DATA CORRUPTION: UNABLE TO PARSE ANALYTICS STREAM.";
          }
      }

      const aiMsg: ChatMessage = {
          id: (Date.now() + 1).toString(),
          role: 'model',
          content: text,
          timestamp: Date.now(),
          groundingMetadata: groundingMetadata,
          thinking: currentMode === 'strategy',
          analytics: analyticsData
      };

      setMessages(prev => [...prev, aiMsg]);
      broadcastMessage(aiMsg);
      addLog('INTEL', 'Data packet received and decrypted.', 'success');

    } catch (err: any) {
        console.error(err);
        addLog('INTEL', `Operation failed: ${err.message}`, 'error');
        const errorMsg: ChatMessage = {
            id: Date.now().toString(),
            role: 'model',
            content: `CRITICAL ERROR: ${err.message}`,
            timestamp: Date.now()
        };
        setMessages(prev => [...prev, errorMsg]);
    } finally {
        setLoading(false);
    }
  };

  const handleSend = () => {
      executeCommand(input, mode);
  };

  const renderGrounding = (metadata: any) => {
    if (!metadata) return null;
    const chunks = metadata.groundingChunks || [];
    const searchChunks = chunks.filter((c: any) => c.web);
    const mapChunks = chunks.filter((c: any) => c.maps);

    if (searchChunks.length === 0 && mapChunks.length === 0) return null;

    return (
        <div className="mt-4 border-l-2 border-ops-accent pl-4 py-2 bg-ops-accent/5">
            <div className="text-ops-accent mb-2 text-[10px] font-mono tracking-widest font-bold flex items-center gap-2">
                <ShieldAlert size={12} /> VERIFIED SOURCES
            </div>
            <div className="grid gap-1">
                {searchChunks.map((chunk: any, i: number) => (
                    <a key={i} href={chunk.web?.uri} target="_blank" rel="noopener noreferrer" 
                       className="flex items-center gap-2 text-ops-text-dim hover:text-white truncate hover:underline text-xs font-mono">
                        <span className="text-ops-500">[{i+1}]</span>
                        {chunk.web?.title || chunk.web?.uri}
                    </a>
                ))}
                 {mapChunks.map((chunk: any, i: number) => (
                    <a key={i} href={chunk.maps?.uri} target="_blank" rel="noopener noreferrer" 
                       className="flex items-center gap-2 text-green-400 hover:text-green-300 truncate hover:underline text-xs font-mono">
                        <MapPin size={12} />
                        {chunk.maps?.title || "COORDINATES_LOCKED"}
                    </a>
                ))}
            </div>
        </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-transparent">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-ops-800 bg-black/60 backdrop-blur z-10">
            <h2 className="text-lg font-bold font-mono text-white tracking-widest flex items-center gap-2">
                <Terminal size={18} className="text-ops-accent" />
                COMMAND_LINE
            </h2>
            <div className="flex gap-2 items-center">
                <div className="flex gap-1">
                    {['flash', 'web_ops', 'geo_int', 'analytics', 'strategy'].map((m) => (
                        <button 
                            key={m}
                            onClick={() => setMode(m as any)}
                            className={`px-3 py-1 rounded-none text-[9px] font-bold font-mono tracking-widest uppercase border transition-all ${
                                mode === m 
                                ? 'bg-ops-accent text-black border-ops-accent' 
                                : 'bg-transparent border-ops-800 text-ops-text-dim hover:border-ops-accent hover:text-white'
                            }`}>
                            {m.replace('_', ' ')}
                        </button>
                    ))}
                </div>
            </div>
        </div>

        {/* Chat Area - Terminal Style */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-6 space-y-6 bg-black/20">
            {messages.length === 0 && (
                <div className="h-full flex flex-col items-center justify-center text-ops-text-dim opacity-30 select-none">
                     <Crosshair size={64} className="text-ops-800 mb-4 animate-spin-slow" />
                     <p className="font-mono text-xs tracking-[0.3em]">AWAITING OPERATOR INPUT</p>
                </div>
            )}
            
            {messages.map((msg) => {
                const isMe = msg.authorId === identity;
                const isSystem = msg.role === 'model';

                return (
                <div key={msg.id} className="w-full font-mono group animate-in fade-in slide-in-from-bottom-2 duration-300">
                    {/* Message Header */}
                    <div className="flex items-center gap-2 text-[10px] tracking-widest mb-1 opacity-60">
                         <span className={isSystem ? 'text-ops-accent' : 'text-ops-500'}>
                             {isSystem ? 'MELI::SYSTEM_RESPONSE' : `OP::${msg.authorId || 'UNKNOWN'}`}
                         </span>
                         <span className="text-ops-800">|</span>
                         <span className="text-ops-text-dim">{new Date(msg.timestamp).toLocaleTimeString()}</span>
                         {msg.thinking && <span className="ml-auto text-purple-500 animate-pulse">DEEP_THOUGHT_PROCESS</span>}
                    </div>

                    {/* Message Content Body */}
                    <div className={`relative pl-4 border-l-2 ${
                        isSystem ? 'border-ops-accent' : 'border-ops-500'
                    } py-1`}>
                        <div className={`text-sm leading-relaxed whitespace-pre-wrap ${
                            isSystem ? 'text-white' : 'text-ops-400'
                        }`}>
                            {msg.content}
                        </div>
                        
                        {/* Interactive Elements */}
                        {msg.analytics && (
                             <div className="mt-4 p-1 border border-ops-800 bg-black/40">
                                <DataVisualizer data={msg.analytics} />
                             </div>
                        )}
                        {renderGrounding(msg.groundingMetadata)}
                    </div>
                </div>
            )})}
            
            {loading && (
                <div className="flex items-center gap-3 pl-4 border-l-2 border-ops-accent animate-pulse">
                     <Loader2 className="animate-spin text-ops-accent" size={14} />
                     <span className="font-mono text-xs text-ops-accent tracking-widest">PROCESSING DATA STREAM...</span>
                </div>
            )}
        </div>

        {/* Input - Terminal Line */}
        <div className="p-4 bg-black border-t border-ops-800 z-10">
            <div className="flex items-center gap-2 bg-ops-900/50 border border-ops-800 px-4 py-3 focus-within:border-ops-accent transition-colors">
                <span className="text-ops-accent font-mono text-sm animate-pulse">❯</span>
                <input 
                    type="text" 
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                    placeholder="ENTER COMMAND..."
                    className="flex-1 bg-transparent border-none text-white font-mono text-sm focus:outline-none placeholder:text-ops-800"
                    autoFocus
                />
                <button 
                    onClick={handleSend}
                    disabled={loading || !input.trim()}
                    className="text-ops-text-dim hover:text-ops-accent disabled:opacity-30"
                >
                    <CornerDownRight size={16} />
                </button>
            </div>
        </div>
    </div>
  );
};

export default IntelChat;