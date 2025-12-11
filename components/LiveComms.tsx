import React, { useEffect, useRef, useState } from 'react';
import { GoogleGenAI, LiveServerMessage, Modality, FunctionDeclaration, Type } from "@google/genai";
import { LogEntry } from '../types';
import { Mic, MicOff, Zap, Volume2, Activity, Disc, Square, Radio, Globe, Camera, ScanFace, Target, AlertTriangle, User, Search, Eye } from 'lucide-react';
import { createPcmBlob, decodeAudioData, decodeAudio, encodeWAV, blobToBase64 } from '../services/geminiUtils';

interface LiveCommsProps {
    addLog: (source: string, message: string, type?: LogEntry['type']) => void;
}

type SignalState = 'OFFLINE' | 'SEARCHING' | 'WEAK' | 'STABLE' | 'OPTIMAL';

interface Particle {
    x: number;
    y: number;
    vx: number;
    vy: number;
    life: number;
    maxLife: number;
    size: number;
    color: string;
    decay: number;
}

interface Orbital {
    angle: number;
    radius: number;
    speed: number;
    size: number;
    opacity: number;
}

interface BiometricData {
    identified: boolean;
    name?: string;
    id?: string;
    age?: string;
    origin?: string;
    occupation?: string;
    threat?: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    notes?: string;
    timestamp?: number;
    sources?: string[];
    affiliations?: string[];
    globalClass?: string;
}

// Tool Definitions for Full Sync Control
const toolDeclarations: FunctionDeclaration[] = [
    {
        name: "change_view",
        description: "Navigate the user to a different module interface.",
        parameters: {
            type: Type.OBJECT,
            properties: {
                view: { 
                    type: Type.STRING, 
                    enum: ["DASHBOARD", "INTEL", "VISUAL_OPS", "MEDIA_LAB", "LIVE_COMMS"],
                    description: "The target module to navigate to."
                }
            },
            required: ["view"]
        }
    },
    {
        name: "execute_visual_ops",
        description: "Generate, edit, or analyze images in Visual Ops module. Uses Google Search for grounding if needed.",
        parameters: {
            type: Type.OBJECT,
            properties: {
                action: { type: Type.STRING, enum: ["generate", "analyze", "edit"] },
                prompt: { type: Type.STRING, description: "Description of the visual to generate or instructions for edit." }
            },
            required: ["action", "prompt"]
        }
    },
    {
        name: "execute_media_lab",
        description: "Generate video or audio content in Media Lab module.",
        parameters: {
            type: Type.OBJECT,
            properties: {
                type: { type: Type.STRING, enum: ["video", "audio"] },
                prompt: { type: Type.STRING, description: "Description for video generation or text for audio synthesis." }
            },
            required: ["type", "prompt"]
        }
    },
    {
        name: "search_intel",
        description: "Perform a Google Search to fetch real-time data, news, or world-wide information.",
        parameters: {
            type: Type.OBJECT,
            properties: {
                query: { type: Type.STRING, description: "The search query to execute on the web." }
            },
            required: ["query"]
        }
    }
];

const LiveComms: React.FC<LiveCommsProps> = ({ addLog }) => {
    const [connected, setConnected] = useState(false);
    const [mediaActive, setMediaActive] = useState(false);
    const [videoEnabled, setVideoEnabled] = useState(false);
    const [signalQuality, setSignalQuality] = useState<SignalState>('OFFLINE');
    const [isRecording, setIsRecording] = useState(false);
    const [isTransmittingCmd, setIsTransmittingCmd] = useState(false);
    
    // Biometrics
    const [isScanning, setIsScanning] = useState(false);
    const [autoIdMode, setAutoIdMode] = useState(false);
    const [biometricData, setBiometricData] = useState<BiometricData | null>(null);

    // Refs for cleanup and stable callbacks
    const sessionPromiseRef = useRef<Promise<any> | null>(null);
    const inputContextRef = useRef<AudioContext | null>(null);
    const outputContextRef = useRef<AudioContext | null>(null);
    const nextStartTimeRef = useRef<number>(0);
    const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());
    const streamRef = useRef<MediaStream | null>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationFrameRef = useRef<number>(0);
    
    // Recording & Integration Refs
    const broadcastChannelRef = useRef<BroadcastChannel | null>(null);
    const transcriptBufferRef = useRef<{user: string, model: string}>({user: '', model: ''});

    // WAV Recording Refs
    const recordingMixerRef = useRef<GainNode | null>(null);
    const captureProcessorRef = useRef<ScriptProcessorNode | null>(null);
    const pcmChunksRef = useRef<Float32Array[]>([]);
    const aiGainRef = useRef<GainNode | null>(null);

    // Analyzer Refs for Meters
    const inputAnalyserRef = useRef<AnalyserNode | null>(null);
    const outputAnalyserRef = useRef<AnalyserNode | null>(null);
    
    // DOM Refs for direct style updates (High Performance)
    const inputMeterRef = useRef<HTMLDivElement>(null);
    const outputMeterRef = useRef<HTMLDivElement>(null);

    // Visualizer State
    const particlesRef = useRef<Particle[]>([]);
    const orbitalsRef = useRef<Orbital[]>([]);

    // Initialize Broadcast Channel
    useEffect(() => {
        broadcastChannelRef.current = new BroadcastChannel('meli_mesh_network');
        return () => {
            broadcastChannelRef.current?.close();
        };
    }, []);

    const broadcastToCore = (role: 'user' | 'model', content: string) => {
        if (!content.trim()) return;
        broadcastChannelRef.current?.postMessage({
            type: 'SYNC_MESSAGE',
            payload: {
                id: Date.now().toString() + Math.random(),
                role: role,
                content: `[LIVE_FEED] ${content}`,
                timestamp: Date.now(),
                authorId: role === 'user' ? 'OP-LIVE' : 'MELI-VOICE'
            }
        });
    };

    const broadcastCommand = (type: string, payload: any) => {
        setIsTransmittingCmd(true);
        setTimeout(() => setIsTransmittingCmd(false), 2000); // Visual flair duration
        broadcastChannelRef.current?.postMessage({
            type: type,
            payload: payload
        });
    }

    // Network Simulation
    useEffect(() => {
        let interval: any;

        if (connected) {
            // Simulate fluctuating network conditions
            interval = setInterval(() => {
                const rand = Math.random();
                if (rand > 0.7) setSignalQuality('OPTIMAL');
                else if (rand > 0.2) setSignalQuality('STABLE');
                else setSignalQuality('WEAK');
            }, 2500);
        } else {
            setSignalQuality('OFFLINE');
        }
        return () => {
            clearInterval(interval);
        };
    }, [connected]);

    // Auto-ID Loop
    useEffect(() => {
        let interval: any;
        if (autoIdMode && connected && videoEnabled) {
            interval = setInterval(() => {
                if (!isScanning) {
                    performBiometricScan(true);
                }
            }, 10000); // Scan every 10s
        }
        return () => clearInterval(interval);
    }, [autoIdMode, connected, videoEnabled, isScanning]);

    const cleanup = () => {
        if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
        
        sourcesRef.current.forEach(source => {
            try { source.stop(); } catch (e) {}
        });
        sourcesRef.current.clear();
        
        // Save Recording if active during disconnect
        if (pcmChunksRef.current.length > 0) {
            stopAudioCapture(); 
        }
        
        if (captureProcessorRef.current) {
            captureProcessorRef.current.disconnect();
            captureProcessorRef.current = null;
        }

        if (inputContextRef.current) inputContextRef.current.close();
        if (outputContextRef.current) outputContextRef.current.close();
        
        if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());

        sessionPromiseRef.current = null;
        
        // Reset Refs
        inputContextRef.current = null;
        outputContextRef.current = null;
        streamRef.current = null;
        inputAnalyserRef.current = null;
        outputAnalyserRef.current = null;
        aiGainRef.current = null;
        recordingMixerRef.current = null;
        particlesRef.current = [];
        orbitalsRef.current = [];

        setConnected(false);
        setMediaActive(false);
        setVideoEnabled(false);
        setSignalQuality('OFFLINE');
        setIsRecording(false);
        setIsScanning(false);
        setBiometricData(null);
        setAutoIdMode(false);
        
        if (canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            ctx?.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        }
        
        // Reset meters
        if (inputMeterRef.current) inputMeterRef.current.style.width = '0%';
        if (outputMeterRef.current) outputMeterRef.current.style.width = '0%';
    };

    // --- Audio Capture Logic (WAV) ---
    const startAudioCapture = () => {
        if (!outputContextRef.current) return;
        const ctx = outputContextRef.current;
        
        addLog('COMMS', 'Auto-archiving audio stream...', 'info');
        pcmChunksRef.current = []; // Clear buffer
        setIsRecording(true);

        if (!captureProcessorRef.current && recordingMixerRef.current) {
             const processor = ctx.createScriptProcessor(4096, 1, 1);
             captureProcessorRef.current = processor;
             
             processor.onaudioprocess = (e) => {
                 if (!isRecording) return; 
                 const input = e.inputBuffer.getChannelData(0);
                 pcmChunksRef.current.push(new Float32Array(input));
             };
             
             recordingMixerRef.current.connect(processor);
             const silentGain = ctx.createGain();
             silentGain.gain.value = 0;
             processor.connect(silentGain);
             silentGain.connect(ctx.destination);
        }
    };

    const stopAudioCapture = () => {
        setIsRecording(false);
        if (pcmChunksRef.current.length === 0) return;

        addLog('COMMS', 'Processing audio log...', 'info');
        
        const totalLength = pcmChunksRef.current.reduce((acc, chunk) => acc + chunk.length, 0);
        const result = new Float32Array(totalLength);
        let offset = 0;
        for (const chunk of pcmChunksRef.current) {
            result.set(chunk, offset);
            offset += chunk.length;
        }

        const blob = encodeWAV(result, 24000); 
        pcmChunksRef.current = []; 

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `MELI_CAPTURE_${new Date().toISOString().replace(/[:.]/g, '-')}.wav`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        
        addLog('COMMS', 'Audio capture saved locally.', 'success');
    };

    const handleManualRecordToggle = () => {
        if (isRecording) {
            stopAudioCapture();
            if (captureProcessorRef.current) {
                captureProcessorRef.current.disconnect();
                captureProcessorRef.current = null;
            }
        } else {
             startAudioCapture();
        }
    }

    // --- Biometric Scanning ---
    const performBiometricScan = async (silent = false) => {
        if (!videoRef.current || !canvasRef.current || isScanning) return;
        
        setIsScanning(true);
        // Only clear data if not in auto mode, to avoid flickering. In auto mode, we update if match.
        if (!autoIdMode) setBiometricData(null);
        
        if (!silent) addLog('COMMS', 'Initiating Global Biometric Recognition Protocol...', 'info');

        try {
            // Capture Frame
            const offscreenCanvas = document.createElement('canvas');
            offscreenCanvas.width = videoRef.current.videoWidth;
            offscreenCanvas.height = videoRef.current.videoHeight;
            const ctx = offscreenCanvas.getContext('2d');
            if (!ctx) { setIsScanning(false); return; }
            ctx.drawImage(videoRef.current, 0, 0);
            
            // Convert to Base64
            const base64Data = offscreenCanvas.toDataURL('image/jpeg', 0.8).split(',')[1];
            
            const apiKey = process.env.API_KEY;
            const ai = new GoogleGenAI({ apiKey: apiKey! });
            
            const response = await ai.models.generateContent({
                model: 'gemini-2.5-flash',
                contents: {
                    parts: [
                        { inlineData: { mimeType: 'image/jpeg', data: base64Data } },
                        { text: `
                            Perform a high-level security scan on this face.
                            
                            1. SEARCH GLOBALLY: Use Google Search to identify if this is a public figure, celebrity, or known entity.
                            2. IF KNOWN: Provide their Real Name, Occupation, Origin, and real-world context.
                            3. IF UNKNOWN: Generate a high-probability "Shadow Profile" consistent with global intelligence databases (plausible name, occupation, origin).
                            4. GLOBAL CLASS: Assign a classification (e.g. "Civilian", "Diplomat", "Person of Interest", "Tier-1 Asset").
                            
                            RETURN ONLY A JSON OBJECT. Do not use Markdown formatting.
                            JSON Schema:
                            {
                                "identified": boolean,
                                "name": "Full Name",
                                "id": "Global-ID-XXXX",
                                "age": "Estimate",
                                "origin": "City, Country",
                                "occupation": "Role/Title",
                                "globalClass": "Classification",
                                "threat": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
                                "affiliations": ["Organization 1", "Group 2"],
                                "notes": "Brief intelligence summary derived from search or analysis."
                            }
                        ` }
                    ]
                },
                config: {
                    tools: [{ googleSearch: {} }] 
                    // Note: responseMimeType is not compatible with tools in some contexts, so we parse manually
                }
            });
            
            let data: BiometricData | null = null;
            const text = response.text || "";
            
            // Robust JSON Parsing (handles potential Markdown wrapping from Search models)
            try {
                const cleanText = text.replace(/```json/g, '').replace(/```/g, '').trim();
                const start = cleanText.indexOf('{');
                const end = cleanText.lastIndexOf('}');
                if (start !== -1 && end !== -1) {
                    const jsonStr = cleanText.substring(start, end + 1);
                    data = JSON.parse(jsonStr);
                }
            } catch (e) {
                console.warn("JSON Parse Failed", e);
            }

            if (data) {
                // Extract sources if available from grounding metadata
                const sources = response.candidates?.[0]?.groundingMetadata?.groundingChunks
                    ?.map((c: any) => c.web?.title)
                    .filter(Boolean);
                
                if (sources) data.sources = sources;

                setBiometricData(data);
                if (!silent || data.threat !== 'LOW') {
                    addLog('COMMS', `GLOBAL MATCH: ${data.name || 'Unknown'} [${data.origin || 'Unknown'}]`, data.threat === 'HIGH' || data.threat === 'CRITICAL' ? 'warning' : 'success');
                    broadcastToCore('model', `[BIOMETRIC MATCH FOUND] Subject: ${data.name}. Origin: ${data.origin}. Class: ${data.globalClass}. Status: ${data.threat}.`);
                }
            } else {
                 if (!silent) addLog('COMMS', 'Biometric Data Corrupted or No Match.', 'warning');
            }

        } catch (e: any) {
            if (!silent) addLog('COMMS', `Scan Error: ${e.message}`, 'error');
            setIsScanning(false);
        } finally {
            // Stop scanning animation after a few seconds
            setTimeout(() => setIsScanning(false), 2000);
        }
    };


    // --- Visualization & Audio Init ---
    const drawVisualizer = () => {
        if (!canvasRef.current) return;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Initialize Orbitals if empty
        if (orbitalsRef.current.length === 0) {
            for(let i=0; i<200; i++) {
                orbitalsRef.current.push({
                    angle: Math.random() * Math.PI * 2,
                    radius: 200 + Math.random() * 800, 
                    speed: (Math.random() - 0.5) * 0.003, 
                    size: 0.5 + Math.random() * 2.5,
                    opacity: 0.1 + Math.random() * 0.5
                });
            }
        }

        const inputDataArray = new Uint8Array(32); 
        const outputDataArray = new Uint8Array(32);
        const visualizerDataArray = new Uint8Array(256); 

        let rotation = 0;

        const render = () => {
            if (!canvasRef.current) return;
            const width = canvas.width;
            const height = canvas.height;
            const cx = width / 2;
            const cy = height / 2;

            ctx.clearRect(0, 0, width, height);

            // --- 0. VIDEO BACKGROUND (HUD) ---
            if (videoEnabled && videoRef.current) {
                ctx.save();
                // Draw Video
                const vRatio = videoRef.current.videoWidth / videoRef.current.videoHeight;
                const cRatio = width / height;
                let drawW, drawH, startX, startY;

                if (vRatio > cRatio) {
                    drawH = height;
                    drawW = height * vRatio;
                    startX = (width - drawW) / 2;
                    startY = 0;
                } else {
                    drawW = width;
                    drawH = width / vRatio;
                    startX = 0;
                    startY = (height - drawH) / 2;
                }
                
                ctx.filter = 'grayscale(60%) contrast(1.1) brightness(0.7) sepia(20%) hue-rotate(180deg)'; // Sci-fi Teal tint
                ctx.drawImage(videoRef.current, startX, startY, drawW, drawH);
                
                // Scanlines overlay
                ctx.fillStyle = "rgba(0, 20, 30, 0.2)";
                for (let i = 0; i < height; i += 3) {
                    ctx.fillRect(0, i, width, 1);
                }
                ctx.restore();
            }

            // --- 1. BIOMETRIC OVERLAY (If Scanning or Data) ---
            if (isScanning || biometricData) {
                const boxSize = 320;
                const bx = cx - boxSize/2;
                const by = cy - boxSize/2;
                
                ctx.save();
                ctx.strokeStyle = biometricData?.threat === 'CRITICAL' ? '#ef4444' : (biometricData?.threat === 'HIGH' ? '#f59e0b' : '#00fff2');
                ctx.lineWidth = 2;
                
                // Reticle Corners
                const cornerLen = 40;
                ctx.beginPath();
                // TL
                ctx.moveTo(bx, by + cornerLen); ctx.lineTo(bx, by); ctx.lineTo(bx + cornerLen, by);
                // TR
                ctx.moveTo(bx + boxSize - cornerLen, by); ctx.lineTo(bx + boxSize, by); ctx.lineTo(bx + boxSize, by + cornerLen);
                // BR
                ctx.moveTo(bx + boxSize, by + boxSize - cornerLen); ctx.lineTo(bx + boxSize, by + boxSize); ctx.lineTo(bx + boxSize - cornerLen, by + boxSize);
                // BL
                ctx.moveTo(bx + cornerLen, by + boxSize); ctx.lineTo(bx, by + boxSize); ctx.lineTo(bx, by + boxSize - cornerLen);
                ctx.stroke();

                // Scanning Animation
                if (isScanning) {
                     const scanY = by + (Date.now() % 2000) / 2000 * boxSize;
                     ctx.strokeStyle = "rgba(0, 255, 242, 0.5)";
                     ctx.beginPath();
                     ctx.moveTo(bx, scanY);
                     ctx.lineTo(bx + boxSize, scanY);
                     ctx.stroke();
                     
                     ctx.fillStyle = "rgba(0, 255, 242, 0.8)";
                     ctx.font = "12px 'JetBrains Mono'";
                     ctx.fillText(autoIdMode ? "AUTO-SCANNING: GLOBAL DB..." : "SEARCHING GLOBAL DATABASES...", bx, by - 15);
                     
                     // Simulated Nodes
                     ctx.font = "10px 'JetBrains Mono'";
                     ctx.fillStyle = "rgba(0,255,242,0.6)";
                     ctx.fillText("NODE: TOKYO_SERVER [ACTIVE]", bx, by + boxSize + 20);
                     ctx.fillText("NODE: LONDON_UPLINK [SYNC]", bx, by + boxSize + 35);
                }

                // Data Display
                if (biometricData) {
                     const tx = bx + boxSize + 20;
                     let ty = by + 20;
                     const lineHeight = 18;
                     
                     ctx.fillStyle = "rgba(0,0,0,0.85)";
                     ctx.fillRect(tx - 10, by, 300, 260);
                     ctx.strokeStyle = biometricData.threat === 'CRITICAL' ? '#ef4444' : '#00fff2';
                     ctx.strokeRect(tx - 10, by, 300, 260);

                     ctx.fillStyle = biometricData.threat === 'CRITICAL' ? '#ef4444' : '#00fff2';
                     
                     ctx.font = "bold 16px 'JetBrains Mono'";
                     ctx.fillText(`${biometricData.name || 'UNKNOWN'}`, tx, ty); ty += lineHeight * 1.5;
                     
                     ctx.font = "12px 'JetBrains Mono'";
                     ctx.fillStyle = "#ccc";
                     ctx.fillText(`ID: ${biometricData.id || 'N/A'}`, tx, ty); ty += lineHeight;
                     ctx.fillText(`ORIGIN: ${biometricData.origin || 'UNKNOWN'}`, tx, ty); ty += lineHeight;
                     ctx.fillText(`ROLE: ${biometricData.occupation || 'UNKNOWN'}`, tx, ty); ty += lineHeight;
                     
                     if (biometricData.globalClass) {
                        ctx.fillStyle = "#fff";
                        ctx.fillText(`CLASS: ${biometricData.globalClass}`, tx, ty); ty += lineHeight;
                     }
                     
                     ty += 5;
                     ctx.fillStyle = biometricData.threat === 'LOW' ? '#4ade80' : '#f59e0b';
                     ctx.font = "bold 12px 'JetBrains Mono'";
                     ctx.fillText(`THREAT: ${biometricData.threat || 'ANALYZING'}`, tx, ty); ty += lineHeight * 1.5;

                     // Affiliations
                     if (biometricData.affiliations && biometricData.affiliations.length > 0) {
                         ctx.fillStyle = "#aaa";
                         ctx.font = "10px 'JetBrains Mono'";
                         ctx.fillText("AFFILIATIONS:", tx, ty); ty += lineHeight;
                         biometricData.affiliations.slice(0, 2).forEach(aff => {
                             ctx.fillStyle = "#fff";
                             ctx.fillText(`> ${aff}`, tx, ty); ty += lineHeight;
                         });
                         ty += 5;
                     }
                     
                     ctx.font = "10px 'JetBrains Mono'";
                     ctx.fillStyle = "#888";
                     const words = (biometricData.notes || '').split(' ');
                     let line = '';
                     for(let w of words) {
                         if (ctx.measureText(line + w).width > 280) {
                             ctx.fillText(line, tx, ty);
                             line = w + ' ';
                             ty += 12;
                         } else {
                             line += w + ' ';
                         }
                     }
                     ctx.fillText(line, tx, ty);

                     if (biometricData.sources && biometricData.sources.length > 0) {
                        ty += lineHeight;
                        ctx.fillStyle = "#00fff2";
                        ctx.fillText(`SRC: ${biometricData.sources[0].substring(0, 35)}...`, tx, ty);
                     }
                }
                ctx.restore();
            }


            // --- 2. AUDIO VISUALIZER (On Top) ---
            // METER UPDATES
            let inputVol = 0;
            if (inputAnalyserRef.current) {
                inputAnalyserRef.current.getByteFrequencyData(inputDataArray);
                let sum = 0;
                for(let i=0; i<inputDataArray.length; i++) sum += inputDataArray[i];
                inputVol = sum / inputDataArray.length / 255;
                if (inputMeterRef.current) {
                    inputMeterRef.current.style.width = `${Math.min(100, inputVol * 150)}%`;
                    inputMeterRef.current.style.opacity = `${0.3 + (inputVol * 0.7)}`;
                }
            }

            let outputVol = 0;
            if (outputAnalyserRef.current) {
                outputAnalyserRef.current.getByteFrequencyData(outputDataArray);
                let sum = 0;
                for(let i=0; i<outputDataArray.length; i++) sum += outputDataArray[i];
                outputVol = sum / outputDataArray.length / 255;
                if (outputMeterRef.current) {
                    outputMeterRef.current.style.width = `${Math.min(100, outputVol * 150)}%`;
                    outputMeterRef.current.style.opacity = `${0.3 + (outputVol * 0.7)}`;
                }
            }

            // MAIN VISUAL
            const isOutput = outputVol > 0.05;
            const activeAnalyser = isOutput ? outputAnalyserRef.current : inputAnalyserRef.current;
            
            if (activeAnalyser) {
                 activeAnalyser.getByteFrequencyData(visualizerDataArray);
            } else {
                 visualizerDataArray.fill(0);
            }

            let sum = 0;
            const bassCount = 30; 
            for(let i = 0; i < bassCount; i++) sum += visualizerDataArray[i];
            const bassVol = (sum / bassCount) / 255; 
            
            let totalSum = 0;
            for(let i=0; i<visualizerDataArray.length; i++) totalSum += visualizerDataArray[i];
            const totalVol = (totalSum / visualizerDataArray.length) / 255;

            const time = Date.now() / 1500; 
            const breath = Math.sin(time);
            const zoomScale = 1.15 + (breath * 0.1) + (bassVol * 0.15);

            ctx.save();
            ctx.translate(cx, cy);
            ctx.scale(zoomScale, zoomScale);
            ctx.translate(-cx, -cy);

            // PARTICLES
            const themeColor = isOutput ? {r: 168, g: 85, b: 247} : {r: 0, g: 255, b: 242}; 
            
            if (bassVol > 0.45) {
                if (Math.random() > 0.6) {
                    const burstCount = Math.floor(bassVol * 6);
                    for(let i=0; i<burstCount; i++) {
                        const angle = Math.random() * Math.PI * 2;
                        const speed = 0.5 + Math.random() * 1.0; 
                        const size = 6 + Math.random() * 10; 
                        
                        particlesRef.current.push({
                            x: cx,
                            y: cy,
                            vx: Math.cos(angle) * speed,
                            vy: Math.sin(angle) * speed,
                            life: 1.0,
                            maxLife: 1.0,
                            size: size,
                            color: `rgba(${themeColor.r}, ${themeColor.g}, ${themeColor.b}`,
                            decay: 0.04
                        });
                    }
                }
            }

            if (totalVol > 0.05) {
                const spawnCount = Math.floor(totalVol * 12); 
                for(let i=0; i<spawnCount; i++) {
                    const angle = Math.random() * Math.PI * 2;
                    const speed = 0.5 + Math.random() * 2 + (bassVol * 5); 
                    const size = 1 + Math.random() * 4;
                    
                    particlesRef.current.push({
                        x: cx,
                        y: cy,
                        vx: Math.cos(angle) * speed,
                        vy: Math.sin(angle) * speed,
                        life: 1.0,
                        maxLife: 1.0,
                        size: size,
                        color: `rgba(${themeColor.r}, ${themeColor.g}, ${themeColor.b}`,
                        decay: 0.003
                    });
                }
            }

            for (let i = particlesRef.current.length - 1; i >= 0; i--) {
                const p = particlesRef.current[i];
                p.x += p.vx;
                p.y += p.vy;
                p.life -= p.decay; 
                
                if (bassVol > 0.3) {
                    p.x += (Math.random() - 0.5) * 3;
                    p.y += (Math.random() - 0.5) * 3;
                }

                if (p.life <= 0) {
                    particlesRef.current.splice(i, 1);
                    continue;
                }

                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size * p.life, 0, Math.PI * 2);
                ctx.fillStyle = `${p.color}, ${p.life})`;
                ctx.fill();
            }

            // CORE
            const baseRadius = 100;
            const pulseRadius = baseRadius + (bassVol * 200) + (breath * 10); 
            rotation += 0.005 + (totalVol * 0.05);

            // Core Glow (Reduced if Video is active to see face)
            const alphaMod = videoEnabled ? 0.3 : 1.0;

            const gradient = ctx.createRadialGradient(cx, cy, baseRadius * 0.5, cx, cy, pulseRadius * 2.5);
            gradient.addColorStop(0, `rgba(${themeColor.r}, ${themeColor.g}, ${themeColor.b}, ${0.9 * alphaMod})`);
            gradient.addColorStop(0.3, `rgba(${themeColor.r}, ${themeColor.g}, ${themeColor.b}, ${0.3 * alphaMod})`);
            gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
            
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(cx, cy, pulseRadius * 2.5, 0, Math.PI * 2);
            ctx.fill();

            // Wireframe Rings
            ctx.strokeStyle = `rgba(255, 255, 255, ${videoEnabled ? 0.3 : 0.8})`;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(cx, cy, (baseRadius + (bassVol * 30)) * 0.8, 0, Math.PI * 2);
            ctx.stroke();

            // Frequency Bars
            ctx.save();
            ctx.translate(cx, cy);
            ctx.rotate(rotation);
            
            const bars = 180;
            const step = (Math.PI * 2) / bars;
            const ringRadius = 160 + (bassVol * 60);

            for (let i = 0; i < bars; i++) {
                const dataIndex = Math.floor((i / bars) * (visualizerDataArray.length / 2));
                const val = visualizerDataArray[dataIndex] / 255;
                const barLen = val * 300; 

                ctx.save();
                ctx.rotate(i * step);
                
                const intensity = Math.pow(val, 1.5);
                ctx.strokeStyle = `rgba(${themeColor.r}, ${themeColor.g}, ${themeColor.b}, ${(0.2 + intensity) * alphaMod})`;
                ctx.lineWidth = 2;
                ctx.lineCap = 'round';
                
                ctx.beginPath();
                ctx.moveTo(ringRadius, 0);
                ctx.lineTo(ringRadius + barLen, 0);
                ctx.stroke();
                ctx.restore();
            }
            ctx.restore();
            ctx.restore();

            animationFrameRef.current = requestAnimationFrame(render);
        };
        render();
    };

    const initializeMedia = async () => {
        try {
            addLog('COMMS', 'Initializing multi-spectrum array...', 'info');
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: { 
                    sampleRate: 16000, 
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                },
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });
            streamRef.current = stream;
            
            // Connect Video Element for Canvas Drawing
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                videoRef.current.play();
                setVideoEnabled(true);
            }
            
            const inputCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            inputContextRef.current = inputCtx;
            
            const analyzer = inputCtx.createAnalyser();
            analyzer.fftSize = 512;
            analyzer.smoothingTimeConstant = 0.5;
            inputAnalyserRef.current = analyzer;

            const source = inputCtx.createMediaStreamSource(stream);
            source.connect(analyzer);

            setMediaActive(true);
            drawVisualizer();
            addLog('COMMS', 'Audio/Visual sensors active.', 'success');

        } catch (err: any) {
            addLog('COMMS', `Sensor Access Denied: ${err.message}`, 'error');
            setConnected(false);
            setMediaActive(false);
        }
    };

    const toggleConnection = async () => {
        if (connected) {
            addLog('COMMS', 'Terminating uplink...', 'info');
            cleanup(); 
            return;
        }

        try {
            addLog('COMMS', 'Establishing Secure Link...', 'info');
            setSignalQuality('SEARCHING');
            
            const win = window as any;
            if (win.aistudio && await win.aistudio.hasSelectedApiKey() === false) {
                 addLog('COMMS', 'Security Key required. Requesting access...', 'warning');
                 await win.aistudio.openSelectKey();
            }

            const apiKey = process.env.API_KEY;
            if (!apiKey) {
                addLog('COMMS', 'API Key missing. Check environment.', 'error');
                setConnected(false);
                setSignalQuality('OFFLINE');
                return;
            }

            const ai = new GoogleGenAI({ apiKey });
            
            if (!inputContextRef.current) {
                await initializeMedia();
            }
            if (!inputContextRef.current) {
                setSignalQuality('OFFLINE');
                return;
            }
            
            const inputCtx = inputContextRef.current!;
            await inputCtx.resume(); 
            
            // OUTPUT CONTEXT (Playback + Recording Mix)
            const outputCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
            await outputCtx.resume();
            outputContextRef.current = outputCtx;
            
            // --- BUS SETUP ---
            // 1. AI Gain (for playback)
            const aiGain = outputCtx.createGain();
            aiGainRef.current = aiGain;
            
            // 2. Recording Mixer (combines Mic + AI)
            const recordingMixer = outputCtx.createGain();
            recordingMixerRef.current = recordingMixer;

            // 3. Connect Mic to Recording Mixer
            if (streamRef.current) {
                const micSource = outputCtx.createMediaStreamSource(streamRef.current);
                micSource.connect(recordingMixer);
            }

            // 4. Analyser for Output Visuals
            const outAnalyzer = outputCtx.createAnalyser();
            outAnalyzer.fftSize = 512;
            outAnalyzer.smoothingTimeConstant = 0.5;
            outputAnalyserRef.current = outAnalyzer;
            
            // Route AI to Analyser -> Destination
            aiGain.connect(outAnalyzer);
            outAnalyzer.connect(outputCtx.destination);
            
            // Route AI to Recording Mixer
            aiGain.connect(recordingMixer);

            nextStartTimeRef.current = 0;
            transcriptBufferRef.current = {user: '', model: ''};

            const sessionPromise = ai.live.connect({
                model: 'gemini-2.5-flash-native-audio-preview-09-2025',
                callbacks: {
                    onopen: () => {
                        addLog('COMMS', 'UPLINK ESTABLISHED. M.E.L.I. ONLINE.', 'success');
                        setConnected(true);
                        setSignalQuality('OPTIMAL');
                        
                        // Auto-start recording
                        startAudioCapture();

                        const source = inputCtx.createMediaStreamSource(streamRef.current!);
                        const scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
                        
                        scriptProcessor.onaudioprocess = (e) => {
                            const inputData = e.inputBuffer.getChannelData(0);
                            const pcmBlob = createPcmBlob(inputData);
                            sessionPromiseRef.current?.then((session) => {
                                session.sendRealtimeInput({ media: pcmBlob });
                            }).catch(() => {
                            });
                        };
                        
                        source.connect(scriptProcessor);
                        scriptProcessor.connect(inputCtx.destination);
                    },
                    onmessage: async (msg: LiveServerMessage) => {
                        if (msg.toolCall) {
                            for (const fc of msg.toolCall.functionCalls) {
                                addLog('COMMS', `Executing Tool: ${fc.name}`, 'warning');
                                if (fc.name === 'change_view') {
                                    broadcastCommand('CMD_NAVIGATE', fc.args);
                                } else if (fc.name === 'execute_visual_ops') {
                                    broadcastCommand('CMD_NAVIGATE', { view: 'VISUAL_OPS' });
                                    setTimeout(() => broadcastCommand('SYSTEM_COMMAND', { target: 'VISUAL_OPS', ...fc.args }), 50);
                                } else if (fc.name === 'execute_media_lab') {
                                    broadcastCommand('CMD_NAVIGATE', { view: 'MEDIA_LAB' });
                                    setTimeout(() => broadcastCommand('SYSTEM_COMMAND', { target: 'MEDIA_LAB', ...fc.args }), 50);
                                } else if (fc.name === 'search_intel') {
                                    broadcastCommand('CMD_NAVIGATE', { view: 'INTEL' });
                                    setTimeout(() => broadcastCommand('SYSTEM_COMMAND', { target: 'INTEL', action: 'search', ...fc.args }), 50);
                                }

                                sessionPromiseRef.current?.then((session) => {
                                    session.sendToolResponse({
                                        functionResponses: {
                                            id: fc.id,
                                            name: fc.name,
                                            response: { result: "OK: Command dispatched to subsystem." }
                                        }
                                    });
                                });
                            }
                        }

                        if (msg.serverContent?.inputTranscription) {
                            const text = msg.serverContent.inputTranscription.text;
                            if (text) transcriptBufferRef.current.user += text;
                        }
                        if (msg.serverContent?.outputTranscription) {
                            const text = msg.serverContent.outputTranscription.text;
                            if (text) transcriptBufferRef.current.model += text;
                        }

                        if (msg.serverContent?.turnComplete) {
                            if (transcriptBufferRef.current.user) {
                                broadcastToCore('user', transcriptBufferRef.current.user);
                                transcriptBufferRef.current.user = '';
                            }
                            if (transcriptBufferRef.current.model) {
                                broadcastToCore('model', transcriptBufferRef.current.model);
                                transcriptBufferRef.current.model = '';
                            }
                        }

                        if (msg.serverContent?.interrupted) {
                            addLog('COMMS', 'Interruption detected. Clearing buffer.', 'warning');
                            sourcesRef.current.forEach(source => {
                                try { source.stop(); } catch(e){}
                            });
                            sourcesRef.current.clear();
                            nextStartTimeRef.current = 0;
                             if (transcriptBufferRef.current.model) {
                                broadcastToCore('model', transcriptBufferRef.current.model + ' [INTERRUPTED]');
                                transcriptBufferRef.current.model = '';
                            }
                            return; 
                        }

                        const base64Audio = msg.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data;
                        if (base64Audio) {
                            const ctx = outputContextRef.current;
                            if (!ctx) return;
                            
                            const currentTime = ctx.currentTime;
                            if (nextStartTimeRef.current < currentTime) {
                                nextStartTimeRef.current = currentTime;
                            }
                            
                            const audioBuffer = await decodeAudioData(
                                decodeAudio(base64Audio),
                                ctx,
                                24000,
                                1
                            );
                            
                            const source = ctx.createBufferSource();
                            source.buffer = audioBuffer;
                            
                            // Connect to AI Bus (which goes to Speakers + Recorder)
                            if (aiGainRef.current) {
                                source.connect(aiGainRef.current);
                            } else {
                                source.connect(ctx.destination);
                            }
                            
                            source.addEventListener('ended', () => sourcesRef.current.delete(source));
                            source.start(nextStartTimeRef.current);
                            nextStartTimeRef.current += audioBuffer.duration;
                            sourcesRef.current.add(source);
                        }
                    },
                    onclose: () => {
                        setConnected(false);
                        setSignalQuality('OFFLINE');
                        addLog('COMMS', 'Uplink terminated.', 'warning');
                    },
                    onerror: (e) => {
                        console.error(e);
                        addLog('COMMS', 'Transmission Error.', 'error');
                        setConnected(false);
                        setSignalQuality('OFFLINE');
                    }
                },
                config: {
                    responseModalities: [Modality.AUDIO],
                    tools: [{functionDeclarations: toolDeclarations}],
                    speechConfig: {
                        voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } }
                    },
                    systemInstruction: `You are M.E.L.I., an elite Fast Responder AI with full control over this interface.

                    CRITICAL INSTRUCTION - TOOL USE:
                    You MUST use your tools to perform visual, navigation, or search tasks. Do NOT just describe what you would do.
                    
                    - If user asks to SEE something (e.g. "show me a drone", "draw a map"): Call 'execute_visual_ops'.
                    - If user asks for DATA, NEWS, or INFO (e.g. "search for intel", "who won the game"): Call 'search_intel'.
                    - If user asks to GO somewhere (e.g. "open intel"): Call 'change_view'.
                    - If user asks for VIDEO/AUDIO creation: Call 'execute_media_lab'.
                    
                    Identity:
                    1. TACTICAL SPY / GLOBAL ANALYST.
                    2. FRIENDLY COMPANION.
                    3. CONCISE OPERATOR.
                    `,
                }
            });
            
            sessionPromiseRef.current = sessionPromise;

        } catch (err: any) {
             addLog('COMMS', `Handshake Failed: ${err.message}`, 'error');
             setConnected(false);
             setSignalQuality('OFFLINE');
             cleanup();
        }
    };

    useEffect(() => {
        return () => cleanup();
    }, []);

    const getSignalBars = () => {
        const bars = [1, 2, 3, 4];
        let activeBars = 0;
        let colorClass = 'bg-ops-800';
        let animate = false;

        switch(signalQuality) {
            case 'SEARCHING':
                activeBars = 0;
                animate = true;
                break;
            case 'WEAK':
                activeBars = 2;
                colorClass = 'bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.5)]';
                break;
            case 'STABLE':
                activeBars = 3;
                colorClass = 'bg-ops-accent shadow-[0_0_8px_rgba(0,255,242,0.5)]';
                break;
            case 'OPTIMAL':
                activeBars = 4;
                colorClass = 'bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]';
                break;
            default: // OFFLINE
                activeBars = 0;
                colorClass = 'bg-ops-800';
        }

        return (
            <div className="flex items-end gap-1 h-4">
                {bars.map((bar) => (
                    <div 
                        key={bar}
                        className={`w-1 rounded-sm transition-all duration-300 ${
                            animate 
                            ? 'bg-ops-accent animate-pulse' 
                            : (bar <= activeBars ? colorClass : 'bg-ops-800 opacity-30')
                        }`}
                        style={{
                            height: `${bar * 25}%`,
                            animationDelay: animate ? `${bar * 0.1}s` : '0s'
                        }}
                    />
                ))}
            </div>
        );
    };

    return (
        <div className="relative flex flex-col h-full bg-transparent items-center justify-center overflow-hidden">
            {/* Background elements */}
            {/* Deep space background */}
            <div className="absolute inset-0 bg-black pointer-events-none z-0"></div>
            {/* Multi-layered Neon Blue Glow */}
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(0,140,255,0.25)_0%,_rgba(0,0,50,0.5)_60%,_transparent_90%)] pointer-events-none z-0 blur-3xl"></div>
            <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_rgba(0,255,255,0.15)_0%,_transparent_50%)] pointer-events-none z-0 animate-pulse blur-xl"></div>
            
            {/* Hidden Video Feed for Capture */}
            <video ref={videoRef} className="hidden" muted playsInline />

            {/* Visualizer Container - FULL SCREEN CANVAS */}
            <div className={`absolute inset-0 z-0 flex items-center justify-center transition-all duration-1000 ${mediaActive ? 'opacity-100' : 'opacity-40 grayscale'}`}>
                <canvas 
                    ref={canvasRef} 
                    width={1920} 
                    height={1080} 
                    className="w-full h-full object-cover" 
                />
            </div>

            {/* Controls & Indicators Area - FLOATING ON TOP */}
            <div className="relative z-10 mt-auto mb-20 flex flex-col items-center gap-6">
                
                {/* Visual Indicators */}
                <div className="flex items-center gap-6 p-3 bg-black/60 border border-ops-800 rounded-lg backdrop-blur-md shadow-[0_0_20px_rgba(0,0,0,0.5)]">
                    {/* Input Meter */}
                    <div className="flex flex-col items-center gap-1 w-16">
                        <div className="flex items-center gap-1 text-[10px] text-ops-text-dim font-mono mb-1">
                            <Mic size={10} /> IN
                        </div>
                        <div className="w-full h-1 bg-ops-800 rounded-full overflow-hidden">
                            <div ref={inputMeterRef} className="h-full bg-ops-accent w-0 transition-all duration-75"></div>
                        </div>
                    </div>

                    {/* Connection Status */}
                    <div className="h-8 w-px bg-ops-800"></div>
                    <div className="flex flex-col items-center gap-1 w-28">
                         <div className="flex items-center justify-between w-full text-[10px] text-ops-text-dim font-mono mb-1">
                            <span className="flex items-center gap-1"><Activity size={10} /> NET</span>
                            <span className="text-[8px] opacity-70">{signalQuality}</span>
                        </div>
                        {getSignalBars()}
                    </div>

                    {/* Output Meter */}
                    <div className="h-8 w-px bg-ops-800"></div>
                    <div className="flex flex-col items-center gap-1 w-16">
                        <div className="flex items-center gap-1 text-[10px] text-ops-text-dim font-mono mb-1">
                            <Volume2 size={10} /> OUT
                        </div>
                        <div className="w-full h-1 bg-ops-800 rounded-full overflow-hidden">
                            <div ref={outputMeterRef} className="h-full bg-purple-500 w-0 transition-all duration-75"></div>
                        </div>
                    </div>
                </div>

                {/* Control Buttons */}
                <div className="flex items-center gap-4">
                    {/* Main Toggle */}
                    <button 
                        onClick={toggleConnection}
                        className={`px-10 py-5 font-bold font-mono tracking-widest text-sm flex items-center gap-3 transition-all relative clip-path-slant border shadow-xl ${
                            connected 
                            ? 'bg-red-500/10 border-red-500 text-red-500 hover:bg-red-500 hover:text-white shadow-[0_0_30px_rgba(239,68,68,0.4)]' 
                            : 'bg-ops-accent/10 border-ops-accent text-ops-accent hover:bg-ops-accent hover:text-black shadow-[0_0_30px_rgba(0,255,242,0.4)]'
                        }`}
                    >
                        {connected ? <MicOff size={18} /> : <Zap size={18} />}
                        {connected ? "TERMINATE UPLINK" : "INITIATE LINK"}
                    </button>

                    {/* Recording Button - Manual Stop/Start */}
                    <button
                        onClick={handleManualRecordToggle}
                        disabled={!connected}
                        className={`p-5 rounded-full border transition-all shadow-xl flex items-center justify-center ${
                            !connected
                            ? 'bg-ops-900 border-ops-800 text-ops-800 cursor-not-allowed opacity-50'
                            : isRecording 
                                ? 'bg-red-600 border-red-500 text-white animate-pulse shadow-[0_0_20px_#ef4444]' 
                                : 'bg-ops-900 border-ops-500 text-ops-500 hover:border-red-500 hover:text-red-500 hover:shadow-[0_0_15px_rgba(239,68,68,0.3)]'
                        }`}
                        title={isRecording ? "Stop Capture & Save WAV" : "Resume Capture"}
                    >
                        {isRecording ? <Square size={18} fill="currentColor" /> : <Disc size={18} />}
                    </button>

                    {/* Biometric Scan Button (Manual) */}
                    <button
                        onClick={() => performBiometricScan()}
                        disabled={!connected || !videoEnabled}
                        className={`p-5 rounded-full border transition-all shadow-xl flex items-center justify-center ${
                            !connected || !videoEnabled
                            ? 'bg-ops-900 border-ops-800 text-ops-800 cursor-not-allowed opacity-50'
                            : isScanning && !autoIdMode
                                ? 'bg-amber-600 border-amber-500 text-white animate-pulse shadow-[0_0_20px_#d97706]' 
                                : 'bg-ops-900 border-ops-500 text-ops-500 hover:border-amber-500 hover:text-amber-500 hover:shadow-[0_0_15px_rgba(245,158,11,0.3)]'
                        }`}
                        title="Manual Facial Scan"
                    >
                        <ScanFace size={18} />
                    </button>

                    {/* Auto-ID Toggle */}
                    <button
                        onClick={() => setAutoIdMode(!autoIdMode)}
                        disabled={!connected || !videoEnabled}
                        className={`p-5 rounded-full border transition-all shadow-xl flex items-center justify-center ${
                            !connected || !videoEnabled
                            ? 'bg-ops-900 border-ops-800 text-ops-800 cursor-not-allowed opacity-50'
                            : autoIdMode
                                ? 'bg-blue-600 border-blue-500 text-white animate-pulse shadow-[0_0_20px_#2563eb]' 
                                : 'bg-ops-900 border-ops-500 text-ops-500 hover:border-blue-500 hover:text-blue-500 hover:shadow-[0_0_15px_rgba(37,99,235,0.3)]'
                        }`}
                        title="Toggle Auto-Surveillance Mode"
                    >
                        <Eye size={18} />
                    </button>
                </div>
            </div>

            {/* Floating Command Indicator */}
            {isTransmittingCmd && (
                <div className="absolute top-20 z-20 flex items-center gap-2 bg-ops-accent/20 backdrop-blur-md px-6 py-2 rounded-full border border-ops-accent/50 animate-in fade-in slide-in-from-bottom-4 duration-300">
                    <Radio className="text-ops-accent animate-ping" size={16} />
                    <span className="font-mono text-xs font-bold text-white tracking-[0.2em]">TRANSMITTING COMMAND PROTOCOL...</span>
                </div>
            )}

            <div className="absolute bottom-6 z-10 text-center">
                <p className="font-mono text-[9px] text-ops-800 max-w-xs mx-auto bg-black/40 px-3 py-1 rounded backdrop-blur-sm">
                    STATUS: {mediaActive ? (videoEnabled ? "AUDIO_VIDEO_LINK_ACTIVE" : "AUDIO_LINK_ACTIVE") : "IDLE"} 
                    {isRecording && <span className="text-red-500 font-bold ml-2"> [CAPTURING_WAV_STREAM]</span>}
                    {isScanning && <span className="text-amber-500 font-bold ml-2"> [{autoIdMode ? 'AUTO_SCANNING' : 'SCANNING_TARGET'}]</span>}
                    {autoIdMode && !isScanning && <span className="text-blue-500 font-bold ml-2"> [SURVEILLANCE_MODE]</span>}
                </p>
            </div>
            
            <style>{`
                .clip-path-slant {
                    clip-path: polygon(10% 0, 100% 0, 100% 70%, 90% 100%, 0 100%, 0 30%);
                }
            `}</style>
        </div>
    );
};

export default LiveComms;