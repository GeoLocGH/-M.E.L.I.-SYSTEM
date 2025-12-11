import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenAI } from "@google/genai";
import { LogEntry } from '../types';
import { Image, Upload, Wand2, ScanEye, Download, RefreshCw, Key, Shield, Radio, Globe } from 'lucide-react';
import { fileToGenerativePart } from '../services/geminiUtils';

interface VisualOpsProps {
    addLog: (source: string, message: string, type?: LogEntry['type']) => void;
}

const VisualOps: React.FC<VisualOpsProps> = ({ addLog }) => {
    const [subMode, setSubMode] = useState<'generate' | 'analyze' | 'edit'>('generate');
    const [prompt, setPrompt] = useState('');
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [resultImage, setResultImage] = useState<string | null>(null);
    const [analysisResult, setAnalysisResult] = useState<string>('');
    const [loading, setLoading] = useState(false);
    
    // Configs
    const [aspectRatio, setAspectRatio] = useState('1:1');
    const [imageSize, setImageSize] = useState('1K');
    const channelRef = useRef<BroadcastChannel | null>(null);

    // Sync Listener
    useEffect(() => {
        const channel = new BroadcastChannel('meli_mesh_network');
        channelRef.current = channel;
        
        channel.onmessage = (event) => {
            const { type, payload } = event.data;
            if (type === 'SYSTEM_COMMAND' && payload.target === 'VISUAL_OPS') {
                addLog('VISUAL', `Remote command received: ${payload.action || 'generate'}`, 'warning');
                
                // Sync State
                if (payload.action) setSubMode(payload.action);
                if (payload.prompt) setPrompt(payload.prompt);

                // Immediate Execution
                if (payload.prompt) {
                    // Force loading state immediately for UX feedback during view switch
                    setLoading(true);
                    triggerAction(payload.prompt, payload.action || 'generate');
                }
            }
        };

        return () => channel.close();
    }, []);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) {
            setImageFile(file);
            const reader = new FileReader();
            reader.onload = (e) => setImagePreview(e.target?.result as string);
            reader.readAsDataURL(file);
            setResultImage(null);
            setAnalysisResult('');
        }
    };

    const triggerAction = async (currentPrompt: string, currentMode: string) => {
         await handleAction(currentPrompt, currentMode);
    };

    const getActiveImagePart = async () => {
        // Priority 1: Uploaded File
        if (imageFile) {
            return await fileToGenerativePart(imageFile);
        }
        // Priority 2: Previously Generated Result (Chain of Thought/Chain of Visuals)
        if (resultImage) {
             const matches = resultImage.match(/^data:(.+);base64,(.+)$/);
             if (matches && matches.length === 3) {
                 return {
                     inlineData: {
                         mimeType: matches[1],
                         data: matches[2]
                     }
                 };
             }
        }
        return null;
    };

    const handleAction = async (overridePrompt?: string, overrideMode?: string) => {
        const activePrompt = overridePrompt || prompt;
        const activeMode = overrideMode || subMode;

        if (!activePrompt && activeMode === 'generate') {
             setLoading(false);
             return;
        }
        
        setLoading(true);
        addLog('VISUAL', `Starting ${activeMode.toUpperCase()} sequence...`, 'info');

        try {
            const win = window as any;
            if (win.aistudio && await win.aistudio.hasSelectedApiKey() === false) {
                 addLog('VISUAL', 'Security Clearance required', 'warning');
                 await win.aistudio.openSelectKey();
            }

            const apiKey = process.env.API_KEY;
            if (!apiKey) {
                 throw new Error("API Key missing. Please select a key.");
            }

            const ai = new GoogleGenAI({ apiKey });

            if (activeMode === 'generate') {
                // Enable Google Search Grounding for Image Generation
                const response = await ai.models.generateContent({
                    model: 'gemini-3-pro-image-preview',
                    contents: { parts: [{ text: activePrompt }] },
                    config: {
                        imageConfig: {
                            aspectRatio: aspectRatio,
                            imageSize: imageSize,
                        },
                        tools: [{ googleSearch: {} }] // Allows model to fetch data to inform the image
                    },
                });

                let found = false;
                for (const part of response.candidates?.[0]?.content?.parts || []) {
                    if (part.inlineData) {
                         const base64EncodeString: string = part.inlineData.data;
                         const imageUrl = `data:image/png;base64,${base64EncodeString}`;
                         setResultImage(imageUrl);
                         found = true;
                         addLog('VISUAL', 'Visual construct generated successfully.', 'success');
                         break;
                    }
                }
                if (!found) throw new Error("No visual data returned. The model might have refused the request.");

            } else if (activeMode === 'analyze') {
                const imagePart = await getActiveImagePart();
                if (!imagePart) throw new Error("Target Media Missing for Analysis. Upload media or generate a visual first.");
                
                const response = await ai.models.generateContent({
                    model: 'gemini-3-pro-preview',
                    contents: {
                        parts: [
                            imagePart, 
                            { text: activePrompt || "Perform forensic image analysis. Identify location, time of day, identifiable individuals, text, and potential security anomalies." }
                        ]
                    }
                });
                
                setAnalysisResult(response.text || "Analysis algorithms yielded no text output.");
                addLog('VISUAL', 'Forensic analysis complete.', 'success');

            } else if (activeMode === 'edit') {
                 const imagePart = await getActiveImagePart();
                 if (!imagePart) throw new Error("Target Media Missing for Edit. Upload media or generate a visual first.");
                 
                 const response = await ai.models.generateContent({
                    model: 'gemini-2.5-flash-image',
                    contents: {
                        parts: [imagePart, { text: activePrompt || "Enhance visual clarity and remove obstructions." }]
                    }
                });

                let found = false;
                for (const part of response.candidates?.[0]?.content?.parts || []) {
                    if (part.inlineData) {
                         const base64EncodeString: string = part.inlineData.data;
                         const imageUrl = `data:image/png;base64,${base64EncodeString}`;
                         setResultImage(imageUrl);
                         found = true;
                         addLog('VISUAL', 'Image modification complete.', 'success');
                         break;
                    }
                }
                if (!found) throw new Error("Modification failed. Model returned no image data.");
            }

        } catch (err: any) {
            console.error(err);
            if (err.message && err.message.includes("Requested entity was not found")) {
                 const win = window as any;
                 if (win.aistudio) {
                     addLog('VISUAL', 'Resource access denied. Re-requesting authorization...', 'warning');
                     await win.aistudio.openSelectKey();
                 }
            }
            addLog('VISUAL', `Operation aborted: ${err.message}`, 'error');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-ops-900/30">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-ops-800 bg-ops-900/80 backdrop-blur z-10">
                <h2 className="text-xl font-bold font-mono text-ops-text-main tracking-widest flex items-center gap-2">
                    <Shield size={20} className="text-purple-400" />
                    VISUAL<span className="text-purple-400">_OPS</span>
                </h2>
                <div className="flex gap-2">
                    <button 
                        onClick={() => { setSubMode('generate'); }}
                        className={`flex items-center gap-1 px-3 py-1 rounded-sm text-[10px] font-bold font-mono border tracking-wider transition-all ${subMode === 'generate' ? 'bg-purple-600 border-purple-600 text-white' : 'border-ops-800 text-ops-text-dim hover:border-purple-600'}`}>
                        <Wand2 size={10} /> RECON_GEN
                    </button>
                    <button 
                        onClick={() => { setSubMode('edit'); }}
                        className={`flex items-center gap-1 px-3 py-1 rounded-sm text-[10px] font-bold font-mono border tracking-wider transition-all ${subMode === 'edit' ? 'bg-indigo-600 border-indigo-600 text-white' : 'border-ops-800 text-ops-text-dim hover:border-indigo-600'}`}>
                        <RefreshCw size={10} /> MODIFY
                    </button>
                    <button 
                        onClick={() => { setSubMode('analyze'); }}
                        className={`flex items-center gap-1 px-3 py-1 rounded-sm text-[10px] font-bold font-mono border tracking-wider transition-all ${subMode === 'analyze' ? 'bg-blue-600 border-blue-600 text-white' : 'border-ops-800 text-ops-text-dim hover:border-blue-600'}`}>
                        <ScanEye size={10} /> FORENSICS
                    </button>
                </div>
            </div>

            <div className="flex-1 p-6 flex gap-6 overflow-hidden">
                {/* Controls */}
                <div className="w-1/3 flex flex-col gap-4">
                    
                    {subMode === 'generate' && (
                        <div className="bg-ops-900 border border-ops-800 p-4 rounded-sm">
                            <label className="text-[10px] font-bold font-mono text-ops-400 mb-2 block tracking-wider flex justify-between">
                                <span>CONFIGURATION</span>
                                <span className="flex items-center gap-1 text-green-500"><Globe size={8}/> GROUNDED</span>
                            </label>
                            <div className="grid grid-cols-2 gap-2 mb-2">
                                <select 
                                    value={aspectRatio} onChange={(e) => setAspectRatio(e.target.value)}
                                    className="bg-ops-950 border border-ops-800 text-ops-text-main text-xs p-2 focus:outline-none focus:border-purple-500 transition-colors"
                                >
                                    <option value="1:1">1:1 (Square)</option>
                                    <option value="16:9">16:9 (Wide)</option>
                                    <option value="9:16">9:16 (Tall)</option>
                                    <option value="4:3">4:3 (Std)</option>
                                    <option value="3:4">3:4 (Std)</option>
                                </select>
                                <select 
                                    value={imageSize} onChange={(e) => setImageSize(e.target.value)}
                                    className="bg-ops-950 border border-ops-800 text-ops-text-main text-xs p-2 focus:outline-none focus:border-purple-500 transition-colors"
                                >
                                    <option value="1K">1K Res</option>
                                    <option value="2K">2K Res</option>
                                    <option value="4K">4K Res</option>
                                </select>
                            </div>
                             <div className="flex items-center gap-2 text-[10px] text-yellow-500 mt-2 font-mono">
                                <Key size={10} /> AUTH KEY REQUIRED
                            </div>
                        </div>
                    )}

                    {(subMode === 'edit' || subMode === 'analyze') && (
                        <div className="bg-ops-900 border border-ops-800 p-4 rounded-sm">
                            <label className="text-[10px] font-bold font-mono text-ops-400 mb-2 block tracking-wider">SOURCE INTEL</label>
                            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-ops-800 border-dashed rounded-sm cursor-pointer hover:bg-ops-800/50 hover:border-ops-500 transition-all">
                                <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                    <Upload className="w-8 h-8 mb-2 text-ops-400" />
                                    <p className="text-[10px] font-mono text-ops-text-dim">{imageFile ? imageFile.name : (resultImage ? "USING GENERATED VISUAL" : "UPLOAD TARGET MEDIA")}</p>
                                </div>
                                <input type="file" className="hidden" onChange={handleFileChange} accept="image/*" />
                            </label>
                            {(imagePreview || (resultImage && !imageFile)) && (
                                <div className="mt-2 h-32 bg-black flex items-center justify-center border border-ops-800 relative">
                                    <div className="absolute top-1 left-1 text-[8px] bg-red-600 text-white px-1 font-mono">
                                        {imageFile ? "CONFIDENTIAL UPLOAD" : "GENERATED ASSET"}
                                    </div>
                                    <img src={imagePreview || resultImage!} alt="Preview" className="max-h-full max-w-full object-contain opacity-80" />
                                </div>
                            )}
                        </div>
                    )}

                    <div className="flex-1 flex flex-col">
                         <div className="bg-ops-900 border border-ops-800 p-1 flex-1 flex flex-col">
                            <textarea
                                value={prompt}
                                onChange={(e) => setPrompt(e.target.value)}
                                placeholder={
                                    subMode === 'generate' ? "Enter visual parameters..." :
                                    subMode === 'edit' ? "Describe modification directives..." :
                                    "Specific forensic questions..."
                                }
                                className="flex-1 bg-ops-950/50 text-ops-text-main font-mono text-sm p-3 resize-none focus:outline-none placeholder:text-ops-text-dim"
                            />
                         </div>
                        <button
                            onClick={() => handleAction()}
                            disabled={loading}
                            className={`mt-4 py-3 font-bold font-mono tracking-widest flex items-center justify-center gap-2 border border-transparent hover:border-white transition-all ${loading ? 'bg-ops-800 text-ops-text-dim' : 'bg-ops-text-main text-ops-950 hover:bg-ops-text-dim'}`}
                        >
                           {loading ? <RefreshCw className="animate-spin" /> : 'EXECUTE PROTOCOL'}
                        </button>
                    </div>
                </div>

                {/* Output */}
                <div className="flex-1 bg-black border border-ops-800 relative flex items-center justify-center overflow-hidden">
                    <div className="absolute inset-0 z-0 opacity-10 pointer-events-none" 
                        style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '20px 20px' }}>
                    </div>

                    {!resultImage && !analysisResult && !loading && (
                        <div className="text-ops-800 flex flex-col items-center z-10">
                            <Image size={64} className="mb-4 opacity-20" />
                            <span className="font-mono text-xs opacity-40 tracking-widest">AWAITING VISUAL OUTPUT</span>
                        </div>
                    )}

                    {loading && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-black/80 backdrop-blur-sm">
                            <Radio className="text-ops-accent animate-ping mb-4" size={48} />
                            <span className="font-mono text-xs text-ops-accent tracking-[0.2em] animate-pulse">RECEIVING DATA STREAM...</span>
                        </div>
                    )}

                    {resultImage && subMode === 'generate' && !analysisResult && !loading && (
                        <div className="relative w-full h-full flex items-center justify-center group z-10">
                            <img src={resultImage} alt="Generated" className="max-h-full max-w-full object-contain shadow-[0_0_30px_rgba(0,0,0,0.5)]" />
                            <a href={resultImage} download="meli_output.png" className="absolute bottom-4 right-4 bg-ops-accent text-black p-2 rounded-full opacity-0 group-hover:opacity-100 transition-opacity">
                                <Download size={20} />
                            </a>
                            <div className="absolute top-4 left-4 text-xs font-mono text-green-500 bg-black/80 px-2 py-1 border border-green-500/30">
                                RENDER_COMPLETE
                            </div>
                        </div>
                    )}

                     {(resultImage && (subMode === 'edit' || subMode === 'analyze')) && !analysisResult && !loading && (
                         <div className="relative w-full h-full flex items-center justify-center group z-10">
                             <img src={resultImage} alt="Target" className="max-h-full max-w-full object-contain shadow-[0_0_30px_rgba(0,0,0,0.5)] opacity-50 grayscale" />
                             <div className="absolute inset-0 flex items-center justify-center">
                                 <ScanEye size={48} className="text-ops-accent animate-pulse" />
                             </div>
                         </div>
                    )}

                    {analysisResult && !loading && (
                        <div className="absolute inset-0 p-6 overflow-y-auto font-mono text-sm text-green-400 leading-relaxed bg-black/95 z-20">
                            <h3 className="text-white border-b border-ops-800 pb-2 mb-4 font-bold flex items-center gap-2">
                                <ScanEye size={16} /> FORENSIC REPORT
                            </h3>
                            {analysisResult}
                            <button onClick={() => setAnalysisResult('')} className="mt-4 text-[10px] text-ops-text-dim underline hover:text-white">CLOSE REPORT</button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default VisualOps;