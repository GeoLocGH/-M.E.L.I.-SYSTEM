import React, { useState, useEffect, useRef } from 'react';
import { GoogleGenAI, Modality } from "@google/genai";
import { LogEntry } from '../types';
import { Video, Mic, Speaker, Play, Loader2, Key } from 'lucide-react';
import { blobToBase64, fileToGenerativePart, decodeAudioData, decodeAudio } from '../services/geminiUtils';

interface MediaLabProps {
    addLog: (source: string, message: string, type?: LogEntry['type']) => void;
}

const MediaLab: React.FC<MediaLabProps> = ({ addLog }) => {
    const [tab, setTab] = useState<'video' | 'audio'>('video');
    
    // Video State
    const [videoPrompt, setVideoPrompt] = useState('');
    const [videoFile, setVideoFile] = useState<File | null>(null); // For start image (image to video)
    const [videoRatio, setVideoRatio] = useState('16:9');
    const [generatedVideoUrl, setGeneratedVideoUrl] = useState<string | null>(null);
    const [videoLoading, setVideoLoading] = useState(false);

    // Audio State
    const [ttsText, setTtsText] = useState('');
    const [isRecording, setIsRecording] = useState(false);
    const [transcript, setTranscript] = useState('');
    const channelRef = useRef<BroadcastChannel | null>(null);

     // Sync Listener
     useEffect(() => {
        const channel = new BroadcastChannel('meli_mesh_network');
        channelRef.current = channel;
        
        channel.onmessage = (event) => {
            const { type, payload } = event.data;
            if (type === 'SYSTEM_COMMAND' && payload.target === 'MEDIA_LAB') {
                if (payload.type === 'video') {
                     setTab('video');
                     if (payload.prompt) {
                         setVideoPrompt(payload.prompt);
                         setTimeout(() => handleGenerateVideo(payload.prompt), 500);
                     }
                } else if (payload.type === 'audio') {
                    setTab('audio');
                     if (payload.prompt) {
                         setTtsText(payload.prompt);
                         setTimeout(() => handleTTS(payload.prompt), 500);
                     }
                }
            }
        };

        return () => channel.close();
    }, []);

    // --- VIDEO HANDLERS ---
    const handleGenerateVideo = async (overridePrompt?: string) => {
        const activePrompt = overridePrompt || videoPrompt;
        if (!activePrompt) return;
        setVideoLoading(true);
        addLog('MEDIA', 'Initializing Veo 3.1 generation protocol...', 'info');

        try {
            const win = window as any;
            if (win.aistudio && await win.aistudio.hasSelectedApiKey() === false) {
                 addLog('MEDIA', 'Paid API Key required for Veo', 'warning');
                 await win.aistudio.openSelectKey();
            }
            
            const apiKey = process.env.API_KEY;
            if (!apiKey) throw new Error("API Key missing. Please select a key.");

            // New instance to get key
            const ai = new GoogleGenAI({ apiKey });
            
            let request: any = {
                model: 'veo-3.1-fast-generate-preview',
                prompt: activePrompt,
                config: {
                    numberOfVideos: 1,
                    resolution: '720p',
                    aspectRatio: videoRatio
                }
            };

            if (videoFile) {
                const imgPart = await fileToGenerativePart(videoFile);
                request.image = {
                    imageBytes: imgPart.inlineData.data,
                    mimeType: imgPart.inlineData.mimeType
                };
            }

            let operation = await ai.models.generateVideos(request);
            addLog('MEDIA', 'Video generation task submitted. Polling...', 'info');

            while (!operation.done) {
                await new Promise(resolve => setTimeout(resolve, 5000)); // Poll every 5s
                operation = await ai.operations.getVideosOperation({operation: operation});
                addLog('MEDIA', 'Rendering frames...', 'info');
            }

            const downloadLink = operation.response?.generatedVideos?.[0]?.video?.uri;
            if (downloadLink) {
                 // Must append API Key
                 const finalUrl = `${downloadLink}&key=${apiKey}`;
                 setGeneratedVideoUrl(finalUrl);
                 addLog('MEDIA', 'Video rendered successfully.', 'success');
            } else {
                throw new Error("No video URI returned.");
            }

        } catch (err: any) {
            console.error(err);
            if (err.message && err.message.includes("Requested entity was not found")) {
                 const win = window as any;
                 if (win.aistudio) {
                     addLog('MEDIA', 'Resource access denied. Re-requesting authorization...', 'warning');
                     await win.aistudio.openSelectKey();
                 }
            }
            addLog('MEDIA', `Video Gen failed: ${err.message}`, 'error');
        } finally {
            setVideoLoading(false);
        }
    };

    // --- AUDIO HANDLERS ---
    const handleTTS = async (overrideText?: string) => {
        const activeText = overrideText || ttsText;
        if (!activeText) return;
        addLog('AUDIO', 'Synthesizing speech...', 'info');
        try {
             const win = window as any;
             if (win.aistudio && await win.aistudio.hasSelectedApiKey() === false) {
                  await win.aistudio.openSelectKey();
             }

             const apiKey = process.env.API_KEY;
             if (!apiKey) throw new Error("API Key missing.");

             const ai = new GoogleGenAI({ apiKey });
             const response = await ai.models.generateContent({
                 model: 'gemini-2.5-flash-preview-tts',
                 contents: { parts: [{ text: activeText }] },
                 config: {
                     responseModalities: [Modality.AUDIO],
                     speechConfig: {
                         voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }
                     }
                 }
             });

             const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
             if (base64Audio) {
                 const ctx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 24000});
                 const audioBuffer = await decodeAudioData(decodeAudio(base64Audio), ctx, 24000, 1);
                 const source = ctx.createBufferSource();
                 source.buffer = audioBuffer;
                 source.connect(ctx.destination);
                 source.start();
                 addLog('AUDIO', 'Audio playback started.', 'success');
             }
        } catch (err: any) {
             addLog('AUDIO', `TTS Failed: ${err.message}`, 'error');
        }
    };

    const handleTranscribe = async () => {
        // Simple 5 second recording for demo purposes
        if (isRecording) return;
        setIsRecording(true);
        addLog('AUDIO', 'Recording 5s audio sample...', 'info');
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const mediaRecorder = new MediaRecorder(stream);
            const chunks: BlobPart[] = [];
            
            mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
            mediaRecorder.onstop = async () => {
                const blob = new Blob(chunks, { type: 'audio/webm' }); // Typically webm/opus
                const base64 = await blobToBase64(blob);
                
                addLog('AUDIO', 'Transcribing...', 'info');
                
                const win = window as any;
                if (win.aistudio && await win.aistudio.hasSelectedApiKey() === false) {
                     await win.aistudio.openSelectKey();
                }

                const apiKey = process.env.API_KEY;
                if (!apiKey) {
                    addLog('AUDIO', 'API Key missing.', 'error');
                    setIsRecording(false);
                    return;
                }

                const ai = new GoogleGenAI({ apiKey });
                const response = await ai.models.generateContent({
                    model: 'gemini-2.5-flash',
                    contents: {
                        parts: [
                            { inlineData: { mimeType: 'audio/webm', data: base64 } },
                            { text: "Transcribe this audio exactly." }
                        ]
                    }
                });
                setTranscript(response.text || "No transcription.");
                addLog('AUDIO', 'Transcription complete.', 'success');
                setIsRecording(false);
            };

            mediaRecorder.start();
            setTimeout(() => mediaRecorder.stop(), 5000);

        } catch (err: any) {
            addLog('AUDIO', `Recording failed: ${err.message}`, 'error');
            setIsRecording(false);
        }
    };

    return (
        <div className="flex flex-col h-full bg-ops-900/30">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-ops-800 bg-ops-900/80 backdrop-blur">
                <h2 className="text-xl font-bold font-mono text-ops-text-main tracking-widest">MEDIA<span className="text-red-500">_LAB</span></h2>
                <div className="flex gap-2">
                    <button onClick={() => setTab('video')} className={`px-3 py-1 rounded text-xs font-mono border ${tab === 'video' ? 'bg-red-600 border-red-600 text-white' : 'border-ops-800 text-ops-text-dim'}`}>VEO VIDEO</button>
                    <button onClick={() => setTab('audio')} className={`px-3 py-1 rounded text-xs font-mono border ${tab === 'audio' ? 'bg-orange-600 border-orange-600 text-white' : 'border-ops-800 text-ops-text-dim'}`}>AUDIO OPS</button>
                </div>
            </div>

            <div className="flex-1 p-6 overflow-y-auto">
                {tab === 'video' ? (
                    <div className="grid grid-cols-2 gap-6 h-full">
                        <div className="space-y-4">
                            <div className="bg-ops-900 p-4 border border-ops-800">
                                <label className="block text-xs font-mono text-ops-400 mb-2">TARGET PROMPT</label>
                                <textarea 
                                    value={videoPrompt} onChange={e => setVideoPrompt(e.target.value)}
                                    className="w-full bg-ops-950 border border-ops-800 p-3 text-sm text-ops-text-main h-32 resize-none focus:outline-none focus:border-red-500"
                                    placeholder="Describe video sequence..."
                                />
                            </div>
                            
                            <div className="flex gap-4">
                                <div className="flex-1 bg-ops-900 p-4 border border-ops-800">
                                    <label className="block text-xs font-mono text-ops-400 mb-2">ASPECT RATIO</label>
                                    <select value={videoRatio} onChange={e => setVideoRatio(e.target.value)} className="w-full bg-ops-950 border border-ops-800 text-ops-text-main text-sm p-2">
                                        <option value="16:9">16:9 (Landscape)</option>
                                        <option value="9:16">9:16 (Portrait)</option>
                                    </select>
                                </div>
                                <div className="flex-1 bg-ops-900 p-4 border border-ops-800">
                                    <label className="block text-xs font-mono text-ops-400 mb-2">START IMAGE (OPTIONAL)</label>
                                    <input type="file" accept="image/*" onChange={e => setVideoFile(e.target.files?.[0] || null)} className="text-xs text-ops-text-dim" />
                                </div>
                            </div>
                            
                            <button 
                                onClick={() => handleGenerateVideo()}
                                disabled={videoLoading || !videoPrompt}
                                className="w-full py-4 bg-red-600 hover:bg-red-500 text-white font-bold tracking-widest font-mono flex items-center justify-center gap-2"
                            >
                                {videoLoading ? <Loader2 className="animate-spin" /> : <Video />}
                                RENDER SEQUENCE
                            </button>
                             <div className="flex items-center justify-center gap-2 text-[10px] text-yellow-500">
                                <Key size={10} /> REQUIRES PAID KEY
                            </div>
                        </div>

                        <div className="bg-black border border-ops-800 flex items-center justify-center">
                            {generatedVideoUrl ? (
                                <video src={generatedVideoUrl} controls autoPlay loop className="max-w-full max-h-full" />
                            ) : (
                                <div className="text-center opacity-30">
                                    <Video size={48} className="mx-auto mb-2 text-ops-text-main" />
                                    <p className="font-mono text-xs text-ops-text-main">NO SIGNAL</p>
                                </div>
                            )}
                        </div>
                    </div>
                ) : (
                    <div className="space-y-6 max-w-2xl mx-auto">
                        <div className="bg-ops-900 p-6 border border-ops-800">
                            <h3 className="text-lg font-mono text-ops-text-main mb-4 flex items-center gap-2"><Speaker /> TEXT-TO-SPEECH TRANSMISSION</h3>
                            <textarea 
                                value={ttsText} onChange={e => setTtsText(e.target.value)}
                                className="w-full bg-ops-950 border border-ops-800 p-3 text-sm text-ops-text-main h-24 mb-4"
                                placeholder="Enter message to broadcast..."
                            />
                            <button onClick={() => handleTTS()} className="bg-orange-600 text-white px-4 py-2 font-mono text-sm flex items-center gap-2">
                                <Play size={14} /> TRANSMIT
                            </button>
                        </div>

                         <div className="bg-ops-900 p-6 border border-ops-800">
                            <h3 className="text-lg font-mono text-ops-text-main mb-4 flex items-center gap-2"><Mic /> AUDIO INTERCEPT & TRANSCRIBE</h3>
                            <div className="flex gap-4 items-center mb-4">
                                <button 
                                    onClick={handleTranscribe} 
                                    disabled={isRecording}
                                    className={`w-16 h-16 rounded-full flex items-center justify-center border-2 ${isRecording ? 'bg-red-900 border-red-500 animate-pulse' : 'bg-ops-950 border-ops-800 hover:border-ops-text-main'}`}
                                >
                                    <Mic className={isRecording ? 'text-red-500' : 'text-gray-400'} />
                                </button>
                                <span className="font-mono text-xs text-ops-text-dim">{isRecording ? "RECORDING..." : "CLICK TO RECORD (5s)"}</span>
                            </div>
                            {transcript && (
                                <div className="bg-black p-4 border border-ops-800 text-green-400 font-mono text-sm">
                                    {transcript}
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default MediaLab;