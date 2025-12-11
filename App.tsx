import React, { useState, useEffect, useRef } from 'react';
import { LayoutDashboard, MessageSquareText, Image, Film, Radio, Globe, Shield, Activity, Database, Lock, Sun, Moon, Power, Wifi, Cpu, ChevronLeft, ChevronRight } from 'lucide-react';
import { ModuleView, LogEntry } from './types';
import LogConsole from './components/LogConsole';
import IntelChat from './components/IntelChat';
import VisualOps from './components/VisualOps';
import MediaLab from './components/MediaLab';
import LiveComms from './components/LiveComms';

declare global {
  interface Window {
    webkitAudioContext: typeof AudioContext;
  }
}

const App: React.FC = () => {
  const [activeView, setActiveView] = useState<ModuleView>(ModuleView.LIVE_COMMS);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [theme, setTheme] = useState<'dark' | 'light'>('dark');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  
  // Simulated stats
  const [threatLevel, setThreatLevel] = useState('NORMAL');
  const [systemLoad, setSystemLoad] = useState(0);

  const channelRef = useRef<BroadcastChannel | null>(null);

  useEffect(() => {
    // Global Mesh Network Listener for Navigation
    const channel = new BroadcastChannel('meli_mesh_network');
    channelRef.current = channel;

    channel.onmessage = (event) => {
      const { type, payload } = event.data;
      if (type === 'CMD_NAVIGATE') {
          if (payload.view && Object.values(ModuleView).includes(payload.view)) {
              setActiveView(payload.view);
              addLog('SYS', `Navigation Override: ${payload.view}`, 'warning');
          }
      }
    };

    const interval = setInterval(() => {
        setSystemLoad(Math.floor(20 + Math.random() * 30));
        // Random "glitch" update
        if (Math.random() > 0.95) {
             addLog('SYS', 'Packet rerouting...', 'info');
        }
    }, 1500);
    return () => {
        clearInterval(interval);
        channel.close();
    };
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'light') {
      root.classList.add('light-theme');
    } else {
      root.classList.remove('light-theme');
    }
  }, [theme]);

  const addLog = (source: string, message: string, type: LogEntry['type'] = 'info') => {
    const newLog: LogEntry = {
      id: Date.now().toString() + Math.random(),
      timestamp: new Date().toLocaleTimeString([], { hour12: false }),
      source,
      message,
      type
    };
    setLogs(prev => [...prev.slice(-49), newLog]); // Keep last 50
  };

  const SidebarItem = ({ view, icon: Icon, label }: { view: ModuleView, icon: any, label: string }) => (
    <button
      onClick={() => setActiveView(view)}
      title={!isSidebarOpen ? label : ''}
      className={`w-full flex items-center ${isSidebarOpen ? 'justify-start px-4 gap-4' : 'justify-center px-0 gap-0'} p-4 mb-1 transition-all duration-300 group relative overflow-hidden border-l-2 ${
        activeView === view 
          ? 'border-ops-accent bg-gradient-to-r from-ops-accent/10 to-transparent text-ops-accent' 
          : 'border-transparent text-ops-text-dim hover:text-white hover:bg-white/5 hover:border-ops-accent/60'
      }`}
    >
      <Icon size={20} className={`relative z-10 transition-transform duration-300 ${activeView === view ? 'scale-110 drop-shadow-[0_0_5px_rgba(0,255,242,0.5)]' : 'group-hover:scale-110 group-hover:text-ops-accent'}`} />
      <span className={`font-mono text-xs font-bold tracking-[0.2em] relative z-10 transition-all duration-300 whitespace-nowrap overflow-hidden group-hover:text-ops-accent ${isSidebarOpen ? 'w-auto opacity-100' : 'w-0 opacity-0'}`}>
        {label}
      </span>
    </button>
  );

  return (
    <div className="flex h-screen w-screen bg-ops-950 overflow-hidden text-ops-text-main selection:bg-ops-accent selection:text-black font-sans transition-colors duration-300">
      
      {/* Background Grid - Global M.E.L. Texture */}
      <div className="absolute inset-0 z-0 opacity-10 pointer-events-none bg-[size:40px_40px] bg-grid-pattern"></div>
      <div className="absolute inset-0 z-0 bg-radial-gradient from-transparent to-ops-950 opacity-80 pointer-events-none"></div>

      {/* Sidebar - M.E.L. style is cleaner, floating feel */}
      <div className={`${isSidebarOpen ? 'w-72' : 'w-20'} bg-black/90 backdrop-blur-md border-r border-ops-800 flex flex-col z-20 shadow-[10px_0_30px_rgba(0,0,0,0.5)] transition-all duration-300 relative`}>
        
        {/* Toggle Button */}
        <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="absolute -right-3 top-8 bg-ops-900 border border-ops-800 text-ops-accent rounded-full p-1 hover:bg-ops-800 hover:border-ops-accent transition-all z-50 shadow-[0_0_10px_rgba(0,0,0,0.5)]"
        >
            {isSidebarOpen ? <ChevronLeft size={14} /> : <ChevronRight size={14} />}
        </button>

        <div className={`p-8 pb-4 flex flex-col gap-2 border-b border-ops-800 ${isSidebarOpen ? 'items-start' : 'items-center px-2'}`}>
          <div className="flex items-center gap-3">
             <div className="relative shrink-0">
                <div className="w-8 h-8 rounded-full border-2 border-ops-accent flex items-center justify-center animate-pulse-fast">
                    <div className="w-2 h-2 bg-ops-accent rounded-full"></div>
                </div>
                <div className="absolute inset-0 border border-ops-accent/50 rounded-full animate-ping"></div>
             </div>
             <div className={`transition-all duration-300 overflow-hidden whitespace-nowrap ${isSidebarOpen ? 'w-auto opacity-100' : 'w-0 opacity-0'}`}>
                <h1 className="text-3xl font-black tracking-tighter text-white font-mono leading-none">M.E.L.I.</h1>
                <p className="text-[9px] font-mono text-ops-accent tracking-[0.3em] opacity-80 mt-1">SYSTEM ONLINE</p>
             </div>
          </div>
        </div>
        
        <div className="flex-1 py-8 space-y-2 overflow-hidden hover:overflow-y-auto custom-scrollbar">
          <SidebarItem view={ModuleView.DASHBOARD} icon={LayoutDashboard} label="OVERVIEW" />
          <SidebarItem view={ModuleView.INTEL} icon={MessageSquareText} label="INTEL_CORE" />
          <SidebarItem view={ModuleView.VISUAL_OPS} icon={Image} label="VISUAL_OPS" />
          <SidebarItem view={ModuleView.MEDIA_LAB} icon={Film} label="MEDIA_LAB" />
          <SidebarItem view={ModuleView.LIVE_COMMS} icon={Radio} label="SECURE_LINK" />
        </div>

        <div className={`p-6 border-t border-ops-800 space-y-4 bg-black/50 overflow-hidden transition-all duration-300 ${isSidebarOpen ? 'opacity-100' : 'opacity-0 h-0 p-0'}`}>
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                     <Cpu size={14} className="text-ops-text-dim" />
                     <span className="text-[10px] font-mono text-ops-text-dim">CPU_LOAD</span>
                </div>
                <span className="text-[10px] font-mono text-ops-accent">{systemLoad}%</span>
            </div>
            <div className="w-full bg-ops-800 h-0.5">
                <div className="h-full bg-ops-accent" style={{width: `${systemLoad}%`}}></div>
            </div>
            
            <div className="flex items-center justify-between text-[10px] font-mono text-ops-500 pt-2">
                <span className="flex items-center gap-1"><Lock size={10} /> ENCRYPTED</span>
                <span>v4.1.0</span>
            </div>
        </div>
        {/* Minimized Footer Icon when closed */}
        {!isSidebarOpen && (
             <div className="p-4 border-t border-ops-800 bg-black/50 flex justify-center animate-in fade-in duration-500">
                <Lock size={16} className="text-ops-800" />
             </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative z-10 min-w-0">
        
        {/* Top Status Bar (HUD style) */}
        <div className="h-12 bg-black/50 border-b border-ops-800 flex items-center justify-between px-6 backdrop-blur-sm z-30">
            <div className="flex items-center gap-6">
                <div className="flex items-center gap-2 text-ops-text-dim text-xs font-mono">
                    <Globe size={12} />
                    <span className="hidden sm:inline">NET_STATUS: <span className="text-green-500">CONNECTED</span></span>
                </div>
                <div className="h-3 w-px bg-ops-800 hidden sm:block"></div>
                <div className="text-ops-text-dim text-xs font-mono tracking-wider hidden sm:block">
                    OP_ID: <span className="text-white">ALPHA-7</span>
                </div>
            </div>
            <div className="flex items-center gap-4">
                 <button onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} className="text-ops-text-dim hover:text-white transition-colors">
                    {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
                 </button>
                 <Power size={14} className="text-ops-alert cursor-pointer hover:text-red-400" />
            </div>
        </div>

        <div className="flex-1 overflow-hidden relative p-6">
          {/* Dashboard is a special case layout, but we can just toggle it like the others */}
          <div className={`h-full w-full flex items-center justify-center relative ${activeView === ModuleView.DASHBOARD ? 'flex' : 'hidden'}`}>
                {/* Central M.E.L. Eye / Visualization */}
                <div className="relative flex items-center justify-center scale-75 sm:scale-100 transition-transform">
                    {/* Outer Rings */}
                    <div className="absolute w-[600px] h-[600px] border border-ops-800 rounded-full opacity-20 animate-[spin_60s_linear_infinite]"></div>
                    <div className="absolute w-[500px] h-[500px] border border-dashed border-ops-800 rounded-full opacity-20 animate-[spin_40s_linear_infinite_reverse]"></div>
                    <div className="absolute w-[400px] h-[400px] border border-ops-500/10 rounded-full animate-pulse"></div>

                    {/* Central Core */}
                    <div className="relative z-10 flex flex-col items-center justify-center bg-black/80 p-12 rounded-full border border-ops-800 backdrop-blur-md shadow-[0_0_50px_rgba(0,255,242,0.1)]">
                        <Activity size={64} className="text-ops-accent mb-4 animate-pulse" />
                        <h2 className="text-4xl font-mono font-bold text-white tracking-[0.2em] mb-2">STANDBY</h2>
                        <div className="flex items-center gap-2 text-ops-accent text-xs font-mono tracking-widest">
                            <span className="w-1.5 h-1.5 bg-ops-accent rounded-full animate-pulse"></span>
                            LISTENING FOR DIRECTIVE
                        </div>
                    </div>

                    {/* Floating HUD Elements */}
                    <div className="absolute -left-64 top-0 tech-border p-4 bg-black/40 backdrop-blur w-48 hidden lg:block">
                        <div className="text-[10px] font-mono text-ops-text-dim mb-1">THREAT ASSESSMENT</div>
                        <div className="text-2xl font-mono text-white tracking-widest">{threatLevel}</div>
                    </div>

                    <div className="absolute -right-64 bottom-0 tech-border p-4 bg-black/40 backdrop-blur w-48 text-right hidden lg:block">
                        <div className="text-[10px] font-mono text-ops-text-dim mb-1">DATA STREAMS</div>
                        <div className="text-2xl font-mono text-ops-accent tracking-widest">ACTIVE</div>
                    </div>
                </div>

                {/* Footer Data Stream */}
                <div className="absolute bottom-0 left-0 right-0 h-12 border-t border-ops-800 flex items-center px-6 gap-8 bg-black/60 backdrop-blur-sm">
                    <div className="flex items-center gap-2 font-mono text-[10px] text-ops-text-dim opacity-50">
                        <Wifi size={10} /> LINK_STABLE
                    </div>
                    <div className="flex-1 font-mono text-[10px] text-ops-500 overflow-hidden whitespace-nowrap opacity-40">
                        0101011010101 SYSTEM_CHECK_COMPLETE ... PROXY_ROUTING ... NODE_SYNC ... ENCRYPTION_KEY_ROTATED ... AWAITING_INPUT
                    </div>
                </div>
          </div>
          
          {/* Main Modules Container - All Mounted, visibility toggled */}
          <div className="h-full w-full tech-border bg-black/40 backdrop-blur-sm overflow-hidden flex flex-col relative rounded-sm">
                 {/* Corner Accents for content area */}
                 <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-ops-accent pointer-events-none z-20"></div>
                 <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-ops-accent pointer-events-none z-20"></div>
                 <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-ops-accent pointer-events-none z-20"></div>
                 <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-ops-accent pointer-events-none z-20"></div>

                <div className={`h-full w-full flex-col ${activeView === ModuleView.INTEL ? 'flex' : 'hidden'}`}>
                    <IntelChat addLog={addLog} />
                </div>
                <div className={`h-full w-full flex-col ${activeView === ModuleView.VISUAL_OPS ? 'flex' : 'hidden'}`}>
                    <VisualOps addLog={addLog} />
                </div>
                <div className={`h-full w-full flex-col ${activeView === ModuleView.MEDIA_LAB ? 'flex' : 'hidden'}`}>
                    <MediaLab addLog={addLog} />
                </div>
                <div className={`h-full w-full flex-col ${activeView === ModuleView.LIVE_COMMS ? 'flex' : 'hidden'}`}>
                    <LiveComms addLog={addLog} />
                </div>
          </div>
        </div>

        {/* Bottom Log Console */}
        <div className="h-48 z-20 shrink-0 border-t border-ops-800 bg-black relative">
          <LogConsole logs={logs} />
        </div>
      </div>
    </div>
  );
};

export default App;