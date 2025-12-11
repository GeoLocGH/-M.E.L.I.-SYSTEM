import React, { useEffect, useRef, useState } from 'react';
import { LogEntry } from '../types';
import { ArrowDown, Pause } from 'lucide-react';

interface LogConsoleProps {
  logs: LogEntry[];
}

const LogConsole: React.FC<LogConsoleProps> = ({ logs }) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [autoScroll, setAutoScroll] = useState(true);

  useEffect(() => {
    if (scrollRef.current && autoScroll) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  return (
    <div className="flex flex-col h-full bg-ops-950 border-t border-ops-800 font-mono text-xs transition-colors duration-300">
      <div className="flex items-center justify-between px-4 py-1 bg-ops-900 border-b border-ops-800 text-ops-400 select-none">
        <div className="flex items-center">
            <span className="mr-2">●</span> SYSTEM LOGS
        </div>
        <button 
            onClick={() => setAutoScroll(!autoScroll)}
            className={`flex items-center gap-1 px-2 py-0.5 rounded border transition-all ${
                autoScroll 
                ? 'border-ops-accent/30 text-ops-accent bg-ops-accent/10 hover:bg-ops-accent/20' 
                : 'border-transparent text-ops-text-dim hover:text-ops-text-main hover:bg-white/5'
            }`}
            title={autoScroll ? "Disable Auto-Scroll" : "Enable Auto-Scroll"}
        >
            {autoScroll ? <ArrowDown size={10} /> : <Pause size={10} />}
            <span className="text-[10px] font-bold tracking-wider">{autoScroll ? "SCROLL: ON" : "SCROLL: PAUSED"}</span>
        </button>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-2 space-y-1">
        {logs.map((log) => (
          <div key={log.id} className="flex gap-3 hover:bg-white/5 px-1 rounded">
            <span className="text-ops-text-dim whitespace-nowrap">[{log.timestamp}]</span>
            <span className={`font-bold whitespace-nowrap w-24 ${
              log.type === 'error' ? 'text-ops-alert' :
              log.type === 'warning' ? 'text-ops-warn' :
              log.type === 'success' ? 'text-green-400' : 'text-ops-accent'
            }`}>{log.source}</span>
            <span className={`${
              log.type === 'error' ? 'text-red-300' : 'text-ops-text-dim'
            }`}>{log.message}</span>
          </div>
        ))}
        {logs.length === 0 && <div className="text-ops-text-dim italic px-2">System initialized. Awaiting input...</div>}
      </div>
    </div>
  );
};

export default LogConsole;