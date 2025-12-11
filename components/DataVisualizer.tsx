import React from 'react';
import {
  LineChart, Line, BarChart, Bar, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts';

interface DataPoint {
  label: string;
  value: number;
}

interface AnalyticsData {
  chartTitle: string;
  chartType: 'line' | 'bar' | 'area';
  data: DataPoint[];
  summary: string;
}

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-ops-900 border border-ops-500 p-2 text-xs font-mono shadow-[0_0_15px_rgba(59,130,246,0.5)]">
        <p className="text-ops-text-main font-bold mb-1 border-b border-ops-800 pb-1">{label}</p>
        <p className="text-ops-400 font-bold">{`Value: ${payload[0].value}`}</p>
      </div>
    );
  }
  return null;
};

const DataVisualizer: React.FC<{ data: AnalyticsData }> = ({ data }) => {
  if (!data || !data.data || data.data.length === 0) return null;

  const renderChart = () => {
    const commonProps = {
      data: data.data,
      margin: { top: 10, right: 30, left: 0, bottom: 0 }
    };

    switch (data.chartType) {
      case 'bar':
        return (
          <BarChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="label" stroke="#94a3b8" fontSize={10} tick={{fill: '#94a3b8'}} />
            <YAxis stroke="#94a3b8" fontSize={10} tick={{fill: '#94a3b8'}} />
            <Tooltip content={<CustomTooltip />} cursor={{fill: '#1e293b'}} />
            <Legend wrapperStyle={{fontSize: '10px', paddingTop: '10px'}} />
            <Bar dataKey="value" fill="#3b82f6" name="Metric Value" radius={[2, 2, 0, 0]} />
          </BarChart>
        );
      case 'area':
        return (
          <AreaChart {...commonProps}>
             <defs>
                <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0}/>
                </linearGradient>
              </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="label" stroke="#94a3b8" fontSize={10} tick={{fill: '#94a3b8'}} />
            <YAxis stroke="#94a3b8" fontSize={10} tick={{fill: '#94a3b8'}} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{fontSize: '10px', paddingTop: '10px'}} />
            <Area type="monotone" dataKey="value" stroke="#06b6d4" fillOpacity={1} fill="url(#colorValue)" name="Trend Volume" />
          </AreaChart>
        );
      case 'line':
      default:
        return (
          <LineChart {...commonProps}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
            <XAxis dataKey="label" stroke="#94a3b8" fontSize={10} tick={{fill: '#94a3b8'}} />
            <YAxis stroke="#94a3b8" fontSize={10} tick={{fill: '#94a3b8'}} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{fontSize: '10px', paddingTop: '10px'}} />
            <Line type="monotone" dataKey="value" stroke="#ef4444" strokeWidth={2} dot={{fill: '#ef4444', r: 4}} activeDot={{r: 6}} name="Data Stream" />
          </LineChart>
        );
    }
  };

  return (
    <div className="w-full mt-4 bg-ops-950/80 border border-ops-800 p-4 rounded-sm relative overflow-hidden group">
      <div className="absolute top-0 right-0 p-2 opacity-20">
        <div className="w-16 h-16 border border-white rounded-full flex items-center justify-center">
            <div className="w-12 h-12 border border-dashed border-white rounded-full animate-spin-slow"></div>
        </div>
      </div>
      
      <div className="mb-4 border-b border-ops-800 pb-2 relative z-10">
        <h4 className="text-sm font-bold text-ops-text-main font-mono uppercase tracking-wider flex items-center gap-2">
            <span className="w-2 h-2 bg-ops-accent rounded-full animate-pulse"></span>
            {data.chartTitle}
        </h4>
        <p className="text-xs text-ops-400 font-mono mt-1">{data.summary}</p>
      </div>
      
      <div className="h-64 w-full relative z-10">
        <ResponsiveContainer width="100%" height="100%">
          {renderChart()}
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default DataVisualizer;