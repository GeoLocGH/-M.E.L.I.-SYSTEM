export enum ModuleView {
  DASHBOARD = 'DASHBOARD',
  INTEL = 'INTEL',
  VISUAL_OPS = 'VISUAL_OPS',
  MEDIA_LAB = 'MEDIA_LAB',
  LIVE_COMMS = 'LIVE_COMMS'
}

export interface LogEntry {
  id: string;
  timestamp: string;
  source: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  content: string;
  timestamp: number;
  authorId?: string; // ID of the user who sent the message
  groundingMetadata?: any; // For Search/Maps results
  thinking?: boolean; // If it was a thinking model
  analytics?: any; // Structured data for charts
}

export interface SystemStatus {
  cpu: number;
  memory: number;
  network: number;
  status: 'ONLINE' | 'PROCESSING' | 'OFFLINE';
}