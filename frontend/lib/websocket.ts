type MessageHandler = (data: any) => void;

const MESSAGE_TYPES = {
  PING: 0x01,
  PONG: 0x02,
  SUBSCRIBE: 0x10,
  SUBSCRIBED: 0x11,
  PREDICTION: 0x20,
  ANOMALY: 0x21,
  DATA: 0x30,
  ERROR: 0xFF,
} as const;

export class WSClient {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectAttempts = 0;
  private maxReconnects = 5;
  private handlers: Map<string | number, MessageHandler[]> = new Map();
  private useBinary = false;
  private messageId = 0;
  private pendingRequests: Map<number, { resolve: Function; reject: Function }> = new Map();

  constructor(url?: string) {
    const apiUrl = typeof window !== 'undefined' 
      ? (process.env.NEXT_PUBLIC_API_URL || `${window.location.protocol}//${window.location.host}`)
      : 'http://localhost:8000';
    
    if (url) {
      this.url = url;
    } else {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsHost = process.env.NEXT_PUBLIC_API_URL 
        ? process.env.NEXT_PUBLIC_API_URL.replace(/^http/, 'ws')
        : `${protocol}//${window.location.host}/ws/stream`;
      this.url = wsHost;
    }
  }

  connect() {
    if (this.ws?.readyState === WebSocket.OPEN) return;

    this.ws = new WebSocket(this.url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      this.reconnectAttempts = 0;
      this.send({ type: 'ping' });
    };

    this.ws.onmessage = async (event) => {
      try {
        if (event.data instanceof ArrayBuffer) {
          this.useBinary = true;
          const message = await this.decodeBinaryMessage(event.data);
          this.dispatchMessage(message);
        } else {
          const msg = JSON.parse(event.data);
          this.dispatchMessage(msg);
        }
      } catch (error) {
        console.error('WebSocket message error:', error);
      }
    };

    this.ws.onclose = () => {
      if (this.reconnectAttempts < this.maxReconnects) {
        this.reconnectAttempts++;
        setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  private async decodeBinaryMessage(buffer: ArrayBuffer): Promise<any> {
    const view = new DataView(buffer);
    const length = view.getUint32(0, false);
    
    if (buffer.byteLength > 4) {
      const payload = buffer.slice(4);
      try {
        const msgpack = await import('msgpackr');
        return msgpack.unpack(new Uint8Array(payload));
      } catch {
        const decoder = new TextDecoder();
        return JSON.parse(decoder.decode(payload));
      }
    }
    return {};
  }

  private encodeBinaryMessage(type: number, data: any): ArrayBuffer {
    try {
      const encoder = new TextEncoder();
      const jsonStr = JSON.stringify({ type, ...data });
      const encoded = encoder.encode(jsonStr);
      const header = new Uint8Array(4);
      new DataView(header.buffer).setUint32(0, encoded.length, false);
      
      const result = new Uint8Array(4 + encoded.length);
      result.set(header);
      result.set(encoded, 4);
      return result.buffer;
    } catch {
      const encoder = new TextEncoder();
      const jsonStr = JSON.stringify({ type, data });
      return encoder.encode(jsonStr).buffer;
    }
  }

  private dispatchMessage(msg: any) {
    const msgType = msg.type || msg.t;
    const handlers = this.handlers.get(msgType) || [];
    handlers.forEach((h) => h(msg));
  }

  on(type: string | number, handler: MessageHandler) {
    const existing = this.handlers.get(type) || [];
    existing.push(handler);
    this.handlers.set(type, existing);
  }

  off(type: string | number, handler: MessageHandler) {
    const existing = this.handlers.get(type) || [];
    this.handlers.set(type, existing.filter((h) => h !== handler));
  }

  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      if (this.useBinary && typeof data.type === 'number') {
        this.ws.send(this.encodeBinaryMessage(data.type, data));
      } else {
        this.ws.send(JSON.stringify(data));
      }
    }
  }

  subscribe(depthRange?: { min: number; max: number }) {
    this.send({ type: this.useBinary ? MESSAGE_TYPES.SUBSCRIBE : 'subscribe', depth_range: depthRange });
  }

  disconnect() {
    this.ws?.close();
    this.ws = null;
    this.useBinary = false;
    this.pendingRequests.clear();
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const wsClient = new WSClient();
