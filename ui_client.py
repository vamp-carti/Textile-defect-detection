#!/usr/bin/env python3
import socket
import json
import time
import threading
import sys
import select
import os

class DataClient:
    def __init__(self, host='127.0.0.1', port=9999):
        self.host = host
        self.port = port
        self.socket = None
        self.running = False
        self.data = {}
        self.callbacks = []
        
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            print(f"✅ Connected to data server at {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return False
    
    def start(self):
        if not self.connect():
            return False
        
        self.running = True
        self.thread = threading.Thread(target=self._receive_loop)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def stop(self):
        self.running = False
        if self.socket:
            self.socket.close()
    
    def _receive_loop(self):
        while self.running:
            try:
                self.socket.send(b"GET_DATA\n")
                
                buffer = b""
                while b'\n' not in buffer:
                    chunk = self.socket.recv(4096)
                    if not chunk:
                        break
                    buffer += chunk
                
                if buffer:
                    try:
                        data_str = buffer.decode('utf-8').strip()
                        if data_str:
                            self.data = json.loads(data_str)
                            for callback in self.callbacks:
                                try:
                                    callback(self.data)
                                except:
                                    pass
                    except:
                        pass
                
                time.sleep(0.2)
                
            except:
                print("\n⚠️ Connection lost, reconnecting...")
                time.sleep(2)
                self.socket.close()
                if not self.connect():
                    break
    
    def register_callback(self, callback):
        self.callbacks.append(callback)
    
    def send_command(self, command):
        """Send command via file"""
        try:
            with open("/tmp/minimind_command.txt", "w") as f:
                f.write(command)
            return True
        except Exception as e:
            print(f"Error sending command: {e}")
            return False

class SimpleUI:
    def __init__(self, client):
        self.client = client
        self.last_data = {}
        self.status = "READY"
        self.last_command = ""
        
    def render(self, data):
        if not data:
            return
        
        if data == self.last_data:
            return
        self.last_data = data.copy()
        
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Build display
        lines = []
        lines.append("=" * 60)
        lines.append("MINIMIND DETECTOR v1.0".center(60))
        lines.append("=" * 60)
        lines.append("")
        lines.append("SYSTEM STATUS")
        lines.append(f"  CPU:      {data.get('cpu_usage', 0):.1f}%")
        lines.append(f"  TEMP:     {data.get('gpu_temp', 0):.1f}°C")
        lines.append(f"  MEMORY:   {data.get('memory_usage_mb', 0):.0f} MB")
        lines.append("")
        lines.append("PIPELINE STATISTICS")
        lines.append(f"  FRAMES PROCESSED:  {data.get('frames_processed', 0)}")
        lines.append(f"  FPS:               {data.get('fps', 0):.1f}")
        lines.append(f"  AVG TIME/FRAME:    {data.get('avg_time_ms', 0):.1f} ms")
        lines.append(f"  COMPONENTS:        {data.get('total_components', 0)}")
        lines.append(f"  ROIs:              {data.get('total_rois', 0)}")
        lines.append(f"  INFERENCES:        {data.get('total_inferences', 0)}")
        lines.append(f"  QUEUE:             {data.get('queue_size', 0)} (active: {data.get('active_count', 0)})")
        lines.append("")
        lines.append(f"STATUS:        {data.get('status', 'IDLE')}")
        lines.append("")
        lines.append(f"  DETECTED DEFECTS SAVED IN OUTPUT FOLDER") 
        lines.append("")
        lines.append("-" * 60)
        lines.append("  [S] START  [E] EXPORT  [Q] QUIT")
        lines.append("=" * 60)
        lines.append("")
        if self.last_command:
            lines.append(f"Last command: {self.last_command}")
        lines.append(f"UI Status: {self.status}")
        
        print("\n".join(lines))
    
    def send_command(self, command):
        """Send command and update UI"""
        self.last_command = f"{command} sent"
        self.status = "SENDING..."
        success = self.client.send_command(command)
        if success:
            self.status = f"Command '{command}' sent successfully"
        else:
            self.status = f"Failed to send '{command}'"

def main():
    print("Connecting to C++ DataSender on port 9999...")
    
    client = DataClient(host='127.0.0.1', port=9999)
    ui = SimpleUI(client)
    client.register_callback(ui.render)
    
    if not client.start():
        print("\n❌ Could not connect. Make sure C++ app is running!")
        return
    
    print("\n✅ Connected! Controls:")
    print("  S - START pipeline (pipeline starts paused)")
    print("  E - Export defect CSV")
    print("  Q - Quit\n")
    
    try:
        while client.running:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1).lower()
                
                if key == 'q':
                    print("\nQuitting...")
                    ui.send_command("QUIT")
                    break
                elif key == 's':
                    ui.send_command("START")
                elif key == 'e':
                    ui.send_command("EXPORT")
                
                # Force a redraw
                if hasattr(ui, 'last_data') and ui.last_data:
                    ui.render(ui.last_data)
                    
    except KeyboardInterrupt:
        pass
    
    client.stop()
    print("\nUI stopped.")

if __name__ == "__main__":
    main()
