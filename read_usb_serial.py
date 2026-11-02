import serial
import json
import argparse

# Configuration
DEFAULT_PORT = "COM3"  # Change this to whatever port your ESP32 uses
DEFAULT_BAUD = 9600  # ESP32 default is often 115200, Arduino Uno is 9600
BATCH_SIZE = 500       # 2 seconds of data at 250Hz

def main():
    parser = argparse.ArgumentParser(description="Read live ECG from ESP32 over USB")
    parser.add_argument("--port", type=str, default=DEFAULT_PORT, help=f"COM Port (default: {DEFAULT_PORT})")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Baud rate (default: {DEFAULT_BAUD})")
    args = parser.parse_args()

    print(f"Connecting to ESP32 on {args.port} at {args.baud} baud...")
    
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
        status_msg = f"Connected successfully to {args.port} at {args.baud} baud!"
        print(status_msg)
        with open("serial_status.txt", "w") as f:
            f.write(status_msg)
    except Exception as e:
        error_msg = f"Failed to connect to {args.port}. Make sure the Arduino Serial Monitor/Plotter is CLOSED. Error: {str(e)}"
        print(error_msg)
        with open("serial_status.txt", "w") as f:
            f.write(error_msg)
        return

    buffer = []
    
    # Initialize the file as an empty list on startup
    try:
        with open("latest_signal.json", "w") as f:
            json.dump([], f)
    except Exception as e:
        print(f"Warning: Could not initialize latest_signal.json: {e}")
        
    print("Recording... Press Ctrl+C to stop.")
    while True:
        try:
            # Read a line from the USB serial port, ignoring garbage bytes
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line:
                try:
                    # Try to convert it to a number
                    val = float(line)
                    buffer.append(val)
                    
                    # Periodically save the entire accumulated buffer to latest_signal.json
                    # Update every 50 samples (0.2 seconds at 250Hz) for a smooth live feed
                    if len(buffer) % 50 == 0:
                        with open("latest_signal.json", "w") as f:
                            json.dump(buffer, f)
                        print(f"Updated live feed. Total samples: {len(buffer)}")
                        
                except ValueError:
                    # Ignore lines that are just text (like ESP32 boot logs)
                    pass
                    
        except KeyboardInterrupt:
            print("\nStopping recording...")
            break
        except Exception as e:
            error_msg = f"Error reading from serial: {e}"
            print(error_msg)
            with open("serial_status.txt", "w") as f:
                f.write(error_msg)
            break

    ser.close()

if __name__ == "__main__":
    main()
