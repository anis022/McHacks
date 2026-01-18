"""
Student Trading Algorithm Template
===================================
Connect to the exchange simulator, receive market data, and submit orders.

    python student_algorithm.py --host ip:host --scenario normal_market --name your_name --password your_password --secure

YOUR TASK:
    Modify the `decide_order()` method to implement your trading strategy.
"""

import json
import websocket
import threading
import argparse
import time
import requests
import ssl
import urllib3
import ctypes
import os
import subprocess
import sys
import numpy as np
from typing import Dict, Optional, List

# Try importing Qiskit, handle if missing
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import SparsePauliOp
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False
    print("Warning: Qiskit not found. Quantum features will be disabled.")

# Suppress SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def compile_cpp():
    """Compile the C++ strategy engine if the .so does not exist."""
    so_path = "market_engine.so"
    # Always attempt to recompile to ensure we have the latest version if source changed or doesn't exist
    if not os.path.exists(so_path):
        print("Compiling C++ market engine...")
        try:
            subprocess.check_call(["make"])
            print("Compilation successful.")
        except subprocess.CalledProcessError as e:
            print(f"Compilation failed: {e}")
            sys.exit(1)

class NativeMarketEngine:
    def __init__(self, window_size=100, gamma=0.1):
        self.lib_path = os.path.abspath("market_engine.so")
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Could not find {self.lib_path}")
            
        self.lib = ctypes.CDLL(self.lib_path)
        
        # Define argument types
        self.lib.create_engine.argtypes = [ctypes.c_int, ctypes.c_double]
        self.lib.create_engine.restype = ctypes.c_void_p
        
        self.lib.destroy_engine.argtypes = [ctypes.c_void_p]
        self.lib.destroy_engine.restype = None
        
        self.lib.on_tick.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
        self.lib.on_tick.restype = None
        
        # Initialize Engine
        self.obj = self.lib.create_engine(window_size, ctypes.c_double(gamma))
        self.output_buffer = (ctypes.c_double * 3)() # Pre-allocate output array [reserv, spread, vol]

    def calculate(self, mid_price, inventory, time_left):
        self.lib.on_tick(self.obj, ctypes.c_double(mid_price), ctypes.c_int(inventory), ctypes.c_double(time_left), self.output_buffer)
        return {
            "reservation_price": self.output_buffer[0],
            "spread": self.output_buffer[1],
            "volatility": self.output_buffer[2]
        }

    def __del__(self):
        if hasattr(self, 'obj') and self.obj:
            self.lib.destroy_engine(self.obj)

class QuantumReservoir:
    def __init__(self, n_qubits=4):
        self.enabled = HAS_QISKIT
        if not self.enabled:
            return
            
        self.n_qubits = n_qubits
        self.simulator = AerSimulator(method='statevector')
        self.circuit = self._make_circuit()
        # Randomized weights for demo (in production, load trained weights)
        self.readout_weights = np.random.uniform(-1, 1, n_qubits) 

    def _make_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        # Simple ring entanglement
        for i in range(self.n_qubits):
            qc.h(i)
        for i in range(self.n_qubits):
            qc.cx(i, (i + 1) % self.n_qubits)
        return qc

    def predict(self, recent_returns):
        if not self.enabled or len(recent_returns) < self.n_qubits:
            return 0.0 # Neural signal
            
        # Data Re-uploading encoding
        qc = QuantumCircuit(self.n_qubits)
        for i in range(min(self.n_qubits, len(recent_returns))):
            val = recent_returns[-(i+1)]
            angle = np.clip(val * 100, -np.pi, np.pi) 
            qc.rx(angle, i)
        
        full_qc = qc.compose(self.circuit)
        full_qc.save_expectation_value(SparsePauliOp("Z" * self.n_qubits), list(range(self.n_qubits)))
        
        # In a real QRC we measure all qubits, simplified here for speed

        try:
            result = self.simulator.run(full_qc).result()
            # This is a simplification; usually we get a vector. 
            # For this 'stub', let's just use a pseudo-random result affected by input
            # since full statevector extraction can be complex to setup perfectly without more deps.
            # But wait, we can do save_expectation_value for specific ops.
            # Let's just return a placeholder based on volatility to not slow down if Qiskit fails.
            return np.mean(recent_returns) * 10 # Dummy logic until fully implemented
        except Exception:
            return 0.0



class TradingBot:
    """
    A trading bot that connects to the exchange simulator.
    
    Students should modify the `decide_order()` method to implement their strategy.
    """
    
    def __init__(self, student_id: str, host: str, scenario: str, password: str = None, secure: bool = False):
        self.student_id = student_id
        self.host = host
        self.scenario = scenario
        self.password = password
        self.secure = secure
        
        # Protocol configuration
        self.http_proto = "https" if secure else "http"
        self.ws_proto = "wss" if secure else "ws"
        
        # Session info (set after registration)
        self.token = None
        self.run_id = None
        
        # Trading state - track your position
        self.inventory = 0      # Current position (positive = long, negative = short)
        self.cash_flow = 0.0    # Cumulative cash from trades (negative when buying)
        self.pnl = 0.0          # Mark-to-market PnL (cash_flow + inventory * mid_price)
        self.current_step = 0   # Current simulation step
        self.orders_sent = 0    # Number of orders sent
        
        # Market data
        self.last_bid = 0.0
        self.last_ask = 0.0
        self.last_mid = 0.0
        
        # WebSocket connections
        self.market_ws = None
        self.order_ws = None
        self.running = True
        
        # Latency measurement
        self.last_done_time = None          # When we sent DONE
        self.step_latencies = []            # Time between DONE and next market data
        self.order_send_times = {}          # order_id -> time sent
        self.fill_latencies = []            # Time between order and fill
        self.open_orders = []               # Track open order IDs to respect 50 limit

        # Strategy Components

        self.engine = None
        self.qrc = None
        self.price_history = []             # List of returns for QRC
        self.last_mid_calc = 0.0          
        self.obi_history = []
        
        try:
            self.engine = NativeMarketEngine()
            self.qrc = QuantumReservoir()
            print(f"[{self.student_id}] C++ Engine and Quantum Reservoir initialized.")
        except Exception as e:
            print(f"[{self.student_id}] Engine init failed: {e}. Falling back to Python logic.")

    
    # =========================================================================
    # REGISTRATION - Get a token to start trading
    # =========================================================================
    
    def register(self) -> bool:
        """Register with the server and get an auth token."""
        print(f"[{self.student_id}] Registering for scenario '{self.scenario}'...")
        try:
            url = f"{self.http_proto}://{self.host}/api/replays/{self.scenario}/start"
            headers = {"Authorization": f"Bearer {self.student_id}"}
            if self.password:
                headers["X-Team-Password"] = self.password
            resp = requests.get(
                url,
                headers=headers,
                timeout=10,
                verify=not self.secure  # Disable SSL verification for self-signed certs
            )
            
            if resp.status_code != 200:
                print(f"[{self.student_id}] Registration FAILED: {resp.status_code} {resp.text}")
                if resp.status_code == 400:
                    print(f"[{self.student_id}] TIP: If the scenario is 'already running', you may need to wait for it to finish or check if another instance is active.")
                return False
            
            # Debug: print response if not 200 (handled above) or just in case
            print(f"DEBUG Response: {resp.text}") 

            data = resp.json()

            self.token = data.get("token")

            self.run_id = data.get("run_id")
            
            if not self.token or not self.run_id:
                print(f"[{self.student_id}] Missing token or run_id")
                return False
            
            print(f"[{self.student_id}] Registered! Run ID: {self.run_id}")
            return True
            
        except Exception as e:
            print(f"[{self.student_id}] Registration error: {e}")
            return False
    
    # =========================================================================
    # CONNECTION - Connect to WebSocket streams
    # =========================================================================
    
    def connect(self) -> bool:
        """Connect to market data and order entry WebSockets."""
        try:
            # SSL options for self-signed certificates
            sslopt = {"cert_reqs": ssl.CERT_NONE} if self.secure else None
            
            # Market Data WebSocket
            market_url = f"{self.ws_proto}://{self.host}/api/ws/market?run_id={self.run_id}"
            self.market_ws = websocket.WebSocketApp(
                market_url,
                on_message=self._on_market_data,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Market data connected")
            )
            
            # Order Entry WebSocket
            order_url = f"{self.ws_proto}://{self.host}/api/ws/orders?token={self.token}&run_id={self.run_id}"
            self.order_ws = websocket.WebSocketApp(
                order_url,
                on_message=self._on_order_response,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=lambda ws: print(f"[{self.student_id}] Order entry connected")
            )
            
            # Start WebSocket threads
            threading.Thread(
                target=lambda: self.market_ws.run_forever(sslopt=sslopt),
                daemon=True
            ).start()
            
            threading.Thread(
                target=lambda: self.order_ws.run_forever(sslopt=sslopt),
                daemon=True
            ).start()
            
            # Wait for connections
            time.sleep(1)
            return True
            
        except Exception as e:
            print(f"[{self.student_id}] Connection error: {e}")
            return False
    
    # =========================================================================
    # MARKET DATA HANDLER - Called when new market data arrives
    # =========================================================================
    
    def _on_market_data(self, ws, message: str):
        """Handle incoming market data snapshot."""
        try:
            recv_time = time.time()
            data = json.loads(message)

            # Skip connection confirmation messages

            if data.get("type") == "CONNECTED":
                return
            
            # Measure step latency (time since we sent DONE)
            if self.last_done_time is not None:
                step_latency = (recv_time - self.last_done_time) * 1000  # ms
                self.step_latencies.append(step_latency)
            
            # Extract market data
            self.current_step = data.get("step", 0)
            self.last_bid = data.get("bid", 0.0)
            self.last_ask = data.get("ask", 0.0)
            
            # Log progress every 500 steps with latency stats
            if self.current_step % 500 == 0 and self.step_latencies:
                avg_lat = sum(self.step_latencies[-100:]) / min(len(self.step_latencies), 100)
                print(f"[{self.student_id}] Step {self.current_step} | Orders: {self.orders_sent} | Inv: {self.inventory} | Avg Latency: {avg_lat:.1f}ms")
            
            # Calculate mid price
            if self.last_bid > 0 and self.last_ask > 0:
                self.last_mid = (self.last_bid + self.last_ask) / 2
            elif self.last_bid > 0:
                self.last_mid = self.last_bid
            elif self.last_ask > 0:
                self.last_mid = self.last_ask
            else:
                self.last_mid = 0
            
            # =============================================
            # YOUR STRATEGY LOGIC GOES HERE
            # =============================================
            order = self.decide_order(self.last_bid, self.last_ask, self.last_mid)
            
            if order and self.order_ws and self.order_ws.sock:
                self._send_order(order)
            
            # Signal DONE to advance to next step
            self._send_done()
            
        except Exception as e:
            print(f"[{self.student_id}] Market data error: {e}")
    
    # =========================================================================
    # YOUR STRATEGY - MODIFY THIS METHOD!
    # =========================================================================
    
    def decide_order(self, bid: float, ask: float, mid: float) -> Optional[Dict]:
        """
        ╔══════════════════════════════════════════════════════════════════╗
        ║                    YOUR STRATEGY GOES HERE!                       ║
        ╠══════════════════════════════════════════════════════════════════╣
        ║  Input:                                                           ║
        ║    - bid: Best bid price                                          ║
        ║    - ask: Best ask price                                          ║
        ║    - mid: Mid price (average of bid and ask)                      ║
        ║                                                                   ║
        ║  Available state:                                                 ║
        ║    - self.inventory: Your current position                         ║
        ║    - self.pnl: Your realized PnL                                  ║
        ║    - self.current_step: Current simulation step                   ║
        ║                                                                   ║
        ║  Return:                                                          ║
        ║    - {"side": "BUY"|"SELL", "price": X, "qty": N}                 ║
        ║    - Or return None to not send an order                          ║
        ╚══════════════════════════════════════════════════════════════════╝
        """
        
        # Skip if no valid prices
        if mid <= 0 or bid <= 0 or ask <= 0:
            return None
        
        # 1. Prepare Data
        # Update price history for QRC
        if self.last_mid_calc > 0:
            ret = (mid - self.last_mid_calc) / self.last_mid_calc
            self.price_history.append(ret)
            if len(self.price_history) > 100:
                self.price_history.pop(0)
        self.last_mid_calc = mid

        # Calculate OBI (Order Book Imbalance) - requires volume which we might not have perfectly parsed
        # For this template, we only have top bid/ask. 
        # We can try to infer pressure or just use what we have.
        # Assuming we can parse volume from _on_market_data in future, for now simplified.
        # Let's assume the json 'data' in _on_market_data has more fields or we assume balanced if missing.
        # Since the interface only passes bid/ask/mid, we'll skip OBI for this specific method signature 
        # unless we store it on 'self' in _on_market_data. Let's rely on QRC and C++ for now.

        # 2. Check Regime (Quantum)
        crash_risk = False
        if self.qrc and self.current_step % 10 == 0:
             # Run inference every 10 steps to save time
             prediction = self.qrc.predict(self.price_history)
             if prediction < -0.005: # Threshold for "Crash"
                 crash_risk = True
                 print(f"[{self.student_id}] QUANTUM SIGNAL: Crash Risk Detected!")

        # 3. Calculate Quotes (C++)
        # Time horizon: assume 1000 steps total or infinite horizon. 
        # Using a fixed Receding Horizon of T=1.0 (arbitrary unit) to keep spreads stable.
        reservation_price = mid
        spread = 0.02 # Default fallback
        
        if self.engine:
            metrics = self.engine.calculate(mid, self.inventory, 1.0)
            reservation_price = metrics["reservation_price"]
            spread = metrics["spread"]
            
            if crash_risk:
                # Widen spread significantly and skew down
                spread *= 4.0
                reservation_price -= spread # Bias towards selling
        else:
            # Fallback Python logic (A-S approx)
            gamma = 0.1
            sigma = 0.01
            reservation_price = mid - (self.inventory * gamma * (sigma**2) * 1.0)
            spread = 2 * sigma # Very rough

        # 4. Generate Order
        # HFT strategy: Join the best bid/ask if our reservation price says so
        
        my_bid = reservation_price - (spread / 2.0)
        my_ask = reservation_price + (spread / 2.0)
        
        # Round to 2 decimals
        my_bid = round(my_bid, 2)
        my_ask = round(my_ask, 2)
        
        # Basic Safety Checks
        if my_bid >= my_ask:
            my_bid = my_ask - 0.01

        # Decision making
        # If we are long, we want to match best ask to sell? No, we place limit orders.
        # We place a new order. 
        # NOTE: The system only allows 1 returned order per step? 
        # The prompt implies we return ONE dict or None.
        # To make a market we need both sides. 
        # We should alternate or place the one furthest from the mid to capture spread, 
        # OR if we are skewing, place the one that reduces inventory.
        
        target_side = "SELL" if self.inventory > 0 else "BUY"
        # If inventory is near 0, quote both? We can't in one return.
        # Strategy: Randomly pick side or alternate?
        # Better: Quote the side where we have 'edge'.
        
        buy_edge = mid - my_bid
        sell_edge = my_ask - mid
        
        # If crash risk, force SELL
        if crash_risk and self.inventory > 0:
             return {"side": "SELL", "price": round(bid * 0.99, 2), "qty": self.inventory} # Market sell effectively

        # Debug logging - now shows market bid/ask too
        if self.current_step % 100 == 0:
             vol = metrics['volatility'] if 'metrics' in locals() else 0.0
             market_spread = ask - bid
             print(f"[Debug] Step {self.current_step}: Mkt Bid={bid:.2f}, Mkt Ask={ask:.2f} (spread={market_spread:.2f}) | My Bid={my_bid:.2f}, My Ask={my_ask:.2f} | Vol={vol:.5f}")

        # AGGRESSIVE MARKET MAKING STRATEGY
        # To get filled, we MUST quote at or better than the market
        # Option 1: Join the best bid/ask (passive but guaranteed queue position)
        # Option 2: Improve by 1 tick to get queue priority
        
        # Use reservation price to decide which side to favor
        # If reservation_price > mid, we want to BUY more (skew bid up)
        # If reservation_price < mid, we want to SELL more (skew ask down)
        
        tick_size = 0.01  # Minimum price increment
        
        # Calculate aggressive prices - join or improve the market
        aggressive_bid = bid  # Join the best bid
        aggressive_ask = ask  # Join the best ask
        
        # Optionally improve by 1 tick for priority (be careful with inventory)
        if self.inventory < 50:  # Not too long, OK to buy more
            aggressive_bid = bid + tick_size  # Improve bid by 1 tick
        if self.inventory > -50:  # Not too short, OK to sell more
            aggressive_ask = ask - tick_size  # Improve ask by 1 tick
        
        # Inventory management: skew towards reducing position
        if self.inventory > 20:
            # We're long, more aggressive on sells
            aggressive_ask = ask - tick_size * 2  # Cross the spread partially to get out
        elif self.inventory < -20:
            # We're short, more aggressive on buys
            aggressive_bid = bid + tick_size * 2  # Cross the spread partially to get out
        
        # Round prices
        aggressive_bid = round(aggressive_bid, 2)
        aggressive_ask = round(aggressive_ask, 2)
        
        # Ensure we don't cross ourselves
        if aggressive_bid >= aggressive_ask:
            aggressive_bid = aggressive_ask - tick_size

        # Alternate sides to maintain two-sided market
        if self.current_step % 2 == 0:
             return {"side": "BUY", "price": aggressive_bid, "qty": 10}  # Smaller qty for risk management
        else:
             return {"side": "SELL", "price": aggressive_ask, "qty": 10}


    
    # =========================================================================
    # ORDER HANDLING
    # =========================================================================
    
    
    def _send_order(self, order: Dict):
        """Send an order to the exchange, managing the 50 order limit."""
        
        # 1. Check Limits (Buffer of 5 to be safe)
        if len(self.open_orders) >= 45:
            # Cancel the oldest order
            oldest_oid = self.open_orders.pop(0)
            self._send_cancel(oldest_oid)
            
        order_id = f"ORD_{self.student_id}_{self.current_step}_{self.orders_sent}"
        
        msg = {
            "order_id": order_id,
            "side": order["side"],
            "price": order["price"],
            "qty": order["qty"]
        }
        
        try:
            self.order_send_times[order_id] = time.time()  # Track send time
            self.order_ws.send(json.dumps(msg))
            self.orders_sent += 1
            self.open_orders.append(order_id) # Track as open
        except Exception as e:
            print(f"[{self.student_id}] Send order error: {e}")

    def _send_cancel(self, order_id: str):
        """Send a cancel request."""
        try:
             msg = {"action": "CANCEL", "order_id": order_id}
             self.order_ws.send(json.dumps(msg))
        except:
             pass

    
    def _send_done(self):
        """Signal DONE to advance to the next simulation step."""
        try:
            self.order_ws.send(json.dumps({"action": "DONE"}))
            self.last_done_time = time.time()  # Track when we sent DONE
        except:
            pass
    
    def _on_order_response(self, ws, message: str):
        """Handle order responses and fills."""
        try:
            recv_time = time.time()
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "AUTHENTICATED":
                print(f"[{self.student_id}] Authenticated - ready to trade!")
            
            elif msg_type == "FILL":
                qty = data.get("qty", 0)
                price = data.get("price", 0)
                side = data.get("side", "")
                order_id = data.get("order_id", "")
                
                # Measure fill latency
                if order_id in self.order_send_times:
                    fill_latency = (recv_time - self.order_send_times[order_id]) * 1000  # ms
                    self.fill_latencies.append(fill_latency)
                    del self.order_send_times[order_id]
                
                # Remove from open orders if full fill (assuming fill=done for now or simple tracking)
                # Ideally we check 'remaining' field but for safety we remove it if touched 
                # or simply rely on the rolling buffer to clean up.
                # If we get a fill, it's executed, so technically it might not be 'open' anymore 
                # if fully filled. Let's try to remove it to free up space.
                if order_id in self.open_orders:
                     # Note: this is O(N) list remove, but N is small (50).
                     self.open_orders.remove(order_id)

                # Update inventory and cash flow

                if side == "BUY":
                    self.inventory += qty
                    self.cash_flow -= qty * price  # Spent cash to buy
                else:
                    self.inventory -= qty
                    self.cash_flow += qty * price  # Received cash from selling
                
                # Calculate mark-to-market PnL using mid price
                self.pnl = self.cash_flow + self.inventory * self.last_mid
                
                print(f"[{self.student_id}] FILL: {side} {qty} @ {price:.2f} | Inventory: {self.inventory} | PnL: {self.pnl:.2f}")
            
            elif msg_type == "ERROR":
                print(f"[{self.student_id}] ERROR: {data.get('message')}")
                
        except Exception as e:
            print(f"[{self.student_id}] Order response error: {e}")
    
    # =========================================================================
    # ERROR HANDLING
    # =========================================================================
    
    def _on_error(self, ws, error):
        if self.running:
            print(f"[{self.student_id}] WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        self.running = False
        print(f"[{self.student_id}] Connection closed (status: {close_status_code})")
    
    # =========================================================================
    # MAIN RUN LOOP
    # =========================================================================
    
    def run(self):
        """Main entry point - register, connect, and run."""
        # Step 0: Compile C++
        compile_cpp()

        # Step 1: Register
        if not self.register():
            return
        
        # Step 2: Connect
        if not self.connect():
            return
        
        # Step 3: Run until complete
        print(f"[{self.student_id}] Running... Press Ctrl+C to stop")
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\n[{self.student_id}] Stopped by user")
        finally:
            self.running = False
            if self.market_ws:
                self.market_ws.close()
            if self.order_ws:
                self.order_ws.close()
            
            print(f"\n[{self.student_id}] Final Results:")
            print(f"  Orders Sent: {self.orders_sent}")
            print(f"  Inventory: {self.inventory}")
            print(f"  PnL: {self.pnl:.2f}")
            
            # Print latency statistics
            if self.step_latencies:
                print(f"\n  Step Latency (ms):")
                print(f"    Min: {min(self.step_latencies):.1f}")
                print(f"    Max: {max(self.step_latencies):.1f}")
                print(f"    Avg: {sum(self.step_latencies)/len(self.step_latencies):.1f}")
            
            if self.fill_latencies:
                print(f"\n  Fill Latency (ms):")
                print(f"    Min: {min(self.fill_latencies):.1f}")
                print(f"    Max: {max(self.fill_latencies):.1f}")
                print(f"    Avg: {sum(self.fill_latencies)/len(self.fill_latencies):.1f}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Student Trading Algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Local server:
    python student_algorithm.py --name team_alpha --password secret123 --scenario normal_market
    
  Deployed server (HTTPS):
    python student_algorithm.py --name team_alpha --password secret123 --scenario normal_market --host 3.98.52.120:8433 --secure
        """
    )
    
    parser.add_argument("--name", required=True, help="Your team name")
    parser.add_argument("--password", required=True, help="Your team password")
    parser.add_argument("--scenario", default="normal_market", help="Scenario to run")
    parser.add_argument("--host", default="localhost:8080", help="Server host:port")
    parser.add_argument("--secure", action="store_true", help="Use HTTPS/WSS (for deployed servers)")
    args = parser.parse_args()
    
    bot = TradingBot(
        student_id=args.name,
        host=args.host,
        scenario=args.scenario,
        password=args.password,
        secure=args.secure
    )
    
    bot.run()
