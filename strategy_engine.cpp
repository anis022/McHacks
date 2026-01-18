#include <cmath>
#include <deque>
#include <numeric> // For std::accumulate, std::inner_product
#include <algorithm> // For std::max, std::min if needed

extern "C" {


    // Internal state structure (kept on C++ side to minimize passing)
    struct EngineState {
        std::deque<double> price_history;
        double volatility_sq; // Variance
        double risk_aversion; // Gamma
        int window_size;
        // Welford's Algorithm state
        double m2; // Sum of squares of differences from the current mean
        double mean; 
        int count;

        EngineState() : volatility_sq(0.0), m2(0.0), mean(0.0), count(0), flow_start_ts(0.0) {}


         // Helper for flow time if needed, but not strictly required for this logic
        double flow_start_ts;
    };

    // Factory method to create an engine instance
    EngineState* create_engine(int window_size, double gamma) {
        EngineState* engine = new EngineState();
        engine->window_size = window_size;
        engine->risk_aversion = gamma;
        return engine;
    }

    // Destructor to prevent memory leaks
    void destroy_engine(EngineState* engine) {
        if (engine) {
            delete engine;
        }
    }

    // Core update function called on every tick
    // Returns a struct packed into a double array for easy ctypes handling
    // output[0] = reservation_price
    // output[1] = spread
    // output[2] = volatility (sqrt(variance))
    void on_tick(EngineState* engine, double mid_price, int inventory, double time_left, double* output) {
        if (!engine) return;

        // 1. Update History & Volatility (Welford's Algorithm for stability)
        engine->count++;
        double delta = mid_price - engine->mean;
        engine->mean += delta / engine->count;
        double delta2 = mid_price - engine->mean;
        engine->m2 += delta * delta2;

        if (engine->count < 2) {
             engine->volatility_sq = 0.0;
        } else {
             engine->volatility_sq = engine->m2 / (engine->count - 1);
        }
        
        // Use a rolling window reset if count gets too large (optional, to adapt to recent regimes)
        // For a simple competition bot, we might reset periodically or decay.
        // Let's implement a simple history deque for resetting if needed, but Welford is O(1).
        // If we want moving variance, we need to remove old points.
        // Given the constraints and description, let's stick to true Welford for full session or windowed approach. 
        // The prompt describes "Welford's Algorithm for online volatility update", which usually implies cumulative,
        // but for trading, Receding Horizon is better. 
        // Let's stick effectively to the provided blueprint which suggested simple deque for window.
        // Re-implementing strictly as blueprint suggested for windowed variance to be safe with "recent" volatility.

        engine->price_history.push_back(mid_price);
        if ((int)engine->price_history.size() > engine->window_size) {
            engine->price_history.pop_front();
            
            // Recompute variance from scratch on the window to be robust (slower but safer)
            // Or use Welford update/downdate. For N=100-1000, recompute is fast enough in C++.
             double sum = 0.0;
             for(double p : engine->price_history) sum += p;
             double mean = sum / engine->price_history.size();
             double sq_sum = 0.0;
             for(double p : engine->price_history) sq_sum += (p - mean) * (p - mean);
             engine->volatility_sq = sq_sum / (engine->price_history.size() - 1);
        }


        // 2. Avellaneda-Stoikov Calculations
        // r = s - q * gamma * sigma^2 * (T - t)
        // Note: T-t logic needs to be robust. if time_left is large, this term dominates.
        // In the blueprint, it suggested Receding Horizon Control (fixed lookahead).
        // We will assume 'time_left' passed from Python is the effective horizon we want to use.
        
        // Apply a minimum volatility floor to prevent zero-volatility from breaking quotes
        double effective_vol_sq = std::max(engine->volatility_sq, 0.0001); // Min vol = 0.01 (1%)
        
        double reservation_price = mid_price - (inventory * engine->risk_aversion * effective_vol_sq * time_left);
        
        // spread = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
        // Approximating k (liquidity) as constant or deriving from recent fill rate
        // Increased k to 200.0 for tighter spreads in competitive markets
        double k = 200.0; 
        double spread = (engine->risk_aversion * effective_vol_sq * time_left) + 
                        (2.0 / engine->risk_aversion) * std::log(1.0 + (engine->risk_aversion / k));
        
        // Cap spread to be competitive - never wider than 0.5% of mid price
        double max_spread = mid_price * 0.005; // 0.5% of mid
        spread = std::min(spread, max_spread);

        // 3. Pack results
        output[0] = reservation_price;
        output[1] = spread;
        output[2] = std::sqrt(engine->volatility_sq); // Return volatility for debug
    }
}
