# SimPy Tutorial: Queue Simulator Walkthrough

SimPy is a **discrete-event simulation** library. Instead of stepping through time tick-by-tick, it jumps directly from one event to the next, making it very efficient.

This document is both a SimPy primer and a mapping of SimPy concepts to their concrete usage in `queue_simulator.py`. If you are new to the project, read top-to-bottom. If you already know SimPy, skip to [How SimPy Is Used In This Project](#how-simpy-is-used-in-this-project).

---

## Core SimPy Concepts

### 1. The Environment

Everything in SimPy revolves around an `Environment` -- the simulation clock and event scheduler:

```python
import simpy

env = simpy.Environment()
print(env.now)  # Current simulation time (starts at 0)
```

### 2. Processes (Generators)

SimPy processes are **Python generators** that `yield` events. When a process yields, it pauses until that event completes:

```python
def my_process(env):
    print(f"Start at {env.now}")
    yield env.timeout(5)  # Wait 5 time units
    print(f"Resume at {env.now}")  # Prints "Resume at 5"
```

The `yield` keyword is the magic -- it tells SimPy "pause here until this event happens."

### 3. Store (FIFO Queue Primitive)

A `Store` is SimPy's built-in unbounded FIFO. It supports `put()` and `get()` which are **awaitable events**:

```python
queue = simpy.Store(env)

# Producer
yield queue.put(item)    # Blocks if store has a capacity limit

# Consumer
item = yield queue.get() # Blocks until an item is available
```

### 4. Running the Simulation

```python
env = simpy.Environment()
env.process(my_process(env))  # Register the process
env.run(until=100)            # Run until time=100
```

**Key insight**: SimPy only advances time when _every_ registered process is blocked on a yield. Between yields, Python code runs synchronously.

---

## How SimPy Is Used In This Project

All SimPy usage lives in **`queue_simulator.py`**, specifically inside `IntegratedQueueSimulator`. The rest of the codebase (model training, RAT selection, etc.) has no SimPy dependency. SimPy serves as the event loop that ties the packet queue, the DTMC packet sizer, the RAT selector, and the PHY-layer model together into a time-coherent simulation.

### Overall Architecture

```
                          IntegratedQueueSimulator
 ┌──────────────────────────────────────────────────────────────────────┐
 │                                                                      │
 │  ┌──────────────┐   simpy.Store   ┌──────────────────┐              │
 │  │  _packet_     │──── put() ────>│  _packet_         │              │
 │  │  generator    │    (FIFO)      │  processor        │              │
 │  │  (producer)   │<── get() ─────│  (consumer)       │              │
 │  └──────────────┘                 └────────┬─────────┘              │
 │    yield timeout                           │                         │
 │    (TX_INTERVAL)                           │                         │
 │                           ┌────────────────┼──────────────┐         │
 │                           v                v              v         │
 │                    QueueSimulator   DTMCPacketSizer   PHY helpers   │
 │                    (byte budget,    (Markov chain     (TX capacity, │
 │                     depth track)     transitions)      latency)     │
 │                                                                      │
 └──────────────────────────────────────────────────────────────────────┘
```

Two SimPy processes run concurrently inside one `simpy.Environment`:

| Process | Role | Primary yield points |
|---------|------|---------------------|
| `_packet_generator` | Creates packets at a fixed rate | `yield env.timeout(interval)` and `yield queue.put(packet)` |
| `_packet_processor` | Dequeues, selects RAT, sizes packet, simulates TX | `yield queue.get()` and `yield env.timeout(latency / 1000.0)` |

Everything else (DTMC transitions, PHY capacity checks, PDR correction) runs synchronously between yields -- no extra SimPy events required.

---

### Environment Creation and Process Registration

From `IntegratedQueueSimulator.run()` (line 1174):

```python
# Create SimPy environment
self.env = simpy.Environment()
queue = simpy.Store(self.env)

# Start processes
self.env.process(self._packet_generator(self.env, queue))
self.env.process(self._packet_processor(self.env, queue))

# Run simulation
if duration is not None:
    self.env.run(until=duration)
```

The `simpy.Store` here is **unbounded** at the SimPy level. Queue bounding is enforced manually by `QueueSimulator.can_enqueue()` before calling `queue.put()`, because the capacity depends on the current RAT and packet size (both of which change over time). SimPy's built-in `capacity` parameter is a fixed integer, so it cannot model this.

---

### Process 1: Packet Generator (`_packet_generator`)

Source: `queue_simulator.py`, line 992.

This generator creates packets at a fixed rate derived from `TX_INTERVAL_MS` (default 100 ms = 10 Hz). Each iteration:

1. **Yields a timeout** equal to the inter-arrival interval.
2. **Checks PHY-layer queue capacity** -- if the byte budget is exceeded, the packet is dropped (never enters the Store).
3. **Yields a put** into the SimPy Store.

```python
def _packet_generator(self, env: simpy.Environment, queue: simpy.Store):
    interval = 1.0 / self.arrival_rate_hz  # seconds between packets
    packet_id = 0

    while not self._stop_simulation:
        yield env.timeout(interval)              # (1) pace arrivals
        if self._stop_simulation:
            break

        estimated_size = self.base_packet_size

        # (2) PHY-layer bounded queue check
        if self.enforce_phy_limits and not self.queue_sim.can_enqueue(estimated_size):
            self.queue_sim.metrics.dropped_packets += 1
            packet_id += 1
            continue                              # drop, do not put

        packet = {
            "arrival_time": env.now,
            "id": packet_id,
            "estimated_size": estimated_size,
        }
        yield queue.put(packet)                   # (3) enqueue
        self.queue_sim.queue_depth += 1
        self.queue_sim.queue_bytes += estimated_size
        packet_id += 1
```

The `can_enqueue` check (line 1006) queries `QueueSimulator`, which computes the byte-level capacity from PHY parameters:

```python
# QueueSimulator.can_enqueue() -- line 659
capacity_bytes = calculate_queue_capacity_bytes(target_rat, TX_INTERVAL_MS)
return (self.queue_bytes + packet_size) <= capacity_bytes
```

The capacity itself comes from a resource-block formula in the PHY helpers (see [PHY-Layer Capacity Model](#phy-layer-capacity-model-and-simpy) below).

---

### Process 2: Packet Processor (`_packet_processor`)

Source: `queue_simulator.py`, line 1025.

This generator is the core simulation loop. Each iteration:

1. **Yields a get** from the Store -- blocks until a packet is available.
2. Reads the next network state from the input DataFrame (`_get_network_state`). If the data is exhausted, sets `_stop_simulation = True` and breaks.
3. Gets a RAT decision (from the API or from pre-computed CSV columns).
4. Calls `QueueSimulator.decide_packet_size()` which runs the **DTMC transition** (see next section).
5. Simulates transmission outcome (success/failure, latency, TX time).
6. Records the outcome into the DTMC's moving-window PDR history via `update_pdr_estimate()`.
7. **Yields a timeout** for the transmission latency, modeling the time the channel is occupied.

```python
def _packet_processor(self, env: simpy.Environment, queue: simpy.Store):
    previous_rat = None
    tx_times: List[float] = []

    while True:
        packet = yield queue.get()                       # (1) wait for packet
        # ... update queue depth/bytes ...

        state = self._get_network_state(env.now)         # (2) next data row
        if state is None:
            self._stop_simulation = True
            break

        rat_decision = self._get_rat_decision(state)     # (3) RAT selection
        selected_rat = rat_decision.selected_rat

        packet_decision = self.queue_sim.decide_packet_size(  # (4) DTMC
            rat_decision, env.now
        )
        packet_size = packet_decision.packet_size_bytes

        success, latency, tx_time = self._simulate_transmission(  # (5)
            selected_rat, packet_size, rat_decision.predicted_pdr,
        )

        outcome = TransmissionOutcome(                   # (6) feedback
            timestamp_ms=int(env.now * 1000),
            rat_used=selected_rat,
            packet_size_bytes=packet_size,
            actual_latency_ms=latency,
            delivered=success,
            network_state=state,
        )
        self.queue_sim.update_pdr_estimate(outcome)

        yield env.timeout(latency / 1000.0)              # (7) TX delay
```

The processor's `yield queue.get()` at (1) and `yield env.timeout(...)` at (7) are the only SimPy interactions. Everything in between is plain Python.

---

### Simulation Termination

The simulation can end in three ways:

1. **Duration limit**: `env.run(until=duration)` stops the clock.
2. **Data exhaustion**: `_get_network_state()` returns `None`, the processor sets `self._stop_simulation = True` and breaks. The generator checks this flag after its next timeout and also breaks.
3. **StopSimulation exception**: caught in the `run()` method as a fallback.

The `_stop_simulation` flag is the primary coordination mechanism between the two processes. It is not a SimPy event -- it is a plain boolean checked after each yield returns.

---

## DTMC Packet Sizer and SimPy

The `DTMCPacketSizer` class (line 249) is **not** a SimPy process. It is a stateful object that is called synchronously from inside the processor process. SimPy provides the clock; the DTMC provides the state machine.

### State Diagram

```
[1024 B] <--PDR<0.95-- [2048 B] <--PDR<0.95-- [3072 B] <--PDR<0.95-- [4096 B]
    |                      |                      |                      |
    +----PDR>0.99-------->-+----PDR>0.99-------->-+----PDR>0.99-------->-+
```

Size levels are defined in `config.py` as `DTMC_PACKET_SIZES = [1024, 2048, 3072, 4096]`.

### How the DTMC Uses Simulation Time

The DTMC maintains a **moving-window PDR** over recent transmission outcomes. Each outcome is timestamped with `env.now` (converted to seconds), and old entries outside the window (`DTMC_WINDOW_SECONDS = 1.0s`) are pruned.

The call chain per packet inside the processor:

```
_packet_processor
  --> queue_sim.decide_packet_size(rat_decision, env.now)
        --> dtmc.get_window_pdr(current_time)       # uses env.now for window cutoff
        --> dtmc.transition(pdr, step)               # deterministic up/down/stay
        --> enforce PHY max packet size
  --> _simulate_transmission(rat, packet_size, predicted_pdr)
  --> queue_sim.update_pdr_estimate(outcome)
        --> dtmc.record_outcome(timestamp, delivered) # stores (env.now, bool)
```

The DTMC's `transition()` method (line 356) is deterministic:

```python
def transition(self, pdr: float, step: int = 0) -> int:
    if pdr > self.threshold_high and self.current_state < self.num_levels - 1:
        self.current_state += 1   # increase packet size
    elif pdr < self.threshold_low and self.current_state > 0:
        self.current_state -= 1   # decrease packet size
    # else: stay
    return self.current_size
```

Thresholds are configurable in `config.py`:
- `DTMC_THRESHOLD_HIGH = 0.99` -- PDR above this triggers an increase.
- `DTMC_THRESHOLD_LOW = 0.95` -- PDR below this triggers a decrease.

### PDR Feedback Loop

When the DTMC has insufficient history (fewer than 5 samples in the window), it falls back to the **predicted PDR** from the RAT selector, corrected for the current packet size using a power-law model:

```python
# correct_pdr_for_packet_size() -- line 417
effective_exponent = (target_size / base_size) ** correction_exponent
corrected_pdr = base_pdr ** effective_exponent
```

This correction exists because the LSTM/GRU/RNN models were trained on fixed 1 kB packets. Larger packets have more bits that can fail, so PDR drops.

Once enough real outcomes accumulate in the moving window, the DTMC switches to using the actual observed PDR, making it self-correcting.

---

## PHY-Layer Capacity Model and SimPy

The PHY helpers (lines 57-243 in `queue_simulator.py`) compute TX capacity and queue bounds per RAT. They are pure functions, not SimPy processes. They feed into the simulation at two points:

1. **Before enqueue** (generator process): `can_enqueue()` checks if adding a packet would exceed the byte budget.
2. **During TX simulation** (processor process): `calculate_tx_time_ms()` determines how long transmission takes, which becomes the `env.timeout()` duration.

### Capacity Formula

For 5G and PC5 (resource-block based):

```
dSF = NSC x Nsym x NRB x Rmod x CR    (bits per subframe)
```

For DSRC (data-rate based):

```
bits_per_ms = data_rate_mbps x 1e6 / 1000
```

TX capacity per interval = `bits_per_ms x TX_INTERVAL_MS / 8` (bytes).

Queue capacity = TX capacity x `queue_multiplier` (default 2, from `config.py`).

### PHY Constants (from `config.py`)

| RAT | Bandwidth | NRB | Modulation | Coding Rate | Base Latency |
|-----|-----------|-----|------------|-------------|--------------|
| 5G NR | 20 MHz | 106 | 64QAM (6) | 0.66 | 15 ms |
| PC5 | 10 MHz | 10/subchannel | QPSK (2) | 0.5 | 8 ms |
| DSRC | 10 MHz | n/a (6 Mbps) | QPSK (2) | 0.5 | 5 ms |

The `--no-phy-limits` CLI flag sets `enforce_phy_limits=False`, which bypasses all capacity checks and lets the queue grow unbounded. This is useful for comparing ideal vs. realistic behavior.

---

## SimPy Timeline Example

Trace with 10 Hz arrival rate (default `TX_INTERVAL_MS=100`):

```
Time (s)  Event
--------  ---------------------------------------------------
0.000     Generator: yields timeout(0.1)
0.000     Processor: yields queue.get() [waiting for first packet]
0.100     Generator: resumes, can_enqueue() -> True, puts pkt #0
          Generator: yields timeout(0.1)
0.100     Processor: gets pkt #0, RAT=5g, DTMC=1024B, TX ok
          Processor: yields timeout(0.015) [15ms base latency + TX time]
0.115     Processor: TX complete, yields queue.get() [waiting]
0.200     Generator: resumes, puts pkt #1
0.200     Processor: gets pkt #1, processes...
...
5.000     Generator: puts pkt #50, queue depth=3 (processor lagging)
5.000     Processor: gets pkt #48, DTMC window PDR=0.96 -> stay at 1024B
...
12.500    DTMC window PDR=1.00 for 50 consecutive -> transition to 2048B
...
18.700    Network degrades, PDR drops to 0.92 -> DTMC back to 1024B
```

---

## What SimPy Does NOT Do Here

To avoid confusion, here is what is handled outside SimPy:

- **RAT selection logic** -- pure Python in `rat_selection.py` or precomputed CSV.
- **DTMC state transitions** -- synchronous calls inside the processor process.
- **Queue byte tracking** -- manual counters in `QueueSimulator` (not SimPy `Container`).
- **PHY capacity computation** -- pure functions, no events.
- **Metrics collection** -- appended to lists between yields.

SimPy provides exactly two things: **(1)** a shared clock (`env.now`) and **(2)** a mechanism for two concurrent processes to synchronize through the Store and timeouts.

---

## Quick Reference

| SimPy Concept | Project Usage | Location |
|---------------|---------------|----------|
| `Environment()` | `self.env` in `IntegratedQueueSimulator.run()` | line 1175 |
| `env.timeout(t)` | Inter-arrival pacing and TX delay | lines 998, 1135 |
| `Store(env)` | Packet FIFO between generator and processor | line 1176 |
| `store.put(x)` | Enqueue packet dict with arrival_time, id, size | line 1016 |
| `store.get()` | Dequeue next packet (blocks if empty) | line 1032 |
| `env.process(g)` | Register generator and processor | lines 1179-1180 |
| `env.run(until=t)` | Execute simulation | line 1184 |
| `env.now` | Timestamp for DTMC window, metrics, state lookup | throughout |

---

## Try It Yourself

Minimal standalone example showing the same producer-consumer pattern used in our simulator:

```python
import simpy

def car(env, name, charging_station):
    print(f'{name} arriving at {env.now}')
    with charging_station.request() as req:
        yield req  # Wait for charger
        print(f'{name} charging at {env.now}')
        yield env.timeout(5)  # Charge for 5 units
        print(f'{name} leaving at {env.now}')

env = simpy.Environment()
charger = simpy.Resource(env, capacity=1)  # Only 1 charger

for i in range(3):
    env.process(car(env, f'Car-{i}', charger))

env.run()
```

This shows `Resource` -- another SimPy primitive for limited resources. Our simulator uses `Store` instead because we need FIFO packet ordering rather than resource locking.

---

## Running Our Queue Simulator

```bash
# Generate RAT decisions first
python rat_selection.py --input /path/to/matched_data --mode api_batch --output rat_decisions.csv

# Run queue simulation with PHY constraints
python queue_simulator.py --input rat_decisions.csv --output sim_results.csv

# Run without PHY constraints (for comparison)
python queue_simulator.py --input rat_decisions.csv --output sim_results_no_phy.csv --no-phy-limits
```

---

## Further Reading

- SimPy Documentation: https://simpy.readthedocs.io/
- SimPy in 10 Minutes: https://simpy.readthedocs.io/en/latest/simpy_intro/index.html
- Our implementation: `queue_simulator.py`
- PHY and DTMC constants: `config.py`
- Data structures exchanged between components: `api_types.py`