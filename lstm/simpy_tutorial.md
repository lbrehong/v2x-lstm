# SimPy Tutorial: Queue Simulator Walkthrough

SimPy is a **discrete-event simulation** library. Instead of stepping through time tick-by-tick, it jumps directly from one event to the next, making it very efficient.

## Core Concepts

### 1. The Environment

Everything in SimPy revolves around an `Environment` - it's the simulation clock and event scheduler:

```python
import simpy

env = simpy.Environment()
print(env.now)  # Current simulation time (starts at 0)
```

In our simulator (`queue_simulator.py:785`):
```python
self.env = simpy.Environment()
```

### 2. Processes (Generators)

SimPy processes are **Python generators** that `yield` events. When a process yields, it pauses until that event completes:

```python
def my_process(env):
    print(f"Start at {env.now}")
    yield env.timeout(5)  # Wait 5 time units
    print(f"Resume at {env.now}")  # Prints "Resume at 5"
```

The `yield` keyword is the magic - it tells SimPy "pause here until this event happens."

### 3. Running the Simulation

```python
env = simpy.Environment()
env.process(my_process(env))  # Register the process
env.run(until=100)            # Run until time=100
```

---

## Our Queue Simulator Architecture

Here's how the pieces fit together:

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│ Packet Generator│────>│  SimPy Store    │────>│ Packet Processor │
│   (Producer)    │     │   (Queue)       │     │   (Consumer)     │
└─────────────────┘     └─────────────────┘     └──────────────────┘
     yields                  put/get               yields
   timeout(0.02)                                 timeout(latency)
```

### The Queue: `simpy.Store`

A `Store` is SimPy's built-in queue. It supports `put()` and `get()` which are **awaitable events**:

```python
queue = simpy.Store(env)

# Producer
yield queue.put(item)    # Blocks if store is full (with capacity)

# Consumer
item = yield queue.get() # Blocks until item available
```

From our code (`queue_simulator.py:786`):
```python
queue = simpy.Store(self.env)
```

---

## Walking Through the Code

### Process 1: Packet Generator (`_packet_generator`)

This process creates packets at a fixed rate (50 Hz = every 20ms):

```python
def _packet_generator(self, env: simpy.Environment, queue: simpy.Store):
    interval = 1.0 / self.arrival_rate_hz  # 0.02 seconds for 50 Hz
    packet_id = 0

    while not self._stop_simulation:
        # 1. Wait for next TX interval
        yield env.timeout(interval)

        # 2. Check PHY-layer queue capacity
        if self.enforce_phy_limits and not self.queue_sim.can_enqueue(estimated_size):
            self.queue_sim.metrics.dropped_packets += 1
            continue  # Drop packet, don't put in queue

        # 3. Create and enqueue packet
        packet = {"arrival_time": env.now, "id": packet_id}
        yield queue.put(packet)  # This "blocks" until queue accepts it

        packet_id += 1
```

**Key SimPy concept**: `yield env.timeout(interval)` pauses this process for `interval` time units. During this pause, other processes can run.

### Process 2: Packet Processor (`_packet_processor`)

This process consumes packets and simulates transmission:

```python
def _packet_processor(self, env: simpy.Environment, queue: simpy.Store):
    while True:
        # 1. Wait for a packet (blocks until one arrives)
        packet = yield queue.get()

        # 2. Get network state and make RAT decision
        state = self._get_network_state(env.now)
        if state is None:
            self._stop_simulation = True
            break

        rat_decision = self._get_rat_decision(state)

        # 3. Simulate transmission
        success, latency, tx_time = self._simulate_transmission(...)

        # 4. Wait for transmission to complete
        yield env.timeout(latency / 1000.0)  # Convert ms to seconds
```

**Key SimPy concept**: `yield queue.get()` blocks until a packet is available. This is how producer-consumer synchronization works.

### Starting Both Processes

```python
def run(self, duration=None, max_packets=None):
    self.env = simpy.Environment()
    queue = simpy.Store(self.env)

    # Register both processes
    self.env.process(self._packet_generator(self.env, queue))
    self.env.process(self._packet_processor(self.env, queue))

    # Run simulation
    self.env.run(until=duration)
```

---

## SimPy Timeline Example

Let's trace what happens with 50 Hz packet rate:

```
Time    Event
-----   -----------------------------------------
0.00    Generator: yields timeout(0.02)
0.00    Processor: yields queue.get() [waiting]
0.02    Generator: resumes, puts packet #0, yields timeout(0.02)
0.02    Processor: gets packet #0, processes, yields timeout(0.015)
0.035   Processor: TX complete, yields queue.get() [waiting]
0.04    Generator: resumes, puts packet #1, yields timeout(0.02)
0.04    Processor: gets packet #1, processes...
```

The key insight: **SimPy only advances time when all processes are blocked on yields.**

---

## PHY-Layer Integration

The PHY modeling I added works alongside SimPy but doesn't use SimPy-specific features:

```python
# Before enqueueing (in generator)
if not self.queue_sim.can_enqueue(packet_size):
    # PHY says queue is full - drop packet
    self.metrics.dropped_packets += 1
    continue

# During transmission simulation
tx_time = calculate_tx_time_ms(packet_size, rat)  # PHY-based calculation
latency = base_latency + tx_time + jitter

# The SimPy part is just the wait
yield env.timeout(latency / 1000.0)
```

---

## Quick Reference

| SimPy Concept    | Our Usage              | Purpose                          |
|------------------|------------------------|----------------------------------|
| `Environment()`  | `self.env`             | Simulation clock                 |
| `env.timeout(t)` | Wait between packets   | Pause process for `t` time       |
| `Store()`        | Packet queue           | FIFO buffer between processes    |
| `store.put(x)`   | Enqueue packet         | Add item (can block if bounded)  |
| `store.get()`    | Dequeue packet         | Remove item (blocks if empty)    |
| `env.process(g)` | Start generator/proc   | Register a process               |
| `env.run(until=t)` | Run simulation       | Execute until time `t`           |
| `env.now`        | Current timestamp      | Get simulation time              |

---

## Try It Yourself

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

This shows `Resource` - another SimPy primitive for limited resources (like our bounded queue).

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
- Our implementation: `queue_simulator.py` (lines 679-1000)