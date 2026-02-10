"""Queue simulator with DTMC packet sizing and PHY-layer modeling."""
try:
    from queuesim.queue_simulator import QueueSimulator, IntegratedQueueSimulator
    from queuesim.dtmc_sizer import DTMCPacketSizer
    from queuesim.sim_types import SimulationMetrics
except ImportError:
    pass  # simpy may not be installed
