"""Optional: fit only the global fusion scalar using detached score inputs."""
_base_ = ['./a0_full.py']
model = dict(selector_gate_trainable=True)
