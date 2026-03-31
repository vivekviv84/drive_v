import torch
import numpy as np
import onnxruntime as ort
import time, os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.liteseg_edge import LiteSegEdge

CKPT     = r'C:\Users\ACER\LiteSegEdge\checkpoints\best.pth'
ONNX     = r'C:\Users\ACER\LiteSegEdge\demo\liteseg_edge.onnx'
OUT_DIR  = r'C:\Users\ACER\LiteSegEdge\demo'
DEVICE   = torch.device('cuda')
RUNS     = 1000
WARMUP   = 100
INPUT    = (1, 3, 360, 640)

print("=" * 50)
print("  LiteSegEdge — Inference Benchmark")
print("=" * 50)

# ── 1. PyTorch CPU ──
print("\n[1/3] PyTorch CPU...")
model_cpu = LiteSegEdge()
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
model_cpu.load_state_dict(ckpt['model'])
model_cpu.eval()
x_cpu = torch.randn(INPUT)
with torch.no_grad():
    for _ in range(10): model_cpu(x_cpu)
    t0 = time.perf_counter()
    for _ in range(200): model_cpu(x_cpu)
    cpu_fps = 200 / (time.perf_counter() - t0)
print(f"   CPU FPS: {cpu_fps:.1f}")

# ── 2. PyTorch CUDA ──
print("[2/3] PyTorch CUDA...")
model_gpu = LiteSegEdge().to(DEVICE)
model_gpu.load_state_dict(ckpt['model'])
model_gpu.eval()
x_gpu = torch.randn(INPUT, device=DEVICE)
with torch.no_grad():
    for _ in range(WARMUP):
        model_gpu(x_gpu)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(RUNS):
        model_gpu(x_gpu)
    torch.cuda.synchronize()
    cuda_fps = RUNS / (time.perf_counter() - t0)
print(f"   CUDA FPS: {cuda_fps:.1f}")

# ── 3. ONNX Runtime CUDA ──
print("[3/3] ONNX Runtime CUDA EP...")
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(
    ONNX,
    sess_options=sess_options,
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
x_np = np.random.randn(*INPUT).astype(np.float32)
inp_name = sess.get_inputs()[0].name
# warmup
for _ in range(WARMUP):
    sess.run(None, {inp_name: x_np})
# benchmark
latencies = []
for _ in range(RUNS):
    t0 = time.perf_counter()
    sess.run(None, {inp_name: x_np})
    latencies.append(time.perf_counter() - t0)
onnx_fps = 1.0 / np.mean(latencies)
onnx_p99 = np.percentile(latencies, 99) * 1000
print(f"   ONNX CUDA FPS : {onnx_fps:.1f}")
print(f"   P99 latency   : {onnx_p99:.2f} ms")

# ── Results ──
print("\n" + "=" * 50)
print(f"  PyTorch CPU  : {cpu_fps:>7.1f} FPS")
print(f"  PyTorch CUDA : {cuda_fps:>7.1f} FPS")
print(f"  ONNX CUDA EP : {onnx_fps:>7.1f} FPS")
print("=" * 50)

# ── Chart ──
labels = ['PyTorch\nCPU', 'PyTorch\nCUDA', 'ONNX\nCUDA EP']
values = [cpu_fps, cuda_fps, onnx_fps]
colors = ['#ff6b6b', '#74c0fc', '#69db7c']

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#161b22')
ax.tick_params(colors='white')
for spine in ['bottom','left']:
    ax.spines[spine].set_color('#30363d')
for spine in ['top','right']:
    ax.spines[spine].set_visible(False)

bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='none')
ax.axhline(60,  color='#ff6b6b', linestyle='--', linewidth=1.5, label='Min target (60 FPS)')
ax.axhline(288, color='#ffd43b', linestyle='--', linewidth=1.5, label='Previous best (288 FPS)')
ax.set_ylabel('Frames Per Second', color='white', fontsize=12)
ax.set_title('LiteSegEdge — Inference Speed Benchmark\nRTX 3070 Ti Laptop GPU',
             color='white', fontsize=14, fontweight='bold')
ax.legend(facecolor='#161b22', labelcolor='white')
ax.yaxis.label.set_color('white')

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 5,
            f'{val:.0f} FPS',
            ha='center', color='white',
            fontweight='bold', fontsize=13)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'benchmark_chart.png'),
            dpi=150, bbox_inches='tight', facecolor='#0d1117')
plt.show()
print("\nSaved to demo/benchmark_chart.png")