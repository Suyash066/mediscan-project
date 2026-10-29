"""Draws the ArogyaNet block diagram -> reports/figures/arogyanet_diagram.png (no data needed)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from lib import FIGS

fig, ax = plt.subplots(figsize=(13, 4.6)); ax.set_xlim(0, 13); ax.set_ylim(0, 4.6); ax.axis("off")
def box(x, y, w, h, t, c="#dbeafe", fs=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=c, ec="#1e3a8a", lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
def arrow(x1, y1, x2, y2, **k):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, lw=1, **k))
y = 2.3
box(0.1, y, 1.2, 1.0, "Input\n1x224x224\ngrey X-ray", "#fde68a"); arrow(1.3, y + .5, 1.55, y + .5)
box(1.55, y, 1.3, 1.0, "Stem\nConv3x3 s2\nBN-ReLU\n24 x 112²"); arrow(2.85, y + .5, 3.1, y + .5)
spec = [("Block 1\n24→48, s2\n56²"), ("Block 2\n48→48\n56²"), ("Block 3\n48→96, s2\n28²"), ("Block 4\n96→96\n28²"), ("Block 5\n96→192, s2\n14²"), ("Block 6\n192→192\n14²")]
x = 3.1
for s in spec:
    box(x, y, 1.05, 1.0, s, "#bbf7d0"); arrow(x + 1.05, y + .5, x + 1.2, y + .5); x += 1.2
box(x, y, 0.95, 1.0, "Global\nAvgPool\n192", "#fecaca"); arrow(x + .95, y + .5, x + 1.2, y + .5)
box(x + 1.2, y, 0.9, 1.0, "Dropout\n0.2\n+ FC(→1)", "#e9d5ff")
ax.text(x + 1.65, y - .25, "logit → sigmoid", ha="center", fontsize=7)
# inside-a-block zoom
bx = 3.0; by = 0.2
ax.text(bx, by + 1.35, "Inside one block  (D1 = skip, D2 = SE are the two ablated decisions)", fontsize=8, weight="bold")
for i, t in enumerate(["Depthwise\n3x3 (stride)\nBN-ReLU", "Pointwise\n1x1\nBN", "SE (D2)\nGAP→FC→FC\n→sigmoid ×", "+  (D1)\nskip", "ReLU"]):
    box(bx + i * 1.7, by + .3, 1.35, .8, t, "#fef9c3" if i not in (2, 3) else "#fbcfe8"); 
    if i < 4: arrow(bx + i * 1.7 + 1.35, by + .7, bx + (i + 1) * 1.7, by + .7)
ax.add_patch(FancyArrowPatch((bx + .1, by + .3), (bx + 3 * 1.7 + .7, by + .3), connectionstyle="arc3,rad=0.25", arrowstyle="-|>", mutation_scale=9, lw=1, ls="--", color="#be185d"))
ax.text(bx + 3.2, by - .12, "identity / (avgpool + 1x1 conv) skip", fontsize=6.5, color="#be185d", ha="center")
box(0.1, 0.5, 2.4, 0.8, "Two-branch variant: [mean intensity,\naspect ratio, area] → Dense16 → Dense16\n→ concat with pooled features", "#ffedd5", 6.5)
plt.savefig(FIGS / "arogyanet_diagram.png", dpi=140, bbox_inches="tight"); print("wrote reports/figures/arogyanet_diagram.png")
