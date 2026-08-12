import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

rect1 = patches.Rectangle((1, 7), 3, 1.5, linewidth=2, edgecolor='blue', facecolor='lightblue')
ax.add_patch(rect1)
ax.text(2.5, 7.75, 'Feature\nExtractor', ha='center', va='center', fontsize=11, fontweight='bold')

rect2 = patches.Rectangle((5, 7), 3, 1.5, linewidth=2, edgecolor='green', facecolor='lightgreen')
ax.add_patch(rect2)
ax.text(6.5, 7.75, 'Contrastive\nLearning', ha='center', va='center', fontsize=11, fontweight='bold')

rect3 = patches.Rectangle((9, 7), 3, 1.5, linewidth=2, edgecolor='orange', facecolor='lightsalmon')
ax.add_patch(rect3)
ax.text(10.5, 7.75, 'Prototype\nMemory', ha='center', va='center', fontsize=11, fontweight='bold')

rect4 = patches.Rectangle((2, 4), 3, 1.5, linewidth=2, edgecolor='purple', facecolor='thistle')
ax.add_patch(rect4)
ax.text(3.5, 4.75, 'EMA\nUpdate', ha='center', va='center', fontsize=11, fontweight='bold')

rect5 = patches.Rectangle((6, 4), 3, 1.5, linewidth=2, edgecolor='red', facecolor='lightcoral')
ax.add_patch(rect5)
ax.text(7.5, 4.75, 'Forgetting\nMechanism', ha='center', va='center', fontsize=11, fontweight='bold')

rect6 = patches.Rectangle((10, 4), 3, 1.5, linewidth=2, edgecolor='cyan', facecolor='lightcyan')
ax.add_patch(rect6)
ax.text(11.5, 4.75, 'Nearest\nPrototype', ha='center', va='center', fontsize=11, fontweight='bold')

ax.arrow(4, 7.75, 1, 0, head_width=0.15, head_length=0.2, fc='black', ec='black')
ax.arrow(8, 7.75, 1, 0, head_width=0.15, head_length=0.2, fc='black', ec='black')
ax.arrow(10.5, 6.25, -8, -1.5, head_width=0.15, head_length=0.2, fc='black', ec='black')
ax.arrow(10.5, 6.25, 0, -1.5, head_width=0.15, head_length=0.2, fc='black', ec='black')
ax.arrow(6.5, 6.25, -3, -1.5, head_width=0.15, head_length=0.2, fc='black', ec='black')

rect7 = patches.Rectangle((6, 1), 3, 1.5, linewidth=2, edgecolor='magenta', facecolor='pink')
ax.add_patch(rect7)
ax.text(7.5, 1.75, 'Output: Few-shot\nIncremental Classification', ha='center', va='center', fontsize=12, fontweight='bold')

ax.arrow(11.5, 3.5, -5, -1.5, head_width=0.15, head_length=0.2, fc='black', ec='black')

ax.text(0.5, 7.75, 'Tabular Data', ha='right', va='center', fontsize=11)
ax.arrow(0.8, 7.75, 0.2, 0, head_width=0.1, head_length=0.15, fc='black', ec='black')

plt.title('Contrastive-Enhanced Prototype Memory Network (CE-PMN)', fontsize=14, fontweight='bold', pad=20)
plt.savefig('plots/framework_architecture.png', dpi=300, bbox_inches='tight')
plt.close()

print("Architecture plot generated successfully!")