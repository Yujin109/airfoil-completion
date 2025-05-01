################################################

# Generate a color cycle using the hsv colormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

N_colors = 6

# Generate a color cycle using the hsv or jet colormap
colors = plt.cm.jet(np.linspace(0.1, 1, N_colors))  # Adjust the number of colors

# add black as the first color
colors = np.insert(colors, 0, [0, 0, 0, 1], axis=0)

# adjust color 3 to green
colors[3] = [0, 1, 0, 1]

# replace color 4 with yellow
colors[4] = [1, 1, 0, 1]

# insert after color 6 red
colors = np.insert(colors, 6, [1, 0, 0, 1], axis=0)

# adjust color 5 to orange
colors[5] = [1, 0.647, 0, 1]

# Save the color cycle as a list of hex color codes
color_cycle = [mcolors.rgb2hex(c) for c in colors]
print(color_cycle)

# Create style sheet content
style_content = f"axes.prop_cycle: cycler('color', {color_cycle})"

# Save to a .mplstyle file
with open("hsv_style.mplstyle", "w") as f:
    f.write(style_content)
    
# Plot all colors
fig, ax = plt.subplots()
for i, color in enumerate(color_cycle):
    ax.plot([0, 1], [i, i], color=color, label=f"Color {i}")
    
ax.legend()
plt.show()


################################################