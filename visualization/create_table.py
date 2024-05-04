import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
csv_file_path = 'perplexity_comparison.csv'
df = pd.read_csv(csv_file_path)

# Create a plot figure
fig, ax = plt.subplots(figsize=(12, 6))  # Adjust size as needed

# Turn off the axes (not required for a table)
ax.axis('off')
ax.axis('tight')

# Draw the DataFrame as a table onto the plot
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')

# Style adjustments for the table
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # Adjust scale for readability

# Save the plot with the table as a PNG file
plt.savefig('perplexity_comparison_table.png', bbox_inches='tight', pad_inches=0.5)
