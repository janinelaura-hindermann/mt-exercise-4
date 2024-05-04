import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
csv_file_path = 'perplexity_comparison.csv'
df = pd.read_csv(csv_file_path, index_col='validation ppl')

# Plot the data as a line chart
plt.figure(figsize=(10, 6))
for column in df.columns:
    plt.plot(df.index, df[column], label=column)

# Customize the chart
plt.xlabel('Validation Step')
plt.ylabel('Perplexity')
plt.title('Perplexity Comparison Across Different Models')
plt.legend()
plt.grid(visible=True)

# Create a list of x-tick locations for every 5000 steps
xticks = [step for step in df.index if step % 5000 == 0]

# Set the x-ticks to only display these selected steps
plt.xticks(xticks, rotation=45)
plt.tight_layout()

# Save the plot as an image file or show it
plt.savefig('perplexity_comparison.png')  # Saves as a PNG image
