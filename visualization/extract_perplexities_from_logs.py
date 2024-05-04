import re
import pandas as pd

# Paths to your log files
log_files = {
    'baseline': '../logs/baseline.log',
    'prenorm': '../logs/model_pre/err',
    'postnorm': '../logs/model_post/err'
}

# Regular expression patterns
step_pattern = re.compile(r'Step:\s+(\d+),')
ppl_pattern = re.compile(r'ppl:\s+(\d+\.\d+)')


# Function to parse a log file and return a dictionary of step-perplexity pairs
def parse_log_file(log_file_path, step_intervals=500):
    current_step = None
    perplexities = {}

    with open(log_file_path, 'r') as file:
        for line in file:
            # Check for the step pattern and update the current step if found
            step_match = step_pattern.search(line)
            if step_match:
                current_step = int(step_match.group(1))

            # Check for the perplexity pattern and store it if the current step is a multiple of the desired interval
            ppl_match = ppl_pattern.search(line)
            if ppl_match and current_step is not None and current_step % step_intervals == 0:
                perplexities[current_step] = float(ppl_match.group(1))

    return perplexities


# Create a dictionary to hold all parsed data
all_perplexities = {name: parse_log_file(path) for name, path in log_files.items()}

# Determine all unique steps across all logs
all_steps = sorted(set(step for data in all_perplexities.values() for step in data))

# Create a DataFrame to hold the final results
results = pd.DataFrame(index=all_steps)
results.index.name = 'validation ppl'

# Fill in the DataFrame with perplexity values for each model/log
for name, data in all_perplexities.items():
    results[name] = [data.get(step, None) for step in all_steps]

# Save the DataFrame to a CSV file
results.to_csv('perplexity_comparison.csv')
