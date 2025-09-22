import pandas as pd

# === Configuration ===
input_file = 'combined_user_problem_stats.csv'         # Replace with your CSV filename
output_file = 'filtered_output2.csv'

# === Step 1: Read the CSV ===
df = pd.read_csv(input_file)

# === Step 2: Count number of rows per user_id ===
user_counts = df['user_id'].value_counts()

# === Step 3: Select only user_ids with 11, 12, or 13 rows ===
valid_user_ids = user_counts[user_counts.isin([11,12,13,14,15,16,17,18,19])].index

# === Step 4: Filter the original DataFrame ===
filtered_df = df[df['user_id'].isin(valid_user_ids)]

# === Step 5: Save to new CSV ===
filtered_df.to_csv(output_file, index=False)

print(f" Filtered data saved to {output_file}")
