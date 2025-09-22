import pandas as pd

# Input CSV file path
input_file = 'E:/cleaned_output.csv'  # Replace with actual filename
output_file = 'user_combined_summary.csv'

# Read the CSV
df = pd.read_csv(input_file)

# Normalize status values
df['status'] = df['status'].str.strip().str.lower()

# Aggregate main stats
grouped = df.groupby('user_id').agg(
    submission_count=('submission_id', 'count'),
    avg_cpu_time=('cpu_time', 'mean'),
    avg_memory=('memory', 'mean'),
    avg_code_size=('code_size', 'mean'),
    avg_total_lines=('total_lines', 'mean'),
    avg_line_spacing=('avg_line_spacing', 'mean'),
    avg_total_comments=('total_comments', 'mean'),
    avg_if_else_count=('if_else_count', 'mean'),
    avg_total_variables=('total_variables', 'mean'),
    avg_loop_count=('loop_count', 'mean'),
    avg_capitalized_variable_names=('capitalized_variable_names', 'mean'),
    avg_percentage_for_loops=('percentage_for_loops', 'mean'),
    submission_count_per_problem=('problem_id', 'nunique')
).reset_index()

# Round all average columns
avg_cols = [col for col in grouped.columns if col.startswith('avg_')]
grouped[avg_cols] = grouped[avg_cols].round(0).astype(int)

# Compute avg submission count per problem
grouped['avg_submission_count_per_problem'] = (
    grouped['submission_count'] / grouped['submission_count_per_problem']
).round(0).astype(int)

# Drop helper column
grouped.drop(columns=['submission_count_per_problem'], inplace=True)

# Status counts manually (without pivot)
status_map = {
    'accepted': 'accepted_count',
    'runtime error': 'runtime_error_count',
    'wrong answer': 'wrong_answer_count',
    'time limit exceeded': 'time_limit_exceeded_count'
}

status_df = pd.DataFrame({'user_id': df['user_id'].unique()})

for status_val, status_col in status_map.items():
    temp = df[df['status'] == status_val].groupby('user_id').size().reset_index(name=status_col)
    status_df = pd.merge(status_df, temp, on='user_id', how='left')

# Fill missing status counts with 0
status_df.fillna(0, inplace=True)
status_df = status_df.astype({col: int for col in status_df.columns if col != 'user_id'})

# Final merge
final_df = pd.merge(grouped, status_df, on='user_id', how='left')

# Save to CSV
final_df.to_csv(output_file, index=False)
print(f" Summary saved to: {output_file}")
