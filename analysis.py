import plotly.express as px

# File paths
file_no_slurm = r"gemmtask1NoSlurm.csv"
file_slurm = r"gemmtask1SLURM.csv"

# Load CSV data
df_no_slurm = pd.read_csv(file_no_slurm)
df_slurm = pd.read_csv(file_slurm)

# Add a label column to differentiate the data
df_no_slurm['Type'] = 'No SLURM'
df_slurm['Type'] = 'SLURM'

# Combine the two dataframes
df = pd.concat([df_no_slurm, df_slurm])

# Calculate MNK for better visualization
df['MNK'] = df['M'] * df['N'] * df['K']

# Create the scatter plot
fig = px.scatter(
    df,
    x='MNK',
    y='AverageTime(ms)',
    color='Type',
    hover_data={'M': True, 'N': True, 'K': True, 'PerformanceProportion(time/MNK)': True},
    title="Comparison of Average Time vs. MNK",
    labels={"MNK": "M*N*K", "AverageTime(ms)": "Average Time (ms)"},
)

# Show the plot
fig.show()