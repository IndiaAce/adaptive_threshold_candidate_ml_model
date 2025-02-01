import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_kpis(num_kpis=30, days=90, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    start_time = datetime.now() - timedelta(days=days)
    time_series = [start_time + timedelta(hours=i) for i in range(days * 24)]
    
    data = []
    
    for i in range(num_kpis):
        kpi_id = f'kpi_{i:03d}'
        
        if i < num_kpis // 2:  # Consistent KPIs
            base_value = random.randint(100, 500)
            daily_pattern = np.sin(np.linspace(0, 6.28, 24)) * 100 + base_value
            weekly_variation = np.sin(np.linspace(0, 6.28, 7)) * 50  # Fix broadcasting issue
            
            values = []
            for j in range(days):
                daily_values = daily_pattern + weekly_variation[j % 7] + np.random.normal(0, 5, 24)
                values.extend(daily_values)
        else:  # Inconsistent KPIs
            values = np.random.randint(50, 1000, size=days * 24) + np.random.normal(0, 100, days * 24)
            
        for t, val in zip(time_series, values):
            data.append([t.isoformat(), kpi_id, max(0, round(val, 2))])  # Ensure no negative values
    
    df = pd.DataFrame(data, columns=['_time', 'itsi_kpi_id', 'avg(alert_value)'])
    return df

# Generate synthetic KPIs
df_synthetic = generate_synthetic_kpis()

# Save to CSV for further use
df_synthetic.to_csv("adaptive_threshold_candidate_ml_model/v2/data/synthetic_data/synthetic_kpis.csv", index=False)

# Display the synthetic dataset
print("Synthetic KPIs dataset generated and saved as 'synthetic_kpis.csv'.")
