import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# -----------------------
# CONFIGURATION
# -----------------------
NUM_KPIS = 100
DAYS = 365
START_DATE = "2024-01-01"  # Adjust as needed
KPI_NAMES = [
    # A list of example KPI names. Customize as needed or generate them dynamically.
    "CPU_Usage", "Memory_Usage", "Disk_Usage", "MFA_Fails", "User_Logons", 
    "API_Latency", "Error_Rate", "Throughput", "Queue_Depth", "Payment_Transactions",
    "Network_Usage", "Session_Count", "Login_Attempts", "Transaction_Duration",
    "Request_Count", "Response_Time", "Cache_Hit_Rate", "Disk_Io", "Database_Locks",
    "Database_Size", "Server_Temperature", "Power_Consumption", "Web_Traffic", 
    "Active_Threads", "Suspicious_Logons", "Failed_Logins", "Successful_Logins", 
    "Memory_Swap", "Process_Count", "Page_Faults", "Thread_Pool_Usage", 
    "Garbage_Collection", "IOT_Events", "IoT_Temp_Sensor", "IoT_Humidity_Sensor",
    "Mobile_Checkins", "Email_Bounce_Rate", "Ad_Clicks", "Ad_Impressions", 
    "Signups", "Password_Resets", "Network_Errors", "Concurrent_Users", 
    "File_Uploads", "Failed_Transfers", "Successful_Transfers", "Read_IOPS", 
    "Write_IOPS", "Data_Volume", "Container_Restarts", "Pod_Crashes", 
    "Node_Count", "Session_Drop_Rate", "Video_Stream_Bitrate", "Stream_Stalls", 
    "Service_Health", "Service_Restarts", "Block_Storage_Usage", "Object_Storage_Usage",
    "Messages_Processed", "Billing_Errors", "VM_Cpu_Steal", "VM_Memory_Ballooning",
    "K8S_CrashLoopBackOff", "Load_Average", "MFA_Success", "User_Tickets", 
    "Ticket_Resolution_Time", "Batch_Job_Failures", "Batch_Job_Success",
    "Cache_Misses", "Cache_Evictions", "DB_Connection_Errors", "DB_Query_Count",
    "WebSocket_Connections", "Thread_Queue_Length", "Image_Processing_Requests",
    "Notifications_Sent", "Notifications_Failed", "Lambda_Invocations", 
    "Lambda_Errors", "IoT_Device_Online", "IoT_Device_Offline", "Server_Response_2xx",
    "Server_Response_4xx", "Server_Response_5xx", "Network_Packets_Dropped",
    "SSL_Handshake_Errors", "Inbound_Traffic", "Outbound_Traffic", "Latency_P95", 
    "Latency_P99", "Daily_Billing", "Daily_Revenue", "Click_Through_Rate",
    "Customer_Churn", "Container_Deployments", "Pod_Scaling_Events", "Deployment_Failures",
    "DNS_Lookup_Time", "DHCP_Lease_Count", "FTP_Connections", "HTTP_Requests",
    "Web_Access_Errors"
]

# If you have fewer or more than 100 names in KPI_NAMES,
# the script will handle them, repeating or truncating as needed.

# Distribution of KPI types
# e.g. 30 consistent, 30 erratic, 40 combination
CONSISTENT_COUNT = 30
ERRATIC_COUNT = 30
COMBINATION_COUNT = 40

# Output directory for CSV files
OUTPUT_DIR = "synthetic_kpi_data"

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def generate_timestamps(start_date, days, freq="H"):
    """
    Generate a DatetimeIndex starting from start_date, covering 'days' days 
    at the specified frequency (default hourly).
    """
    start = pd.to_datetime(start_date)
    end = start + timedelta(days=days)
    # end is exclusive, so to get full 365 days hourly: range should be up to end
    # We'll generate 365 * 24 = 8760 hours
    dt_index = pd.date_range(start, periods=days*24, freq=freq)
    return dt_index

def is_weekend_func(day_of_week):
    """ Return True if day_of_week is Sat (5) or Sun (6). """
    return day_of_week >= 5

def rolling_stats(series, window=3):
    """
    Calculate rolling average and rolling std for a given series.
    Using a minimum period so that the first few hours still get a result.
    """
    rolling_avg = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std().fillna(0)
    return rolling_avg, rolling_std

def generate_consistent_pattern(timestamps):
    """
    Generate a consistent pattern: e.g., daily cycles with smaller random noise.
    Incorporate hour-of-day and day-of-week effects.
    """
    base_values = []
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.dayofweek
        
        # Simple baseline that changes by hour.
        # For instance, let's assume highest usage midday, lower usage at night
        daily_cycle = 10 + 5 * np.sin((hour / 24) * 2 * np.pi)  # Sine wave for day cycle
        
        # Weekly effect: slight increase on weekdays vs weekends (just an example)
        if day_of_week < 5:
            weekly_factor = 1.0  # weekdays
        else:
            weekly_factor = 0.8  # weekends
        
        # Combine the factors
        value = (daily_cycle * weekly_factor) + random.uniform(-1, 1)  # add small random noise
        base_values.append(value)
        
    return pd.Series(base_values, index=timestamps)

def generate_erratic_pattern(timestamps):
    """
    Generate an erratic pattern: random spikes, irregular fluctuations.
    """
    values = []
    current_value = 50  # arbitrary starting point
    
    for ts in timestamps:
        # step can be large or small, up or down
        step = random.uniform(-5, 5)  # normal hourly change
        # occasionally inject big spikes
        if random.random() < 0.01:  # 1% chance
            step = random.uniform(-50, 50)
        
        current_value += step
        # Floor to 0 to avoid negative if it doesn't make sense (you can remove if negatives are valid)
        current_value = max(0, current_value)
        values.append(current_value)
    
    return pd.Series(values, index=timestamps)

def generate_combination_pattern(timestamps):
    """
    Generate a combination pattern: partly consistent daily cycle + random spikes.
    """
    consistent_series = generate_consistent_pattern(timestamps)
    # Add random spikes on top
    combined_values = []
    for val in consistent_series:
        # 3% chance of a large spike or drop
        if random.random() < 0.03:
            spike = random.uniform(-20, 20)
        else:
            spike = 0
        combined_values.append(val + spike)
    
    return pd.Series(combined_values, index=timestamps)

def insert_anomalies(series, factor=3, anomaly_prob=0.01):
    """
    Mark certain points as anomalies based on a probability or 
    if they deviate significantly from local rolling stats.
    
    Return a boolean Series indicating anomalies.
    """
    # We'll do a simple approach: each point has a small chance (anomaly_prob) 
    # to become an anomaly by random injection. 
    # Or you could refine by checking actual deviation from rolling stats.
    is_anomaly = pd.Series(False, index=series.index)
    
    # Probability-based injection
    anomaly_indices = np.where(np.random.rand(len(series)) < anomaly_prob)[0]
    for idx in anomaly_indices:
        is_anomaly.iloc[idx] = True
    
    return is_anomaly

def generate_kpi_data(kpi_name, kpi_type, timestamps):
    """
    Generate the DataFrame for a single KPI with the specified type.
    """
    # Step 1: Generate base values according to the type
    if kpi_type == 'consistent':
        base_series = generate_consistent_pattern(timestamps)
    elif kpi_type == 'erratic':
        base_series = generate_erratic_pattern(timestamps)
    else:  # 'combination'
        base_series = generate_combination_pattern(timestamps)
    
    # Step 2: Determine day_of_week, hour_of_day, weekend, etc.
    day_of_week = timestamps.dayofweek
    hour_of_day = timestamps.hour
    weekend_flags = day_of_week.map(is_weekend_func)
    
    # Step 3: Compute rolling stats
    rolling_avg_series, rolling_std_series = rolling_stats(base_series, window=3)
    
    # Step 4: Insert anomalies
    anomaly_series = insert_anomalies(base_series, anomaly_prob=0.01)
    
    # Step 5: Build the DataFrame
    df = pd.DataFrame({
        "timestamp": timestamps,
        "value": base_series,
        "kpi_name": kpi_name,
        "day_of_week": day_of_week,
        "hour_of_day": hour_of_day,
        "is_weekend": weekend_flags,
        # This is just a simplistic approach to units. You can refine or randomize as needed.
        "unit": "count" if "Count" in kpi_name or "Fails" in kpi_name or "Logons" in kpi_name else "percentage",
        "rolling_avg": rolling_avg_series,
        "rolling_std": rolling_std_series,
        "is_anomaly": anomaly_series,
        "kpi_type": kpi_type
    })
    
    return df

def ensure_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# -----------------------
# MAIN GENERATION SCRIPT
# -----------------------
def main():
    # 1. Generate a DatetimeIndex for the year
    timestamps = generate_timestamps(START_DATE, DAYS, freq="H")
    
    # 2. Prepare KPI distribution
    # If you have exactly 100 KPI names, you can shuffle them and assign types.
    # Or if you have more than 100, it will truncate. If fewer, it will reuse.
    all_kpi_names = []
    while len(all_kpi_names) < NUM_KPIS:
        all_kpi_names.extend(KPI_NAMES)
    all_kpi_names = all_kpi_names[:NUM_KPIS]
    
    # Shuffle KPI names to randomize distribution
    random.shuffle(all_kpi_names)
    
    kpi_types = []
    kpi_types.extend(['consistent'] * CONSISTENT_COUNT)
    kpi_types.extend(['erratic'] * ERRATIC_COUNT)
    kpi_types.extend(['combination'] * COMBINATION_COUNT)
    
    # If the sum of the above does not match NUM_KPIS, adjust dynamically
    if len(kpi_types) < NUM_KPIS:
        kpi_types.extend(['combination'] * (NUM_KPIS - len(kpi_types)))
    kpi_types = kpi_types[:NUM_KPIS]
    
    # Shuffle the types for variety
    random.shuffle(kpi_types)
    
    ensure_output_dir(OUTPUT_DIR)
    
    # 3. Generate data for each KPI and save to CSV
    for i in range(NUM_KPIS):
        kpi_name = all_kpi_names[i]
        kpi_type = kpi_types[i]
        
        df_kpi = generate_kpi_data(kpi_name, kpi_type, timestamps)
        
        # 4. Save to CSV
        filename = f"{kpi_name.replace(' ', '_')}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_kpi.to_csv(filepath, index=False)
        
        print(f"Generated {filepath}")

if __name__ == "__main__":
    main()
