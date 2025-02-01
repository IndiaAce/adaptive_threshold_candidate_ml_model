import pandas as pd
import numpy as np
import random
import os
import secrets
from datetime import datetime, timedelta

# -----------------------
# CONFIGURATION
# -----------------------
NUM_KPIS = 100
DAYS = 365
START_DATE = "2024-01-01"  # Adjust as needed

# A list of example KPI names. (Feel free to adjust or generate dynamically.)
KPI_NAMES = [
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

# Distribution of KPI types (change as needed)
# e.g. 30 consistent, 30 erratic, 40 combination
CONSISTENT_COUNT = 30
ERRATIC_COUNT = 30
COMBINATION_COUNT = 40

# Output directory for CSV files
OUTPUT_DIR = "synthetic_kpi_data"

# -----------------------
# HELPER FUNCTIONS
# -----------------------

def generate_kpi_id():
    """Generate a random 24-character hexadecimal string (e.g., similar to a MongoDB ObjectID)."""
    return secrets.token_hex(12)

def generate_timestamps(start_date, days, freq="H"):
    """
    Generate a timezone-aware DatetimeIndex starting from start_date, covering 'days' days 
    at the specified frequency (default hourly). We use a fixed offset of -05:00.
    """
    start = pd.to_datetime(start_date)
    # Create a timezone with a fixed offset of -300 minutes (-05:00)
    tz = pd.FixedOffset(-300)
    # Generate a date_range with the given frequency and assign the timezone
    dt_index = pd.date_range(start, periods=days*24, freq=freq, tz=tz)
    return dt_index

def generate_consistent_pattern(timestamps, kpi_name):
    """
    For "consistent" KPIs, we return a repeatable daily pattern.
    
    • If the KPI name contains "logon" (e.g., User_Logons), then on weekdays
      only the 9:00 and 12:00 hours produce a high value while other hours (and weekends)
      return nearly 0.
      
    • Otherwise, we use a fixed hourly pattern (the same for every weekday) and return 0 on weekends.
    """
    values = []
    if "logon" in kpi_name.lower():
        # Special pattern for user logons (or similar) – peaks only at 9:00 and 12:00 on weekdays.
        for ts in timestamps:
            if ts.dayofweek < 5:  # Weekday
                if ts.hour in [9, 12]:
                    val = random.uniform(200, 300)
                else:
                    val = random.uniform(0, 10)
            else:
                val = 0
            values.append(val)
    else:
        # Define a fixed pattern for hours 0-23. (You can adjust these base values as needed.)
        hourly_pattern = [200, 190, 180, 170, 160, 150, 750, 400, 370, 470, 
                          385, 365, 370, 250, 400, 370, 350, 330, 320, 310, 
                          300, 290, 280, 270]
        for ts in timestamps:
            if ts.dayofweek < 5:  # Weekday
                base = hourly_pattern[ts.hour] if ts.hour < len(hourly_pattern) else 300
                # Add a small random noise
                val = base + random.uniform(-20, 20)
            else:
                val = 0
            values.append(val)
    return pd.Series(values, index=timestamps)

def generate_erratic_pattern(timestamps, kpi_name):
    """
    Generate an erratic pattern using a random walk.
    
    • If the KPI name contains "cpu", then the values stay within 0–100 (suitable for percentages).
    • Otherwise, we use a random walk starting at 400.
    """
    values = []
    if "cpu" in kpi_name.lower():
        current_value = random.uniform(20, 80)
        for ts in timestamps:
            step = random.uniform(-5, 5)
            # Occasionally inject a larger spike/drop
            if random.random() < 0.01:
                step = random.uniform(-20, 20)
            current_value += step
            current_value = max(0, min(100, current_value))
            values.append(current_value)
    else:
        current_value = 400
        for ts in timestamps:
            step = random.uniform(-20, 20)
            if random.random() < 0.01:
                step = random.uniform(-100, 100)
            current_value += step
            current_value = max(0, current_value)
            values.append(current_value)
    return pd.Series(values, index=timestamps)

def generate_combination_pattern(timestamps, kpi_name):
    """
    Generate a combination pattern by taking a consistent pattern and adding occasional spikes.
    """
    base_series = generate_consistent_pattern(timestamps, kpi_name)
    values = []
    for val in base_series:
        # With a 3% chance, add a spike (or drop)
        spike = random.uniform(-50, 50) if random.random() < 0.03 else 0
        values.append(val + spike)
    return pd.Series(values, index=timestamps)

def generate_kpi_data(kpi_name, kpi_type, itsi_kpi_id, timestamps):
    """
    Generate a DataFrame for one KPI.
    
    The output DataFrame has three columns:
      • _time: the formatted timestamp (ISO format with milliseconds and timezone offset)
      • itsi_kpi_id: a unique id for the KPI
      • avg(alert_value): the generated value
    """
    if kpi_type == 'consistent':
        base_series = generate_consistent_pattern(timestamps, kpi_name)
    elif kpi_type == 'erratic':
        base_series = generate_erratic_pattern(timestamps, kpi_name)
    else:  # 'combination'
        base_series = generate_combination_pattern(timestamps, kpi_name)
    
    # Build a DataFrame with the desired three columns.
    # Format the timestamp as: "YYYY-MM-DDTHH:MM:SS.000±ZZZZ"
    df = pd.DataFrame({
        "_time": base_series.index.strftime("%Y-%m-%dT%H:%M:%S.000%z"),
        "itsi_kpi_id": itsi_kpi_id,
        "avg(alert_value)": base_series.values
    })
    return df

def ensure_output_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# -----------------------
# MAIN GENERATION SCRIPT
# -----------------------
def main():
    # 1. Generate a timezone-aware DatetimeIndex for the period.
    timestamps = generate_timestamps(START_DATE, DAYS, freq="H")
    
    # 2. Prepare the list of KPI names (repeat or truncate as needed).
    all_kpi_names = []
    while len(all_kpi_names) < NUM_KPIS:
        all_kpi_names.extend(KPI_NAMES)
    all_kpi_names = all_kpi_names[:NUM_KPIS]
    random.shuffle(all_kpi_names)
    
    # 3. Create a list of KPI types.
    kpi_types = (['consistent'] * CONSISTENT_COUNT +
                 ['erratic'] * ERRATIC_COUNT +
                 ['combination'] * COMBINATION_COUNT)
    if len(kpi_types) < NUM_KPIS:
        kpi_types.extend(['combination'] * (NUM_KPIS - len(kpi_types)))
    kpi_types = kpi_types[:NUM_KPIS]
    random.shuffle(kpi_types)
    
    # 4. Create the output directory.
    ensure_output_dir(OUTPUT_DIR)
    
    # 5. Generate data for each KPI and save.
    all_dfs = []
    for i in range(NUM_KPIS):
        kpi_name = all_kpi_names[i]
        kpi_type = kpi_types[i]
        itsi_kpi_id = generate_kpi_id()
        
        df_kpi = generate_kpi_data(kpi_name, kpi_type, itsi_kpi_id, timestamps)
        
        # Save each KPI's data to its own CSV file.
        filename = f"{itsi_kpi_id}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df_kpi.to_csv(filepath, index=False)
        print(f"Generated {filepath} for KPI '{kpi_name}' (type: {kpi_type}).")
        
        all_dfs.append(df_kpi)
    
    # (Optional) Combine all KPI data into one CSV file "model_test.csv".
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_filepath = os.path.join(OUTPUT_DIR, "model_test.csv")
    combined_df.to_csv(combined_filepath, index=False)
    print(f"\nCombined data written to {combined_filepath}")

if __name__ == "__main__":
    main()
