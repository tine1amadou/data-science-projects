# Import Pandas for data manipulation
import pandas as pd
# Read the 'Call_Logs.csv' file into a DataFrame
df = pd.read_csv('Call_Logs.csv')
df.head()
# Preview one observation from the Logs Column
df.Logs[0]
# Define a function to extract date, time, and conversation details from the 'Logs' field
def extract_info(df):
    # Split the 'Logs' column content by new lines
    lines = df['Logs'].split('\n')
    
    # Extract date and time details from their respective lines
    date = lines[0].split(': ')[1]
    time = lines[1].split(': ')[1]
    
    # Extract conversation content starting from the fourth line onward
    conv = "\n".join([line for line in lines[3:] if line != ""])
    
    return date, time, conv
  # Apply 'extract_info' function to each row of the DataFrame to get structured columns
df[['Date', 'Time', 'Conversation']] = df.apply(extract_info, axis=1, result_type="expand")
df.head()
  # Drop the original 'Logs' column and any unnamed indices
df.drop(['Logs', 'Unnamed: 0'], axis=1, inplace=True)
df.head()
