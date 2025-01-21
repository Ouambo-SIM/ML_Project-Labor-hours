import pandas as pd
from datetime import datetime, timedelta

# Loading the data
df = pd.read_excel(r"C:\Users\Simo\OneDrive - Tindall Corporation\Desktop\MYPR vids NO TOUCH\TimeClockingsList_Oct12_Oct30.xlsx",sheet_name="Prod", usecols=["Employee Description","Date", "clock_type","Bed"])  # Clock times data
typical_start_df = pd.read_excel(r"C:\Users\Simo\OneDrive - Tindall Corporation\Desktop\MYPR vids NO TOUCH\ProductionTypical_Start Time.xlsx")  # Bed typical start times

# Ensuring 'Date' column is in datetime format and sort the data
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Employee Description', 'Date'], ascending=[True, True]).reset_index(drop=True)

# Creating an empty DataFrame for the estimated labor hours table
est_table = pd.DataFrame(columns=['bed', 'date', 'labor_hours'])

# Helper function to get typical start time for a bed
def get_typical_start(bed):
    result = typical_start_df[typical_start_df['Bed'] == bed]['Typical Start']
    return result.values[0] if not result.empty else None

# Iterating through the records to handle issues
i = 0
while i < len(df) - 1:
    current_row = df.iloc[i]
    next_row = df.iloc[i + 1]

    # ISSUE 1: Handling multiple clock-ins within 30 minutes
    if (current_row['clock_type'] == 'In Organization' and 
        next_row['clock_type'] == 'In Organization' and 
        current_row['Date'].date() == next_row['Date'].date() and
        (next_row['Date'] - current_row['Date']) <= timedelta(minutes=30)):  
        df.drop(i + 1, inplace=True)  # Drop the duplicate clock-in         
        df.reset_index(drop=True, inplace=True)
        continue 

    # ISSUE 2: Handling multiple clock-outs within 30 minutes
    elif (current_row['clock_type'] == 'Out Normal' and 
          next_row['clock_type'] == 'Out Normal' and 
          current_row['Date'].date() == next_row['Date'].date() and
          (next_row['Date'] - current_row['Date']) <= timedelta(minutes=30)): 
        df.drop(i, inplace=True)  # Keep the latest clock-out
        df.reset_index(drop=True, inplace=True)
        continue 

    # ISSUE 3: Handling clock-in without clock-out
    elif current_row['clock_type'] == 'In Organization' and (i == len(df) - 1 or next_row['clock_type'] != 'Out Normal'):
        new_row = pd.DataFrame([{
            'bed': current_row['Bed'],
            'date': current_row['Date'],
            'labor_hours': 8  # Assume 8 hours for missing clock-out
        }])
        est_table = pd.concat([est_table, new_row], ignore_index=True)
        df.drop(i, inplace=True)
        df.reset_index(drop=True, inplace=True)
        continue  

    # ISSUE 4: Handling clock-out without clock-in
    elif current_row['clock_type'] == 'Out Normal' and (i == 0 or df.iloc[i - 1]['clock_type'] != 'In Organization'):
        typical_start = get_typical_start(current_row['Bed'])
        if typical_start:
            
            #typical_start is of type 'time' and current_row['date'] is a 'datetime'
            current_date = current_row['Date'].date()  # Extracted date part
            typical_start_datetime = datetime.combine(current_date, typical_start)

            labor_hours = (current_row['Date'] - typical_start_datetime).total_seconds() / 3600
            
            
               
            new_row = pd.DataFrame([{
                'bed': current_row['Bed'],
                'date': current_row['Date'],
                'labor_hours': labor_hours
            }])
            est_table = pd.concat([est_table, new_row], ignore_index=True)
            
        df.drop(i, inplace=True)
        df.reset_index(drop=True, inplace=True)
        continue  

    # Moving to the next row
    i += 1

# Saving the cleaned data and estimated labor table
df.to_excel('cleaned_clock_times_TEST.xlsx', index=False)
est_table.to_excel('Labor_Hours_Est_Table_TEST.xlsx', index=False)

print("Data cleaning and estimation completed.")
print("Beginning phase 1 of labor hours calculations. Don't forget to combine result with labor_Hours_Est table")

Cleaned_clock_times= df
Total_Labor_Hours= pd.DataFrame(columns=['Bed', 'Date', 'Labor_Hours'])

i = 0
while i < len(df) - 1:
    current_row = df.iloc[i]
    next_row = df.iloc[i + 1]
    
    labor_hours = (next_row['Date'] - current_row['Date']).total_seconds() / 3600
    
    new_row = pd.DataFrame([{
        'Bed': current_row['Bed'],
        'Date': current_row['Date'].date(),
        'Labor_Hours': labor_hours
    }])
    
    Total_Labor_Hours = pd.concat([Total_Labor_Hours, new_row], ignore_index=True)
    
    
    i += 2
    

Total_Labor_Hours.to_excel('Labor_Hours_without_est_TEST.xlsx',index=False)