# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:14:47 2024

@author: Simo
"""
import os
import pandas as pd
import numpy as np
import openpyxl

dataset=0
while dataset<2:

   # Load Clock tables individually based on dataset index
   if dataset == 0:
        clock_data = pd.read_excel(r"C:\Users\Simo\OneDrive - Tindall Corporation\Desktop\MYPR vids NO TOUCH\Labor_Hours_PLUS_Estimate.xlsx")
   else:
        clock_data = pd.read_excel(r"C:\Users\Simo\OneDrive - Tindall Corporation\Desktop\MYPR vids NO TOUCH\Labor_Hours_PLUS_est_TEST.xlsx")

    # Aggregating Labor hours by bed and date
   Clock_Table = clock_data.groupby(["Bed", "Date"]).agg({"Labor_Hours": 'sum'}).reset_index()
   print("Clock_Table columns after aggregation:", Clock_Table.columns)
    
   #Load and Aggregate Clock-in data
   #clock_table=pd.read_excel(r"C:\Users\Simo\OneDrive - Tindall Corporation\Desktop\MYPR vids NO TOUCH\Labor_Hours_PLUS_Estimate.xlsx", usecols=["Bed","Date", "Labor_Hours"])# This is a table that shopws what Bed and Date every worker was producing on as well as the time spent on the beds on those days
   #clock_table_test=pd.read_excel(r"C:\Users\Simo\OneDrive - Tindall Corporation\Desktop\MYPR vids NO TOUCH\Labor_Hours_PLUS_est_TEST.xlsx", usecols=["Bed","Date", "Labor_Hours"])
   
   

   # Read prod_table from Excel file
   prod_table=pd.read_excel(r"C:\Users\Simo\OneDrive - Tindall Corporation\Desktop\MYPR vids NO TOUCH\Prod_data_July2024_Oct24.xlsx", usecols=["Bed","Earliest Start Date", "piece", "Lot_Size"])
   prod_table_test=pd.read_excel(r"C:\Users\Simo\OneDrive - Tindall Corporation\Desktop\MYPR vids NO TOUCH\Testset_ProdData_Oct12_Oct30.xlsx", usecols=["Bed","Earliest Start Date", "piece", "Lot_Size"])

   #CLOCKS=[clock_table, clock_table_test]

   PROD=[prod_table,prod_table_test]

   #Aggregating Labor hours by bed and date
   #  Clock_Table = CLOCKS[dataset].groupby(["Bed", "Date"]).agg({"Labor_Hours":'sum'}).reset_index(drop=True) 
   
   # Check columns to ensure 'Bed' and 'Date' are present
   #  print("Clock_Table columns:", Clock_Table.columns)

   #sorting production table
   prod_table=PROD[dataset].sort_values(by=["Bed","Earliest Start Date",]).reset_index(drop=True)
   

   print(prod_table.head(10))  # Inspect the initial rows
 
   #Initialize grouping arrays

   grouped_piece, grouped_date, grouped_bed, grouped_Lot_Size=[], [], [], []

   # Ensure correct initialization for the first row
   first_row = prod_table.iloc[0]
   date_choice, bed_choice, piece, Lot_Size = first_row["Earliest Start Date"], first_row["Bed"], str(first_row["piece"]), str(first_row["Lot_Size"])


   #Construct piece lists and lot size lists for each pour
   
   for index, row in prod_table.iloc[1:].iterrows(): 
 
 
    #assuming prod table dates are sorted by earliest start date 
    if row['Earliest Start Date']==date_choice and row['Bed']==bed_choice:
     
        piece+= "," +str(row['piece'])
        Lot_Size+= "," + str(row['Lot_Size'])
    else:
     
        grouped_date.append(date_choice)
        grouped_bed.append(bed_choice)
        grouped_piece.append(piece)  
        grouped_Lot_Size.append(Lot_Size)
     
        date_choice, bed_choice, piece, Lot_Size = row['Earliest Start Date'], row['Bed'], str(row['piece']), str(row['Lot_Size'])
     
     
   #append last group    
   grouped_date.append(date_choice)
   grouped_bed.append(bed_choice)
   grouped_piece.append(piece)
   grouped_Lot_Size.append(Lot_Size)      
       
   #Create DataFrame from grouped data  
   Groupedtable_df = pd.DataFrame({'Bed': grouped_bed,'Date': grouped_date, 'PIECES': grouped_piece, 'LOT_SIZE': grouped_Lot_Size})


   Groupedtable_df.to_excel('Groupedtable_df.xlsx',index=False)

   #Ensure data types are expected
   Groupedtable_df['Bed'] = Groupedtable_df['Bed'].astype(np.int64)
   Groupedtable_df['PIECES'] = Groupedtable_df['PIECES'].astype(str)

   #Join Clock_Table to Groupedtable_df and filter according to bed 
   Master_ProdTable = pd.merge(Groupedtable_df,Clock_Table, on=['Bed','Date'], how='inner')
   mask = Master_ProdTable["Bed"].isin([420,402,410,405,]) ####Choose beds according to product type
   Master_ProdTable= Master_ProdTable[mask]    
   
       
   Master_ProdTable= Master_ProdTable.reset_index(drop=True) #Table of all beds on each day with total labor hours, pices and lot sizes

   Master_ProdTable.to_excel('Master production vs labor hours table.xlsx',index=False)

   print("Master Prod table indexes--------")
   print(Master_ProdTable.index) 
   
   #Initialize matrices for total time and aggregated entry data for training and test set
   if dataset==0:
       No_Samples_Train= len(Master_ProdTable)
       
       Totals_matrix=np.zeros((No_Samples_Train,1))  
       Master_entrydata=np.zeros((No_Samples_Train,116)) #######This is the aggregated entry data matrix for individual production days. 
   else:
       No_Samples_Test= len(Master_ProdTable)
       Totals_matrix_Test=np.zeros((No_Samples_Test,1))  
       Master_entrydata_Test=np.zeros((No_Samples_Test,116))
   
   
   entrydata_list=[]
   
   #Function for finding entry sheets on PC
   
   def fetch_entry_sheet(piece_name, base_directory):
       """
       Fetch and read the Excel file from the appropriate project folder.
    
       Parameters:
       - piece_name: str, The piece name in the format '34265DT-008'
       - base_directory: str, The base directory where project folders are located
    
       Returns:
       - pd.DataFrame: The contents of the entry sheet if found, otherwise None.
       """
       
       # Extract project number and part number from piece_name
       #project_number = piece_name.split('DT-')[0]  # '34265'
       #part_number = 'DT-' + piece_name.split('DT-')[1]  # 'DT-008'
       
       # Extract project number as the first five characters
       project_number = piece_name[:5]
    
       # Extract part number as everything after the first five characters
       part_number = piece_name[5:]

       # Build the path to the project folder and the entry sheet file
       project_folder = os.path.join(base_directory, project_number)
       entry_sheet_path = os.path.join(project_folder, f"{part_number}.xlsm")

       try:
           # Attempt to read the Excel file into a DataFrame
           df = pd.read_excel(entry_sheet_path,sheet_name= "ENTRY", engine='openpyxl')
           print(f"Successfully read: {entry_sheet_path}")
           return df
       except FileNotFoundError:
           print(f"Error: File not found -> {entry_sheet_path}")
           return None
       except Exception as e:
           print(f"An error occurred: {e}")
           return None


   #Directory with entry sheet folders
   base_directory= r"C:\Users\Simo\OneDrive - Tindall Corporation\Desktop\MYPR vids NO TOUCH\Quest Routings"#########write directory where all the entry sheet data will be stored
   Missing_entrysheets=[]
    
   print("Entering the folder-processing loop")

   #Processing each row in the Master_ProdTable
   for index, row in Master_ProdTable.iterrows(): 
    
     print("Entered the folder-processing loop")   
    
     print(f"Processing row {index} with pieces: {row['PIECES']}")
  
     #Float List of Lot_Sizes and array item locator initialization
     LotSize_list= np.array(row['LOT_SIZE'].split(",")) #string array
     print(f"Lots found in current row: {LotSize_list}. Now converting to float")
     LotSize_Scalars=[float(element) for element in LotSize_list]# float list of lot sizes
     M=0 #index counter for LotSize_List
  
     #list of pieces to filter by  
     filter_criterion = [piece.strip() for piece in row['PIECES'].split(',')] 
  
     #Stores Labor Hours in the Totals Matrix 
     if dataset==0:
         
       if isinstance(row["Labor_Hours"], float): 
         Totals_matrix[index]= row['Labor_Hours'] 
     else:
       if isinstance(row["Labor_Hours"], float):  
         Totals_matrix_Test[index]=row['Labor_Hours'] 
        
        
     #initializing array for aggregated entry data 
     masterrow_entrydata=np.zeros((116,)) #######find the number of elements to read from entry sheet
  
  
     #processing each pour's entry sheets
     for piece in filter_criterion:
        
           #Scalar to account for several pieces with same mark number
        
           scalar_choice= LotSize_Scalars[M]
           M+=1
        
           # Loading entry sheet data for the group of pieces and dropping rows
      
           entrysheet_data= fetch_entry_sheet(piece, base_directory) # accessing entry sheets for the group of pieces
           print("number of columns-------------")
           print(len(entrysheet_data.columns))
           try:
               entrysheet_data.drop([9,15,34,50,56,72,78,82,96,100,103,124], inplace=True)########ENTER INFO
               entrysheet_data.reset_index(drop=True, inplace=True)
        
           except Exception as e:
               print(f"Error processing entry sheet for {piece}: {e}")
               Missing_entrysheets.append(piece)
               continue
        
           c=1
        
           if c==1:
             print("----------entrysheet dataframe-------")
             entrysheet_data.to_excel("Entry data dataframe.xlsx", index=False)  
             c=2
          
           #Extracting and aggregating data from entry sheet
           entrydata_list=[]
           for idx, entry_row in entrysheet_data.iterrows():
            
               #Difining the range of values to be read
               if len(entrydata_list) > 116:
                   break
          
               if idx<6 : ###### find the row number where entry sheet data actually starts#####
                   continue
            
            
               if pd.isna(entry_row.iloc[2]): ######## marking empty values as zero. verify column location
                   entrysheet_data.iloc[idx, 2]=0#######verify location
                   entrydata_list.append(0)###### verify location
              
               else:    
                 try:
                   entrydata_list.append(float(entrysheet_data.iloc[idx,2])) #######appending entry sheet data into a list verify location
                 except (ValueError, TypeError):
                  # If conversion fails, append zero
                   entrydata_list.append(0)
                   print(f"Non-numeric value found at row index {idx}: {entrysheet_data.iloc[idx,2]}. Recorded as zero.")  
                
                
           if entrydata_list:
               
               entrydata_list = entrydata_list[:116]
               print(f"entrysheet data found: {entrydata_list}")
            
               for i, value in enumerate(entrydata_list):
                 print(f"Index {i}: {repr(value)}")
               entrydata_array=np.array(entrydata_list) # converting list to array for aggregation
       
               if entrydata_array.shape[0] == masterrow_entrydata.shape[0]:
                 masterrow_entrydata+= scalar_choice*entrydata_array # aggregating entry sheet data for the group of pieces using scalar_choice
      
               else:
                 print(f"Warning: entrydata_array shape {entrydata_array.shape} does not match masterrow_entrydata shape {masterrow_entrydata.shape}")
            
           
         
     #Storing aggregated entry data into master matrix 
     masterrow_matrix=np.reshape(masterrow_entrydata,(1,116)) #To ensure consistent data structure 
     if dataset==0:
       Master_entrydata[index]=masterrow_matrix.flatten() # inputing the aggregates of entry data into a master matrix before moving to the next set of pieces for further building of the matrix
     else:
       Master_entrydata_Test[index]=masterrow_matrix.flatten() 
  
  
   #end of entry sheet folder data colection from iterations
   
   #Review output of data manipulation
   
   if dataset==0:
     
     print("Totals_matrix:\n", Totals_matrix)
     print("Master_entrydata:\n", Master_entrydata)
     np.savetxt("Missing_EntrySheets.txt", np.array(Missing_entrysheets, dtype=str), fmt="%s", delimiter=",")
     np.savetxt("Master_entrydata_matrix.txt", Master_entrydata, fmt="%s", delimiter=",")
   else: 
     
     print("Totals_matrix_TEST:\n", Totals_matrix_Test)
     print("Master_entrydata_TEST:\n", Master_entrydata_Test)
     np.savetxt("Missing_EntrySheets_Testset.txt", np.array(Missing_entrysheets, dtype=str), fmt="%s", delimiter=",")
     np.savetxt("Master_entrydata_matrix_Test.txt", Master_entrydata_Test, fmt="%s", delimiter=",")
   dataset+=1  
     
print("Totals_matrix:\n", Totals_matrix)
print("Master_entrydata:\n", Master_entrydata)     
print("Shape of Master_entrydata:", Master_entrydata.shape)
print("Shape of Master_entrydataTEST:", Master_entrydata_Test.shape)
     
#Model

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler

# Custom activation function to scale sigmoid output to (0, 200)
def scaled_sigmoid(x):
    return 200 * tf.keras.activations.sigmoid(x)


# Scale the features for better convergence(normalizing)
scaler = StandardScaler()
Master_entrydata = scaler.fit_transform(Master_entrydata)
Master_entrydata_test= scaler.transform(Master_entrydata_Test)

                                            
# Define the model
model = keras.Sequential([
    layers.Dense(256, activation='relu',kernel_initializer='he_uniform', input_shape=(116,)),  # Input layer with 116 features
    Dropout(0.3),  # Drop 30% of the neurons to prevent overfitting
    layers.Dense(128, activation='relu',kernel_initializer='he_uniform'),  # Hidden layer with 128 neurons
    Dropout(0.3),  # Dropout to improve generalization
    layers.Dense(64, activation='relu',kernel_initializer='he_uniform'),  # Hidden layer with 64 neurons
    Dropout(0.3),
    layers.Dense(32, activation='relu',kernel_initializer='he_uniform'),
    layers.Dense(1, activation='linear')  # Output layer for regression. linear activation will display overtime as well.
])
#      layers.Dense(1, activation=scaled_sigmoid)# Output layer with scaled sigmoid activation just as an alternative

# Compile the model
model.compile(optimizer='adam',
              loss='mean_squared_error',# Use mean squared error to update weights during training
              metrics=['mae'])

#Implementing early stop
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)#patience determines the number of epochs after which the model will stop training once val loss starts plateauing


# Train the model
history = model.fit(Master_entrydata, Totals_matrix, epochs=50, validation_split=0.2, verbose=1, callbacks=[early_stopping])


# Plot the loss curves
plt.figure(figsize=(10, 6))

# Plot training loss and validation loss with MSE
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss(MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# Plot training and validation MAE(more intuitive)
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error (MAE)')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# Print model summary
print("-----------------BELOW IS THE MODEL SUMMARY------------------")
model.summary()

# Evaluate the model
print("Now evaluating the model on the test set:")
test_loss=model.evaluate(Master_entrydata_Test, Totals_matrix_Test)
print(f"Test Loss: {test_loss}")

