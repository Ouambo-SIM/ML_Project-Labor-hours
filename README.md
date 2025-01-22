Investigation on Application of Machine Learning to Labor Hour Estimations

Company: Tindall Corporation
Duration: 4 months
Technologies: Python, Spyder, Pandas, openpyxl, Numpy, tensorflow, matplotlib, sklearn
Skills: Machine learning, Python, Data cleaning, Data Analysis, ERP Systems.

GitHub Link for full code: https://github.com/Ouambo-SIM/ML_Project-Labor-hours

In this project, labour hours is often referred to. Labor hours is the sum of work hours required per employee for the completion of a certain task. It is often used for estimating labour expenses in manufacturing.
Background:

Tindall products are usually different as, their design heavily depends on the architectural and structural requirements of the building they are made for. They vary in size, shape, and even components. However, in manufacturing, it is imperative to have a Labor hours estimation method for evaluating employee productivity and forecasting overtime. The evaluation and forecast help ensure the company is on track to meeting production goals, production deadlines and minimizing overtime, which is often more expensive compared to work during a normal 8-hour shift. 
Tindall uses routings as labour hour estimation method. Routings is a Macro-enabled excel file saved on the company’s server and used for reporting labour productivity and overtime. Routings mainly estimate labour hours using two sheets: entry sheet and extension sheet. 
Every product made has a routings file tailored specifically to its manufacturing process, which depends on its product type. 
The entry sheet is a predetermined list of characteristics(features) pertaining to a certain product type. Before manufacturing, production planning fills every scheduled product’s entry sheet with quantities for each item in the sheet.
 

Usually, products of the same type have the same extension sheet template.

The extension sheet of a product type is a table of all production steps(tasks) pertaining to that product type, the standard time for each step, unit of measure and predicted time for each step. Standard time is a sum of average task time over several samples and allowance (for fatigue or awkward handling, for example). Meanwhile, predicted time is calculated with respect to quantities entered in the entry sheet. For example, if the product at hand is a precast set of stairs with four vector connectors, in its routings, production planning will enter the number, 4, for the entry sheet item called “EA VECTOR CONNECTOR”, while the extension sheet in the routings will automatically multiply the number of vector connectors by standard time for the production step called place and tie vector connector. If for that step, standard time was 2.12 minutes, predicted time will evaluate to 8.48 minutes.
 The sum of all extension sheet predicted times gives the total estimated labour hours required to make a certain product.

 


Problem Statement:
-Routings entry sheet characteristics and extension sheet steps are detailed. Also, standard time estimations in the extension sheet come from time studies which account for each task’s completion process, how long tasks take, number of workers required and necessary allowances but, unfortunately, routings is not the most optimal labour hours estimation method.
To keep good labour hour estimates, constant standard time updates are required. However, constantly updating routings is a challenge as every product type’s routings has hundreds of extension sheet production steps, hence rendering it almost impossible to make the updates frequently enough. 
Moreover, production processes at Tindall are mostly manual, making time per production step very variable. Due to this, time studies for routings require a large sample set. Hence, making standard time updates even more of a daunting task.

Proposed Solution:
Feeding and training a neural network based on historical product characteristics and labour hours. Then, using the trained model to predict labour hours for future products.

Challenges:

-Production employees’ clock-in and clock-out records is the best source of past labour hours. However, there are situations where operators clock in and don't clock out, clock out without clocking in, and even clock in or clock out several times within 30 minutes on the same production day.

-Training data is very limited. Production is scheduled by form (bed or Workcenter) and operators frequently switch beds over the course of each production day. However, properly clocking into workstations did not happen until very recently. Hence rendering only four months of clocking data usable for the model training.





Process:
To train the neural network, I needed training samples constituting product characteristics and labour hours for each product characteristic combination per product. However, Tindall simultaneously produces several products(pieces) on each bed, and workers do not clock in or out according to products they make but the bed(form) they work on. So, for training, I made product characteristics samples the sum of product characteristics values (entry sheet values) for each set of pieces made on a specific form and date.
Product characteristics were collected from past routings’ entry sheets into a master entry data matrix while labour hours were collected from past employee clocking data into another matrix called Totals_matrix.

Data Cleaning
Using python and comparing two records at a time, I was able to clean the employee clock data to account for instances when employees did not clock in or clock out properly.

-Employee clocking data was first sorted by bed and date to ensure clocking issues are properly identified and resolved.

-For records where employees clocked in twice within a 30-minute interval, the latest clock in record was deleted. 

- For records where employees clocked out twice within a 30-minute interval, the earliest record was deleted, and the latest one kept.
  
-If an employee clocked in but did not clock out, the form number (bed number) and date of the clock in record was read into predefined variables and saved in a labour hour estimation dataframe with estimated labour hours of 8 hours, as it is the minimum shift length at Tindall. After that, the clock in record was deleted from the master clock data table.

-When an employee clocked out without clocking in, the from number (bed number) and date of the clock out record was read into predefined variables, and saved in the labour hour estimation data frame with estimated labour hours calculated as the difference in hours between the typical start time (shift start time) at the bed(form) concerned and the recorded clock out time. After that, the clock out record was deleted from the clock data table

-The difference between clock in and clock out times in the cleaned clocking data table was used to calculate total labour hours of each employee on each bed.

-Finally, Labor hours obtained from the cleaned clock table and estimated labour hours data frame were combined in Microsoft excel to generate an excel file of total labour hours spent on each form(bed) on each production day per employee. This provided labour hours that i later used to fill the Totals_Matrix (labour hour matrix) to be used for training.


Building Characteristics and Labor Hours Matrices

While making sure to check data types in data frames generated or modified along the way, the matrices were built as such:

-Started by exporting and filtering past production data from the company’s ERP system to get production data for the training and test set. Production data provided records on pieces(products) made, lot sizes(quantities) of the products, as well as dates and forms used for the pieces. 

-Read the production data into a dataframe

-Read the clock data excel file with cleaned and estimated labour hours for each employee. 

-Aggregated labour hours in clock data by bed(form) and date to obtain total labour hours spent on each bed and date combination.

-Since production on each bed(form) happens in groups, I used the production data table to generate a grouped production table where there are no duplicate combination of bed and date as all piece names (product IDs) produced on the same bed and date are concatenated and entered as a single value for products(pieces) produced while their respective lot sizes are also concatenated and entered as a single value for the Lot Size field in the grouped data table. 

-I then performed an inner join on bed and date between the Grouped production table and the clock data table(clocking) to get a Master production table with total labour hours for every piece(product) group made on each bed and date.

-Initialized matrices for aggregated entry data and Total labour hours 

-Defined a function to fetch and read entry sheets since entry sheets cannot be accessed directly. Routings for each product made by Tindall is stored in a project folder named with a project number pertaining to pieces the folder is for.  Piece names (product ID ) are made of the project for which it is for, the product type code and the width of the piece (For example, 42456DT-005). For each line in the master production table, each piece (product ID) has to be identified in the Piece field, then used to find and read the corresponding routings entry sheet and build the master entry data matrix.

-Processed every record in the master production table to build the master entry data matrix (while accounting for lot sizes) as well as the matrix for total labour hours.



Building the Neural Network:
- I defined a custom activation function to limit the network’s output to 200 hours and facilitate learning since it is impossible for labour hours per bed to exceed 200 hours on a given production day
  
-Built the neural network’s architecture using he uniform initializer as it is the most suitable for regression problems

-Compiled the model using the Adam optimizer, mean square error to quantify loss, and mean absolute error as metric for me to monitor the model’s performance during training and testing.

-Implemented early stop to prevent the model from training for too long and overfitting.

-Trained the model with a 20% validation set.

-Evaluated the model’s performance on the test set 



Outcome:
-30 labour hours loss during training

-Error on test set exceeded that on training set 
 


Next steps/ Thoughts:
-Product(piece) characteristics in entry sheet might not have a strong correlation with labour hours. The current characteristics will be evaluated for training suitability.

-Three months of training production data provided around 300 samples, which might not have been enough for the neural network to catch patterns and effectively learn for good labour hour prediction on the test set. For the continuation of the project, more training samples will be collected to improve training 

- Estimated labour hours from the clock data cleaning process might have had a negative impact on labour hours data per bed. About 20% of clocking data had to be cleaned due to employees improperly clocking in. For the continuation of the project better clocking data might improve model accuracy as well.

GitHub : https://github.com/Ouambo-SIM/ML_Project-Labor-hours

