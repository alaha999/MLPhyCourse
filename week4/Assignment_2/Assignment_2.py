####################################### Task 1 #######################################
# Import relevant packages. Install any modules if needed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_pdf import PdfPages

# One PDF for all graphs
outputname = 'GW_classification_week4.pdf'
pp = PdfPages(outputname)


####################################### Task 2 #######################################
''' 
Read any of the datafiles - GW_30k.csv, GW_200k.csv
The columns of the file are separated by commas and not tabs/space so make 
sure to set appropriate delimiters.

Tips:

1. If you are familiar with numpy.loadtxt then you can use that instead of csv. 
Make sure to load "GW_30k_nohead.csv" file. Make sure to remember which index corresponds to which column
link - https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html

2. If you prefer using pandas as in the previous assignment then feel free to use pd.read_csv().
link - https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html

3. If you want to try out the csv module then refer:
https://docs.python.org/3/library/csv.html 
https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html

### We request you to not change any variable names and leave already completed lines untouched ###
'''

# Tip: If you wish to comment multiple lines at once try enclosing the lines like ''' ... '''

# If you are using pandas.read_csv() then comment the other lines of code
file = ""      # Filename as string
cols = [0, 1, 2, 3, 4, 5, 6, 7]  # Columns to read
col_names = ['m1', 'm2', 'inc', 'pol', 'ra',
             'dec', 'SNR', 'dist']  # Column names

# Read using pandas.read_csv().
data_set =


# If you are using numpy.loadtxt() then comment the other lines of code
file = ""  # Filename

# Read using numpy.loadtxt()
data_set =


# If you are using csv.reader() comment the other lines of code
file = ""     # Filename
data_set = []           # Empty array to append datapoints to

# Each row is a datapoint. Read rows one-by-one and append the empty array
with open(file, newline='') as csv_file:
    # Define a reader object
    csv_reader =
    for row in csv_reader:
        # Add data points to the empty array

        # Array to numpy array. Note that this array contains only strings.
data_set =
data_set =     # Convert string array into float

print("Shape of our dataset (No-of-points, dimensions): {}".format(data_set.shape))


####################################### Task 3 #######################################
'''
Construct your Y_train and Y_val arrays and 
strip the X_train and X_val of its distance column.

Rules for constructing Y_train and Y_test 
1. dist < 600 Y = 0
2. dist > 600 Y = 1

Tips: 
1. If you are using pandas method then please take a look at pd.DataFrame.drop 
link - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop.html. 
When using pandas method make sure to pass "inplace" argument if needed. 
Another clue - axis = 1

2. If you are using the numpy method or csv method make sure to keep an eye out 
for the cols[] array from the previous block to know what each column represents. 
Take a look at numpy.delete() if you are finding it hard to delete columns
https://numpy.org/doc/stable/reference/generated/numpy.delete.html 
'''

# If you used pandas to prepare your arrays in the previous step then comment out all the other lines of code

# Add a new column which contains 0s and 1s according to the rule mentioned above
data_set['y_value'] =

# Store the 'y_value' column in a new array. This is so that we can split the array into train-val-test easily
Y = data_set['y_value']

# Delete 'dist' and 'y_value' columns


# If you used numpy.loadtxt() or csv method to read your data comment the other lines

# Define an empty-array/ 0-array that will correspond to each data-point
Y =
for i in range(data_set.shape[0]):
    # Add Y values according to the rules mentioned above

    # Delete 'dist' column
data_set =


# Spilt your data set into training set and test set.
# You might have to use the function from last assignment twice
x_train, x_test, y_train, y_test =
x_val, x_test, y_val, y_test =

# To get an idea of what number of datapoints we are working with
print("Training array - X: {} Y: {}".format(x_train.shape, y_train.shape))
print("Validation array - X: {} Y: {}".format(x_val.shape, y_val.shape))
print("Test array - X: {} Y: {}".format(x_test.shape, y_test.shape))


####################################### Task 4 #######################################
'''
Create your model

Links - 
1. Sequential() - https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
2. Dense() - https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
3. BatchNormalization() - https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
4. compile() - https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile
5. summary () - https://www.tensorflow.org/api_docs/python/tf/keras/Model#summary

Tips - 
1. Use axis = 1 for BatchNorm if you use BatchNorm at all
2. Use **adam** as your optimizer when calling compile()
3. use **binary_crossentropy** for loss
4. use **accuracy** for metrics
'''

# Store the number of features
n_features =
model = tf.keras.models.Sequential([
    # Add your layers
])

# Compile the model

# Try to get a summary. You might not get one until you train your network.
# If this throws an error then delete this line of code


####################################### Task 5 #######################################
'''
Train your model using model.fit()
link - https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

Tips - 
1. Make sure to use batches since we are dealing wtih huge numbers of data points.
2. Make sure to use **steps_per_epoch** argument and **batches** to get a smoother descent
3. Make sure your **batch_size * steps_per_epoch equals** the size of your train_data_set
4. Make sure your **batch_size * validation_steps equals** the size of your validation_data_set
5. Make use of EarlyStopping() if are not sure about the number of epochs
link - https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
'''

history = model.fit(
    # parameters
)

####################################### Task 6 #######################################
'''
Plot History
You can choose to simply leave this block as is and you will get results
If you really want to plot loss and accuracy yourself feel free to comment the following lines
Please keep the plt.savefig() lines undisturbed
'''

# You can choose to simply leave this block as is and you will get results
# If you really want to plot loss and accuracy yourself feel free to comment the following lines
# Please keep the plt.savefig() lines undisturbed

# Create a pandas Dataframe which would contain various columns (i.e., val_loss, loss, val_acc, acc etc)
# and values or those columns correspoding to each epoch
df_loss_acc = pd.DataFrame(history.history)

# Select the training and validation losses
df_loss = df_loss_acc[['loss', 'val_loss']]

# Renaming for clarity
df_loss.rename(
    columns={'loss': 'train', 'val_loss': 'validation'}, inplace=True)

# Selecting accruacies and renaming for clarity
df_acc = df_loss_acc[['accuracy', 'val_accuracy']]
df_acc.rename(columns={'accuracy': 'train',
              'val_accuracy': 'validation'}, inplace=True)

# Plotting and saving train_loss & val_loss vs epoch
df_loss.plot(title='Model loss', figsize=(12, 8)).set(
    xlabel='Epoch', ylabel='Loss')
plt.savefig(pp, format='pdf')

# Plotting and saving train_acc & val_acc vs epoch
df_acc.plot(title='Model Accuracy', figsize=(12, 8)).set(
    xlabel='Epoch', ylabel='Accuracy')
plt.savefig(pp, format='pdf')

# You can write your own code below


####################################### Task 7 #######################################
'''
Test your model and plot ROCs. 

You can pretty much copy paste the code from your last assignment here and it should work. Although you might have to change a few variable names here and there.

Links - 
1. model.evaluate() - https://www.tensorflow.org/api_docs/python/tf/keras/Model#evaluate
2. model.predict() - https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
'''

# Evaluate your model using the test_data_set
test_loss, test_acc =
print("Test_set performance\nTest_Loss = {}\tTest_accuracy = {}".format(
    test_loss, test_acc))

# Make DataFrames for training and validation
t_df =
v_df =

# Add truth and prediction columns to each DataFrame
t_df['train_truth'] =
t_df['train_prob'] = 0
v_df['val_truth'] =
v_df['val_prob'] = 0

# Now we evaluate the model on the train and validation data by calling the predict function
train_pred_proba =
val_pred_proba =

# Add the predictions to DataFrames
t_df['train_prob'] = train_pred_proba
v_df['val_prob'] = val_pred_proba


# Okay so now we have the two dataframes ready.
# t_df has two columns for training data  (train_truth and train_prob)
# v_df has two columns for testing data  (train_truth and train_prob)


# Now we plot the NN output
mybins = np.arange(0, 1.05, 0.05)

# Add the values to be made a histogram out of
plt.figure(figsize=(7, 5))
plt.hist('''val_truth = 1 ''', bins=mybins, histtype='step',
         label="Test Signal", linewidth=3, color='xkcd:green', density=False, log=False)
plt.hist('''val_truth = 0''', bins=mybins, histtype='step',
         label="Test Background", linewidth=3, color='xkcd:denim', density=False, log=False)
plt.hist('''train_truth = 1''', bins=mybins, histtype='step',
         label="Train Signal", linewidth=3, color='xkcd:greenish', density=False, log=False)
plt.hist('''train_truth = 0''', bins=mybins, histtype='step',
         label="Train Background", linewidth=3, color='xkcd:sky blue', density=False, log=False)
plt.legend(loc='upper center')
plt.xlabel('Score', fontsize=15)
plt.ylabel('Examples', fontsize=15)
plt.title(f'NN Output', fontsize=20)
plt.xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=10)
plt.yticks(fontsize=10)
plt.show()
plt.savefig(pp, format='pdf')


########

# Now we get the ROC curve, first for validation data
fpr, tpr, _ =
auc_score =
# Now the ROC curve for training data
fpr1, tpr1, _ =
auc_score1 =

# We plot the ROC curves together to assess the performance

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color='xkcd:denim blue',
         label='Testing ROC (AUC = %0.4f)' % auc_score)
plt.plot(fpr1, tpr1, color='xkcd:sky blue',
         label='Training ROC (AUC = %0.4f)' % auc_score1)
plt.legend(loc='lower right')
plt.title(f'ROC Curve', fontsize=20)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.xlim(0., 1.)
plt.ylim(0., 1.)
plt.show()
plt.savefig(pp, format='pdf')

pp.close()

########################################################################################
'''
## Good Job!
You have successfully completed the second assignment in this course. 

You can try out regression to find the exact distance or 
a multi-classifier with 3 classes where the distances are split as 
[500,566) [566,633) [633,700)
We have a optional assignment for a multi-classifier which you can find at 
Assignment_2_optional.py
link - 
'''
