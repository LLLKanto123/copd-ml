# Import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            matthews_corrcoef, roc_auc_score, roc_curve, auc

# Load the dataset
data_df = pd.read_csv("copd_data.csv")

def fill_missing_values(data):
    """
    Imputes missing values in the dataset with the column mean.
   
    Args:
    data (DataFrame): The input DataFrame containing missing values.
   
    Returns:
    DataFrame: DataFrame with missing values filled.
    """
    # Iterate over the columns, excluding the first two and the last one
    for col in data.columns[2:-1]:
        if data[col].isnull().sum() > 0:
            mean_value = data[col].mean()
            data[col].fillna(mean_value, inplace=True)
            print(f"Filled missing values in '{col}' with mean: {mean_value}")
   
    return data

def load_data(data):
    """
    Cleans the dataset by filling missing values, then extracts features and target labels.
   
    Args:
        data (DataFrame): Input DataFrame to be cleaned and processed.
       
    Returns:
        tuple: A tuple containing the feature DataFrame and target Series.
    """
    # Fill missing values in the dataset
    data = fill_missing_values(data)
   
    # Select the features and target from the DataFrame
    features = data[['sex', 'age', 'smoke', 'bmi', 'location', 'rs10007052',
                     'rs8192288', 'rs20541', 'rs12922394', 'rs2910164',
                     'rs161976', 'rs473892', 'rs159497', 'rs9296092']]
   
    target = data['label']
   
    return features, target


# Function to check and convert feature columns to numeric
def convert_columns_to_numeric(df, feature_columns):
    """
    Converts specified feature columns to numeric, coercing errors to NaN.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_columns (list): List of columns to convert to numeric.

    Returns:
        pd.DataFrame: DataFrame with numeric feature columns.
    """
    # Convert feature columns to numeric, setting errors='coerce' to convert invalid entries to NaN
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors='coerce')

    # Print any remaining non-numeric columns (NaNs)
    if df[feature_columns].isnull().sum().any():
        print(f"Warning: Found NaNs in columns after conversion: \n{df[feature_columns].isnull().sum()}")
    
    return df

# Function to plot feature means
def plot_feature_means(df, feature_columns, target_column):
    """
    Plots the mean values of selected features for each class in the target column.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_columns (list): List of feature columns to plot.
        target_column (str): The target column to group by.
    """
    # Convert feature columns to numeric
    df = convert_columns_to_numeric(df, feature_columns)

    # Drop rows with NaN values in feature columns after conversion
    df = df.dropna(subset=feature_columns)

    # Check if the columns were converted properly
    print(f"Data types after conversion:\n{df[feature_columns].dtypes}")

    # Group by the target column and compute the mean for each class
    df_grouped = df.groupby(target_column).mean()[feature_columns]

    # Transpose and plot the data
    df_grouped.transpose().plot.barh(figsize=(10, 6), fontsize=12)
    plt.title('Mean Values of Features Grouped by Class', fontsize=16)
    plt.xlabel('Mean Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.show()

# Call the plotting function
plot_feature_means(data_df, feature_columns=['sex', 'age', 'bmi', 'smoke'], target_column='label')