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
from xgboost import plot_importance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
                            matthews_corrcoef, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings

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

# Execute the data loading process
features, target = load_data(data_df)

def prepare_data(data_df):
    """
    Prepares train and test sets by splitting and shuffling the data.
   
    Args:
        data_df (pd.DataFrame): Input dataframe with a 'class' column to split data.
       
    Returns:
        train_df, test_df (pd.DataFrame, pd.DataFrame): Shuffled training and test dataframes.
    """
    # Split the data into train and test sets based on the 'class' column
    train_df = data_df[data_df['class'] == 'train']
    test_df = data_df[data_df['class'] == 'validate']

    # Shuffle the data for randomness
    train_df = shuffle(train_df, random_state=42)
    test_df = shuffle(test_df, random_state=42)

    return train_df, test_df

def get_classifiers(pos_weight):
    """
    Initializes a list of classifiers with default parameters.
   
    Args:
        pos_weight (float): Weight to be used in XGBoost classifier.
       
    Returns:
        list, list: List of classifiers and their names.
    """
    knn = KNeighborsClassifier()
    svm = SVC(probability=True, random_state=42)
    lr = LogisticRegression(random_state=42, penalty='l1', C=10, max_iter=50, intercept_scaling=10, solver='liblinear')
    tree = DecisionTreeClassifier(random_state=42)
    nn = MLPClassifier(hidden_layer_sizes=(4, 4, 4), random_state=42)
    xgboost = xgb.XGBClassifier(scale_pos_weight=pos_weight, random_state=42, learning_rate=0.05,
                                n_estimators=5000, max_depth=500, reg_alpha=0.25, subsample=1, eval_metric="auc")

    classifiers = [knn, lr, svm, tree, nn, xgboost]
    classifier_names = ['KNN', 'LR', 'SVM', 'DT', 'MLP', 'XGBoost']

    return classifiers, classifier_names

def make_baseline(train_df, test_df, classifier):
    """
    Trains the classifier and evaluates its performance using various metrics.
   
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        classifier (classifier object): The classifier to train.
       
    Returns:
        tuple: Performance metrics (accuracy, precision, recall, f1 score, auc score, mcc score).
    """
    X_train, y_train = load_data(train_df)
    X_test, y_test = load_data(test_df)
   
    # Train the classifier
    classifier.fit(X_train, y_train)
   
    # Predictions and probabilities
    pred_proba = classifier.predict_proba(X_test)
    pred = classifier.predict(X_test)
   
    # Calculate performance metrics
    acc = accuracy_score(y_test, pred)
    precisions = precision_score(y_test, pred)
    recalls = recall_score(y_test, pred)
    f1s = f1_score(y_test, pred)
    mcc = matthews_corrcoef(y_test, pred)
   
    aucs1 = roc_auc_score(y_test, pred)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:, 1])
    aucs = auc(fpr, tpr)

    return acc, precisions, recalls, f1s, aucs, mcc

def train_models(train_df, test_df, classifiers, classifier_names):
    """
    Trains and evaluates each classifier in the list, and collects performance metrics.
   
    Args:
        train_df (pd.DataFrame): Training data.
        test_df (pd.DataFrame): Test data.
        classifiers (list): List of classifier objects.
        classifier_names (list): Names of classifiers for labeling results.
       
    Returns:
        pd.DataFrame: DataFrame with all performance metrics.
    """
    acc_mean_lst = []
    precisions_mean_lst = []
    recalls_mean_lst = []
    f1s_mean_lst = []
    auc_mean_lst = []
    mcc_mean_lst = []

    # Train each classifier and collect performance metrics
    for classifier in classifiers:
        acc, precisions, recalls, f1s, aucs, mcc = make_baseline(train_df, test_df, classifier)
       
        acc_mean_lst.append(acc)
        precisions_mean_lst.append(precisions)
        recalls_mean_lst.append(recalls)
        f1s_mean_lst.append(f1s)
        auc_mean_lst.append(aucs)
        mcc_mean_lst.append(mcc)

    # Store results in a DataFrame
    all_para = {
        "accuracy": acc_mean_lst,
        "precision": precisions_mean_lst,
        "recall": recalls_mean_lst,
        "f1score": f1s_mean_lst,
        "mcc_score": mcc_mean_lst,
        "aucscore": auc_mean_lst
    }

    return pd.DataFrame(all_para, index=classifier_names)

# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

# Function to plot ROC curve
def plot_ROC(model, model_name, color, X_train, y_train, X_test, y_test):
    """
    Plots ROC curve for a given model.
    
    Args:
        model (classifier object): Classifier to evaluate.
        model_name (str): Name of the classifier.
        color (str): Color for the plot line.
        X_train, y_train: Training features and target.
        X_test, y_test: Testing features and target.
    """
    # Train the model and get predicted probabilities
    model.fit(X_train, y_train)
    probas_ = model.predict_proba(X_test)
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve
    plt.plot(fpr, tpr, color=color,
             label=f'{model_name} (AUC = {roc_auc:.2f})',
             lw=2, alpha=.8)

# Main function to run the pipeline
if __name__ == "__main__":
    # Prepare data (assuming you have defined prepare_data and load_data as per the previous code)
    train_df, test_df = prepare_data(data_df)

    # Load training and testing data
    X_train, y_train = load_data(train_df)
    X_test, y_test = load_data(test_df)

    # Get classifiers and their names
    pos_weight = 0.355
    classifiers, classifier_names = get_classifiers(pos_weight)

    # Plot ROC curves for each classifier
    plt.figure(figsize=(10, 6))
    color_list = ['b', 'g', 'r', 'c', 'm', 'y']
    
    for model, model_name, color in zip(classifiers, classifier_names, color_list):
        plot_ROC(model, model_name, color, X_train, y_train, X_test, y_test)
    
    # Plot diagonal reference line for random guess
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', alpha=0.8)

    # Set plot labels and title
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multiple Models')
    plt.legend(loc="lower right")
    
    # Show the plot
    plt.show()  

def load_data_for_xgb(train_df):
    """
    Prepares the data specifically for XGBoost by loading the features and labels.
   
    Args:
        train_df (pd.DataFrame): DataFrame containing the training data.
   
    Returns:
        tuple: Features and labels (train_x, train_y).
    """
    train_x, train_y = load_data(train_df)

def train_xgb_model(train_x, train_y):
    """
    Trains an XGBoost model on the training data.
    
    Args:
        train_x (DataFrame): Training features.
        train_y (Series): Training labels.
    
    Returns:
        xgb_model (XGBClassifier): Trained XGBoost model.
    """
    xgb_model = xgb.XGBClassifier(random_state=42, learning_rate=0.05,
                                  n_estimators=5000, max_depth=500, 
                                  reg_alpha=0.25, subsample=1, eval_metric="auc")
    
    # Train the model
    xgb_model.fit(train_x, train_y)
    
    return xgb_model

def plot_feature_importance(model):
    """
    Plots the feature importance of the trained XGBoost model.
    
    Args:
        model (XGBClassifier): Trained XGBoost model.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_importance(model, ax=ax, importance_type="total_gain", show_values=True)
    plt.title('Feature Importance (Total Gain)')
    plt.show()

def display_importance_scores(model):
    """
    Prints the feature importance scores from the trained XGBoost model for different types.
    
    Args:
        model (XGBClassifier): Trained XGBoost model.
    """
    importance_types = ('weight', 'gain', 'cover', 'total_gain', 'total_cover')
    
    for importance_type in importance_types:
        print(f'{importance_type}:\n', model.get_booster().get_score(importance_type=importance_type), "\n")

# Main function to integrate into the existing pipeline
if __name__ == "__main__":
    # Load and prepare training data (from the previously defined prepare_data function)
    train_df, test_df = prepare_data(data_df)
    train_x, train_y = load_data_for_xgb(train_df)

    # Train the XGBoost model
    xgb_model = train_xgb_model(train_x, train_y)

    # Plot feature importance
    plot_feature_importance(xgb_model)

    # Display importance scores for debugging
    display_importance_scores(xgb_model)