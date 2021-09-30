# Save Model Using joblib
import joblib
import numpy as np
import pandas as pd
import random
import os
from pprint import pprint
from decision_tree_functions import decision_tree_algorithm, decision_tree_predictions
from helper_functions import train_test_split, calculate_accuracy



try:
	os.remove("Trained_Model.sav")
	print("Old Forest removed. Training new Forest")
except:
	print('No Forest found. Creating new Forest.')


df = pd.read_csv("./datasets/Training_1.csv")
df["label"] = df.prognosis
df = df.drop("prognosis", axis=1)

column_names = []
for column in df.columns:
    name = column.replace(" ", "_")
    column_names.append(name)
df.columns = column_names

random.seed(0)
train_df = df

def bootstrapping(train_df, n_bootstrap):
    bootstrap_indices = np.random.randint(low=0, high=len(train_df), size=n_bootstrap)
    df_bootstrapped = train_df.iloc[bootstrap_indices]
    
    return df_bootstrapped

def random_forest_algorithm(train_df, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(train_df, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)
    
    return forest


forest = random_forest_algorithm(train_df, n_trees=10, n_bootstrap=800, n_features=132, dt_max_depth=5)


# save the model to disk
filename = 'Trained_Model.sav'
joblib.dump(forest, filename)

