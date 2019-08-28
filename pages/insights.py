import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app





column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ## Insights


            """
        ),
    ],
    md=4,
)

 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# !pip install --upgrade category_encoders pandas-profiling plotly
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
from ipywidgets import interact, fixed
import matplotlib.pyplot as plt


#!pip install --upgrade category_encoders pandas-profiling plotly
#import category_encoders as ce

# from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.pipeline import make_pipeline
import plotly.express as px

# read csv
df = pd.read_csv('https://data.humdata.org/dataset/9e3665cb-e4a3-4749-9622-f5c8146542c3/resource/c99f90a6-8a3b-4bd0-a081-3bed5d70781f/download/redcross_westafrica_rapidassessment_survey.csv')



# 3-way split: 60-20-20
train1, test = train_test_split(df, train_size=0.80, test_size=0.20, 
                               random_state=42) #stratify=df['ebola'],

# Need to duplicate target proportionate to # of infections (found in 'ebola')
# Check out: https://stackoverflow.com/questions/24029659/python-pandas-replicate-rows-in-dataframe

train, val = train_test_split(train1, train_size=0.75, test_size=0.25, 
                              random_state=42) #stratify=df['ebola'],

# A place to clean data...

def wrangle(X):
    # Wrangle train/val/test sets in the same way
    
    # Prevent SettingWithCopyWarning
    X = X.copy()
    
    # Imputes single 9999995 value, change target to catagory
    X['commebola'] = X.commebola.replace(999995, 0)
    X['commebola'] = X['commebola'].astype('category')
    
    # Drop unhelpful or data leaking features
    droprs = ['ebola', 'relgo', 'religiousmatch'] # 'comm_dist' ?
    X = X.drop(columns=droprs)
    
    # Chopping off all features with Permutation Importance less than zero 
    columns = ['commebola', 'Unnamed: 0', 'X', 'hh_gps.latitude', 'hh_gps.longitude', 'marketclean',
       'marketcomsum', 'marketmatch', 'hh_gps.altitude', 'hhnum',
       'comm_dist_oth', 'drink_wat', 'water_time', 'toilet', 'floor',
       'treat_time', 'birth_main', 'birth_loc', 'birth_time', 'comm_health',
       'ch_or3_num', 'ch_or3_time', 'floor_oth', 'ch_or1_time', 'ch_or4_time',
       'ch_or2_num', 'ch_or2_time', 'mark_other', 'finalend', 'drought',
       'fire', 'distnone', 'othdist', 'clinunsp', 'trad', 'lgmoh', 'RedCross',
       'cHWComm', 'WorkOth', 'fruit', 'wednesday', 'thursday', 'friday',
       'saturday', 'whmatch', 'relclean', 'instanceName', 'deviceId',
       'namevillage.1.altvillage', 'loc_adm1', 'loc_adm2', 'URBAN_OR_RURAL',
       'OTHER_DETAILS', 'comm_dist', 'treat_mult', 'birth_mult', 'marktype',
       'markday', 'border_crossing_type', 'border_crossing_name',
       'border_crossing_dest', 'epidemic_specifics', 'treat_loc_specify',
       'birmatron', 'loc_adm3', 'loc_adm4', 'District', 'Secteur_Rural',
       'name_village', 'name_part_village', 'QUARTIER_OU_DISTRICT',
       'alts_secteur_rurale.0.alt_secteur_rurale',
       'names_part_village.4.alt_part_village', 'alt_secteur_urbain', 'date',
       'alts_village.0.altvillage.1', 'alts_village.2.altvillage.1',
       'constituency', 'gsm_service', 'birhome']
    X = X[columns]
    
    # return the wrangled dataframe
    return X

# Wrangle each split 
train = wrangle(train)
val = wrangle(val)
test = wrangle(test)


# Arrange data into X features matrix and y target vector
target = 'commebola'
X_train = train.drop(columns=target)
y_train = train[target]

X_val = val.drop(columns=target)
y_val = val[target]

X_test = test.drop(columns=target)
y_test = test[target]


# Baseline Random Forest
pipeline = make_pipeline(
    ce.OrdinalEncoder(), 
    SimpleImputer(strategy='median'), 
    RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1,
                          ))

# Fit on train, score on val
pipeline.fit(X_train, y_train)


# y_pred = pipeline.predict(X_val)

# Just get column with index 1
y_pred_proba = pipeline.predict_proba(X_val)[:,1]


# y_pred = y_pred_proba > .1

# Plot a confusion matrix heatmap Ta-Da! 
def plot_confusion_matrix(y_true, y_pred):
    labels = unique_labels(y_true)
    columns = [f'Predicted {label}' for label in labels]
    index = [f'Actual {label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_true, y_pred), 
                         columns=columns, index=index)
    return sns.heatmap(table, annot=True, fmt='.0f', cmap='viridis')

  
# plot_confusion_matrix(y_val, y_pred);

def set_threshold(y_true, y_pred_proba, threshold=0.5):
    y_pred = y_pred_proba > threshold
    ax = sns.distplot(y_pred_proba)
    ax.axvline(threshold, color='red')
    plt.show() 
    plot_confusion_matrix(y_true, y_pred)
    # print(classification_report(y_true, y_pred))
    
interact(set_threshold, 
         y_true=fixed(y_val), 
         y_pred_proba=fixed(y_pred_proba), 
         threshold=(0,1,0.05));

column2 = dbc.Col(
    [
        interact(set_threshold, 
         y_true=fixed(y_val), 
         y_pred_proba=fixed(y_pred_proba), 
         threshold=(0,1,0.05))
    ]
)

layout = dbc.Row([column1, column2])