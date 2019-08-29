import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sns
import matplotlib.pyplot as plt

from plotly.tools import mpl_to_plotly


#import category_encoders as ce

# from sklearn.impute import SimpleImputer

import plotly.express as px



# df = pd.read_csv('https://data.humdata.org/dataset/9e3665cb-e4a3-4749-9622-f5c8146542c3/resource/c99f90a6-8a3b-4bd0-a081-3bed5d70781f/download/redcross_westafrica_rapidassessment_survey.csv')

# # 3-way split: 60-20-20
# train1, test = train_test_split(df, train_size=0.80, test_size=0.20, 
#                                random_state=42) #stratify=df['ebola'],

# # Need to duplicate target proportionate to # of infections (found in 'ebola')
# # Check out: https://stackoverflow.com/questions/24029659/python-pandas-replicate-rows-in-dataframe

# train, val = train_test_split(train1, train_size=0.75, test_size=0.25, 
#                               random_state=42) #stratify=df['ebola'],

# # A place to clean data...

# def wrangle(X):
#     # Wrangle train/val/test sets in the same way
    
#     # Prevent SettingWithCopyWarning
#     X = X.copy()
    
#     # Imputes single 9999995 value, change target to catagory
#     X['commebola'] = X.commebola.replace(999995, 0)
#     X['commebola'] = X['commebola'].astype('category')
    
#     # Drop unhelpful or data leaking features
#     droprs = ['ebola', 'relgo', 'religiousmatch'] # 'comm_dist' ?
#     X = X.drop(columns=droprs)
    
#     # Chopping off all features with Permutation Importance less than zero 
#     columns = ['commebola', 'Unnamed: 0', 'X', 'hh_gps.latitude', 'hh_gps.longitude', 'marketclean',
#        'marketcomsum', 'marketmatch', 'hh_gps.altitude', 'hhnum',
#        'comm_dist_oth', 'drink_wat', 'water_time', 'toilet', 'floor',
#        'treat_time', 'birth_main', 'birth_loc', 'birth_time', 'comm_health',
#        'ch_or3_num', 'ch_or3_time', 'floor_oth', 'ch_or1_time', 'ch_or4_time',
#        'ch_or2_num', 'ch_or2_time', 'mark_other', 'finalend', 'drought',
#        'fire', 'distnone', 'othdist', 'clinunsp', 'trad', 'lgmoh', 'RedCross',
#        'cHWComm', 'WorkOth', 'fruit', 'wednesday', 'thursday', 'friday',
#        'saturday', 'whmatch', 'relclean', 'instanceName', 'deviceId',
#        'namevillage.1.altvillage', 'loc_adm1', 'loc_adm2', 'URBAN_OR_RURAL',
#        'OTHER_DETAILS', 'comm_dist', 'treat_mult', 'birth_mult', 'marktype',
#        'markday', 'border_crossing_type', 'border_crossing_name',
#        'border_crossing_dest', 'epidemic_specifics', 'treat_loc_specify',
#        'birmatron', 'loc_adm3', 'loc_adm4', 'District', 'Secteur_Rural',
#        'name_village', 'name_part_village', 'QUARTIER_OU_DISTRICT',
#        'alts_secteur_rurale.0.alt_secteur_rurale',
#        'names_part_village.4.alt_part_village', 'alt_secteur_urbain', 'date',
#        'alts_village.0.altvillage.1', 'alts_village.2.altvillage.1',
#        'constituency', 'gsm_service', 'birhome']
#     X = X[columns]
    
#     # return the wrangled dataframe
#     return X

# # Wrangle each split 
# train = wrangle(train)
# val = wrangle(val)
# test = wrangle(test)


# # Arrange data into X features matrix and y target vector
# target = 'commebola'
# X_train = train.drop(columns=target)
# y_train = train[target]

# X_val = val.drop(columns=target)
# y_val = val[target]

# X_test = test.drop(columns=target)
# y_test = test[target]


# # Baseline Random Forest
# pipeline = make_pipeline(
#     ce.OrdinalEncoder(), 
#     SimpleImputer(strategy='median'), 
#     RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1,
#                           ))

# # Fit on train, score on val
# pipeline.fit(X_train, y_train)


# # y_pred = pipeline.predict(X_val)

# # Just get column with index 1
# y_pred_proba = pipeline.predict_proba(X_val)[:,1]

# @app.callback(
#     Output('myGraph', 'children'),
#     [Input('threshold', 'value'), 
#     #  Input('bathrooms', 'value'), 
#     #  Input('longitude', 'value'), 
#     #  Input('latitude', 'value')
#     ]
# )

# # Visualizing moving threshold...
# def visualiz_it(threshold):
#     fig = sns.distplot(y_pred_proba)
#     fig.axvline(threshold, color='red')
#     return fig

# fig = sns.distplot(y_pred_proba)


# Part of my attempt to make a seaborn an output
# Here I'm trying to convert the returned figure from my visualiz_it func to a plotly object
# plotly_fig = mpl_to_plotly(fig)
# graph = dcc.Graph(id='myGraph', fig=plotly_fig)


column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
            ### Model Metrics

            The plot to the right represents the distribution of infection
            probabilities. Note the right skew of the data; only 0.5% 
            of communities surveyed had cases of Ebola infection. Even at an initial 
            96% "accuracy", the predictive model is insufficiently tuned to identify 
            communities at risk. 
            
            
            To properly adjust for the low probability of infection, the predicitve threshold of the model
            was set to 10%. Communities identified as at risk are up to 90%
            unlikely to have an infect.
             

            Below, adjust the model threshold to see the effect on model predictions. 


            """
        ),
        
        dcc.Markdown(
            '''
            ##### Recall / Precision Tradeoff
            
            Adjusting the predictive threshold way down to 10%
            greatly improves recall at the cost of model precision. 
            

            Even so, for the sake of containment, efforts can be more efficiently focused on 
            85%
            fewer locations. Effectively concentrating efforts on communities the highest risk. 
            

            '''
        ), 
        # Attempt to make the seaborn distplot an output graph...
        # dcc.Graph(figure='plot'
        # ),
    ],
    md=4,
)

# Wasn't going to work... 
# interact(set_threshold, 
#          y_true=fixed(y_val), 
#          y_pred_proba=fixed(y_pred_proba), 
#          threshold=(0,1,0.05));

column2 = dbc.Col(
    [
     dcc.Markdown(
            """
        
            #### Probability Distribution with Threshold line


            """
            
        ),
        html.Img(src='assets/distribution10.png', className='img-fluid'
        ),
        dcc.Markdown('#### Confusion Table Heatmap: predicted vs. actual'),
        html.Img(src='assets/heatmap10.png', className='img-fluid'
        ),   
        dcc.Markdown('#### Adjust Threshold'), 
        dcc.Slider(
            id='threshold', 
            min=0, 
            max=1, 
            step=0.05, 
            value=.1, 
            marks={n: str(n) for n in range(0,2,1)}, 
            className='mb-5', 
        ),   
    ],
)

column3 = dbc.Col(
    [
    #  dcc.Markdown(
    #         """
        
    #         ### (Stand-In Title / Stand-In Image...)

    #         text optional


    #         """
            
    #     ),
          
    ],
)



layout = html.Div(
    [
            dbc.Row([column1, column2]
            ),
            dbc.Row([column3]
            ),
            
    ],
)
