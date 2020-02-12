# Shapley Plots and Feature Importances

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# import category_encoders as ce
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import make_pipeline
# import seaborn as sns
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
# import shap
# import plotly.express as px

from app import app

# # read csv
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
#        'constituency', 'gsm_service', 'birhome', 'instanceId']
#     X = X[columns]
    
#     # return the wrangled dataframe
#     return X

# #Wrangle each split
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

# # Save the instanceId for later to compare predicted to actual
# train_id = X_train['instanceId']
# val_id = X_val['instanceId']
# test_id = X_test['instanceId']

# X_train = X_train.drop(columns='instanceId')
# X_val = X_val.drop(columns='instanceId')
# X_test = X_test.drop(columns='instanceId')



# # Model: XGBClassifier in broken pipeline format
# processor = make_pipeline(               # Needs to be a numeric df to run into Shapley Value Force plot below
#     ce.OrdinalEncoder(), 
#     SimpleImputer(strategy='median')
# )

# X_train_processed = processor.fit_transform(X_train)
# X_val_processed = processor.transform(X_val)

# eval_set = [(X_train_processed, y_train), 
#             (X_val_processed, y_val)]

# model = XGBClassifier(n_estimators=1000, n_jobs=-1)
# model.fit(X_train_processed, y_train, eval_set=eval_set, eval_metric='auc', 
#           early_stopping_rounds=10)


# # process X_test to inspect predicted probabilities
# X_test_processed = processor.transform(X_test)
# y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

# # grabs shapley values from the model above
# explainer = shap.TreeExplainer(model)
 
# @app.callback(
#     Output('shapley_plot', 'children'),
#     [Input('community', 'value'), 
#     #  Input('bathrooms', 'value'), 
#     #  Input('longitude', 'value'), 
#     #  Input('latitude', 'value')
#     ]
# )


# # Mega-Function --- yeilds shapley value plot and top 3 reasons for model decision....
# def explain(community):
#     positive_class = 'commebola'
#     positive_class_index = 1
    
#     # Get & process the data for the row
#     row = X_test.iloc[[community]]
#     row_processed = processor.transform(row)
    
#     # Make predictions (includes predicted probability)
#     pred = model.predict(row_processed)[0]
#     pred_proba = model.predict_proba(row_processed)[0, positive_class_index]
#     pred_proba *= 100
#     if pred != positive_class:
#         pred_proba = 100 - pred_proba
    
#     # Show prediction & probability
#     print(f'The model predicts this community\'s infection is {pred}, with {pred_proba:.0f}% probability.')
    
#     # Get shapley additive explanations
#     shap_values = explainer.shap_values(row_processed)
    
#     # Get top 3 "pros & cons"
#     feature_names = row.columns
#     feature_values = row.values[0]
#     shaps = pd.Series(shap_values[0], zip(feature_names, feature_values))
#     pros = shaps.sort_values(ascending=False)[:3].index
#     cons = shaps.sort_values(ascending=True)[:3].index
    
#     # Show top 3 reasons for prediction
#     print('\n')
#     print('Top 3 reasons for prediction:')
#     evidence = pros if pred == positive_class else cons
#     for i, info in enumerate(evidence):
#         feature_name, feature_value = info
#         print(f'{i+1}. {feature_name} is {feature_value}.')
    
#     # Show top 1 counter-argument against prediction
#     print('\n')
#     print('Top counter-argument against prediction:')
#     evidence = cons if pred == positive_class else pros
#     feature_name, feature_value = evidence[0]
#     print(f'- {feature_name} is {feature_value}.')
    
#     # Show Shapley Values Force Plot
#     shap.initjs()
#     return shap.force_plot(
#         base_value=explainer.expected_value,
#         shap_values=shap_values, 
#         features=row
#     )

#fig = explain(955)
    
# explain(955) # 955 is a sample here representing our worst prediction




column1 = dbc.Col(
    [
        dcc.Markdown(
            ''' 
            ### Predictive Factors of Infection

            After running a cleaned the survey data through a selected model 
            (XGBClassier), the relative predictive power of each variable was assessed and 
            ranked according to importance. Of all the information collected in the survey,
            79 factors were found to help our model accuractely predict infection. 

            Proximity was, not too surprisingly, among the top contributing factors for 
            predicting Ebola infection. 

            Perhaps the most interesting finding is the strong relative importance of other 
            disasters faced by the community during the time of the outbreak. Drought, food shortages,
            flood, conflict, and other calamities all greatly increased a community's likelihood of
            having Ebola infection. 

            Other major contributing factors include: the time it takes one to get treatment when needed, 
            if representatives from health
            network organizations were active in the community, whether the community had access 
            to a women's health professional, if fruit was sold at the local market, access to 
            a dedicated facility for child birthing, the proximity to a boarder crossing. 


            '''
        ), 
    ]
)

column2 = dbc.Col(
    [
        dcc.Markdown('#### Permutation Importance: top 25 features'
        ),
        html.Img(src='assets/tempsnip.jpg', className='img-fluid'
        ),       
    ]
)

column3 = dbc.Col(
    [
        dcc.Markdown('## Leading Factors in Sample Communities'
        ),
        dcc.Dropdown(
            id='community', 
            options = [
                {'label': 'Africa', 'value': 955}, 
                # {'label': 'Americas', 'value': 'Americas'}, 
                # {'label': 'Asia', 'value': 'Asia'}, 
                # {'label': 'Europe', 'value': 'Europe'}, 
                # {'label': 'Oceania', 'value': 'Oceania'}, 
            ], 
            value = 166, 
            className='mb-5',
        ),
        dcc.Markdown('#### Shapley Values for Model\'s Worst Prediction'
        ),
        html.Img(src='assets/shapley_low.png', className='img-fluid'
        ),  
        dcc.Markdown(
            ''' 
            For the village of Peivalor, our model predicted only a 2.2%
            probability of the community having any cases of Ebola. However, there was
            in fact at least one unconfirmed case of infection. 
            
            Inspecting the plot above, the top two factors decreasing likelihood of infection: (1) this is a 
            relatively small community of 18 households; (2) the local market does not sell
            fresh fruit. 

            On the lefthand side in red, the top factor that lead our model to predict an increased 
            infect risk was that the community had no person or facility designated for giving birth. 
            The next two biggest factors were interrelated. The member of the community surveyed specified that 
            there were other disasters experienced by the community during the outbreak, but the nature of the 
            coinciding diaster was either not disclosed or was not included in the report. 

            '''
        ), 
                dcc.Markdown('#### Shapley Values for Model\'s Most Certain Prediction'
        ),
        html.Img(src='assets/shapley_high.png', className='img-fluid'
        ),  
        dcc.Markdown(
            ''' 
            The village of Oremai was given the highest probability of infection at 97%. All 
            told, this village of 100 households experience 4 cases of Ebola. 

            Most notably from the diagram, there simply does not appear to be very many 
            factors in this village's favor. The largest factor contributing to our model
            predicting an infection in this community was that it experience two major 
            disasters during the same time as the outbreak. The next largest factor was
            the amount of time the it takes for the average community memeber to get to and
            return from a treatment facility. 

            '''
        ), 
        
        # dcc.Graph(figure=fig
        # ),
        # html.Div(id='shapley_values', className='lead' 
        # ),
    ],
    md=9,
)



layout = html.Div(
    [
            dbc.Row([column1, column2]
            ),
            dbc.Row([column3]
            ),
            
    ],
)