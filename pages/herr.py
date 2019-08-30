# # Dash imports
# import dash
# import dash_bootstrap_components as dbc
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output

# # My app imports
# from app import app

# # Other 3rd party library imports
# from joblib import load
# import pandas as pd
# import shap

# model = load('assets/model.joblib')
# print('Model loaded successfully')

# @app.callback(
#     Output('prediction-content', 'children'),
#     [Input('bedrooms', 'value'), 
#      Input('bathrooms', 'value'), 
#      Input('longitude', 'value'), 
#      Input('latitude', 'value')],
# )
# def predict(bedrooms, bathrooms, longitude, latitude):
#     df = pd.DataFrame(
#         data=[[bedrooms, bathrooms, longitude, latitude]],
#         columns=['bedrooms', 'bathrooms', 'longitude', 'latitude']
#     )
#     pred = model.predict(df)[0]
    
#     explainer = shap.TreeExplainer(model)
#     shap_values = explainer.shap_values(df)
    
#     feature_names = df.columns
#     feature_values = df.values[0]
#     shaps = pd.Series(shap_values[0], zip(feature_names, feature_values))
    
#     result = [html.Div(f'Rent is estimated at ${pred:,.0f} for this New York City apartment. \n\n')]
#     result.append(html.Div(f'Starting from a baseline of ${explainer.expected_value:,.0f}. \n'))
#     explanation = shaps.to_string()
#     lines = explanation.split('\n')
#     for line in lines:
#         result.append(html.Div(line))
#     return result


# column1 = dbc.Col(
#     [
#         dcc.Markdown('## Inputs', className='mb-5'), 
#         dcc.Markdown('#### Bedrooms'), 
#         dcc.Slider(
#             id='bedrooms', 
#             min=0, 
#             max=5, 
#             step=1, 
#             value=1, 
#             marks={n: str(n) for n in range(0,6,1)}, 
#             className='mb-5', 
#         ), 
#         dcc.Markdown('#### Bathrooms'), 
#         dcc.Slider(
#             id='bathrooms', 
#             min=0, 
#             max=5, 
#             step=0.5, 
#             value=1, 
#             marks={n: str(n) for n in range(0,6,1)}, 
#             className='mb-5', 
#         ), 
#         dcc.Markdown('#### Longitude'), 
#         dcc.Input(
#             id='longitude', 
#             placeholder=-73.978,
#             type='number',
#             value=-73.978
#         ),  
#         dcc.Markdown('#### Latitude'), 
#         dcc.Input(
#             id='latitude', 
#             placeholder=40.751,
#             type='number',
#             value=40.751
#         ),  
#     ],
#     md=4,
# )

# column2 = dbc.Col(
#     [
#         html.H2('NYC Estimated Rent', className='mb-5'), 
#         html.Div(id='prediction-content', className='lead')
#     ]
# )

# layout = dbc.Row([column1, column2])