import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import dash_html_components as html

from app import app

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
https://dash-bootstrap-components.opensource.faculty.ai/l/components/layout

Layout in Bootstrap is controlled using the grid system. The Bootstrap grid has 
twelve columns.

There are three main layout components in dash-bootstrap-components: Container, 
Row, and Col.

The layout of your app should be built as a series of rows of columns.

We set md=4 indicating that on a 'medium' sized or larger screen each column 
should take up a third of the width. Since we don't specify behaviour on 
smaller size screens Bootstrap will allow the rows to wrap so as not to squash 
the content.
"""

column1 = dbc.Col(
    [
        dcc.Markdown(
            """
        
                ## Ebola TITLE Ebola
                
                sdgsda
                sdg
                sadg wadg


            """
        ),
        dcc.Link(dbc.Button('Call To Action', color='primary'), href='/predictions')
    ],
    md=4,
)


# Example:
# gapminder = px.data.gapminder()
# fig = px.scatter(gapminder.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",
#            hover_name="country", log_x=True, size_max=60)

# read csv
df = pd.read_csv('https://data.humdata.org/dataset/9e3665cb-e4a3-4749-9622-f5c8146542c3/resource/c99f90a6-8a3b-4bd0-a081-3bed5d70781f/download/redcross_westafrica_rapidassessment_survey.csv')

# 3-way split: 60-20-20
train1, test = train_test_split(df, train_size=0.80, test_size=0.20, 
                               random_state=42) #stratify=df['ebola'],

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
    
    # return the wrangled dataframe
    return X

df = wrangle(df)

fig = px.scatter_mapbox(df, 
                        lat='hh_gps.latitude', lon='hh_gps.longitude', 
                        color='commebola', opacity=.9)

fig.update_layout(mapbox_style='stamen-terrain')



column2 = dbc.Col(
    [
        dcc.Graph(figure=fig),
    ]
)



layout = dbc.Row([column1, column2])