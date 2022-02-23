
import dash
import pandas as pd
from dash import dcc
from dash import html
from dash import Input, Output
import dash_bootstrap_components as dbc
from generate_datasets import make_point_clouds
import plotly.express as px
from gtda.diagrams import PersistenceLandscape
from gtda.diagrams import BettiCurve

app = dash.Dash(__name__,external_stylesheets=[dbc.themes.DARKLY])


X, Y = make_point_clouds(n_samples_per_shape=1, n_points=20, noise=0.2)

TDAdata = {'Manifold': ['S_1','T','S_2'],'Nuage de Point': [X[0],X[1],X[2]]}
df = pd.DataFrame(data=TDAdata)
fig1= px.scatter_3d(x=X[0][:,0],y=X[0][:,1],z=X[0][:,2])


PL = PersistenceLandscape()
Bc = BettiCurve()
from gtda.homology import VietorisRipsPersistence
VR = VietorisRipsPersistence(homology_dimensions=[0,1,2])
diags = VR.fit_transform(X)
fig2= VR.plot(diags, sample=1)

app.layout = html.Div(children = [
    html.H1(children='TDA Viz'),

    html.Div(children='''
        TDA Visualization with Dash.
    '''),
    html.Div(children=[
        html.Label('Manifold'),
        dcc.Dropdown(
            options=[
                {'label': 'Cercle', 'value': 'S_1'},
                {'label': u'Sphère', 'value': 'T'},
                {'label': 'Tore', 'value': 'S_2'}
            ],
            value='S_1',
            id ='Input_Manifold'
        ),
        dcc.Graph(
            id='Représentation 3d',
            figure=fig1,
            style={'width': '90vh', 'height': '90vh'}
            ),
        ],style={'padding': 10, 'flex': 1}
        ),
        html.Div(children=[
        html.Label('TDA'),
        dcc.Dropdown(
            options=[
                {'label': 'Persistance Diagram', 'value': 'PersD'},
                {'label': 'Persistance Landscape', 'value': 'PersL'},
                {'label': 'Betti curve', 'value': 'BeC'}
            ],
            value='PersD',
            id='Dropdown-TDA'
        ),
        dcc.Graph(
            id='Analyse Topologique',
            figure=fig2,
            style={'width': '90vh', 'height': '90vh'}
        )
        ],style={'padding': 10, 'flex': 1})
    ],style={'display': 'flex', 'flex-direction': 'row'}
)


@app.callback(
    Output('Représentation 3d', 'figure'),
    Input('Input_Manifold', 'value')
    )
def update_figure1(selected_mf):
    filtered_df = df[df.Manifold == selected_mf]
    filtered_df = filtered_df.to_numpy()
    fig1 = px.scatter_3d(x=filtered_df[0][1][:,0],y=filtered_df[0][1][:,1],z=filtered_df[0][1][:,2])

    fig1.update_layout(transition_duration=500)

    return fig1

@app.callback(
    Output('Analyse Topologique', 'figure'),
    Input('Dropdown-TDA', 'value'),
    Input('Input_Manifold', 'value')
    )
def update_figure2(selected_TDA,selected_mf):
    filtered_df = df[df.Manifold == selected_mf]
    filtered_df = filtered_df.to_numpy()
    diags = VR.fit_transform(filtered_df[0][1][None, :, :])
    if selected_TDA == 'PersD':
        fig2= VR.plot(diags)

    if selected_TDA == 'PersL':
        fig2 = PL.fit_transform_plot(diags)

    if selected_TDA == 'BeC':
        fig2 = Bc.fit_transform_plot(diags)
    
    fig2.update_layout(transition_duration=500)
    return fig2




if __name__ == '__main__':
    app.run_server(debug=True)