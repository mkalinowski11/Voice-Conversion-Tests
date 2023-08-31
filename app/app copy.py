import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Voice conversion app", style={'textAlign': 'center', 'margin-top': '20px'}),
    html.Div([
        html.Div([
            dcc.Upload(
                id='upload-data1',
                children=html.Div([
                    'Source voice sample'
                ]),
                style={
                    'width': '100%',
                    'height': '100px',
                    'lineHeight': '100px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'backgroundColor': '#F1F1F1',
                    'color': '#333333',
                },
                multiple=False
            ),
            html.Div(id='output-data-upload1', style={'textAlign': 'center'}),
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'middle', 'textAlign': 'center'}),
        html.Div([
            dcc.Upload(
                id='upload-data2',
                children=html.Div([
                    'Target voice sample'
                ]),
                style={
                    'width': '100%',
                    'height': '100px',
                    'lineHeight': '100px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'backgroundColor': '#F1F1F1',
                    'color': '#333333',
                },
                multiple=False
            ),
            html.Div(id='output-data-upload2', style={'textAlign': 'center'}),
            dbc.Button("Large button", size="100px", className="me-1")
            # html.Div([html.Button('Perform voice conversion', id='conversion-button')], style={'verticalAlign': 'center'}),
        ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'middle', 'textAlign': 'center'}),
    ], style={'max-width': '800px', 'margin': '0 auto', 'text-align': 'center', 'height': '100vh', 'display': 'flex', 'justify-content': 'center'}),
])

@app.callback(Output('output-data-upload1', 'children'),
              Input('upload-data1', 'contents'))
def update_output1(content1):
    if content1 is not None:
        children = [
            html.Audio(src=content1, controls=True)
        ]
        return children

@app.callback(Output('output-data-upload2', 'children'),
              Input('upload-data2', 'contents'))
def update_output2(content2):
    if content2 is not None:
        children = [
            html.Audio(src=content2, controls=True)
        ]
        return children

if __name__ == '__main__':
    app.run_server(debug=True)