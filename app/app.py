import dash
from dash import dcc
from dash import html
from callbacks.file_upload import register_upload_callbacks
from callbacks.conversion import register_conversion_callback
from model.utils import get_model
import os

MODEL_PATH = os.path.join('model', 'model_weights.ckpt')
CONFIG_PATH = os.path.join('model', 'config.json')
DEVICE = 'cpu'

app = dash.Dash(__name__)
model = get_model(MODEL_PATH, CONFIG_PATH, DEVICE)

register_upload_callbacks(app)
register_conversion_callback(app, model)

app.layout = html.Div([
    html.H1("Voice conversion app", style={'textAlign': 'center', 'margin-top': '20px'}),
    dcc.Upload(
        id='upload-data1',
        children=html.Div([
            'Source voice sample'
        ]),
        style={
                    'width': '60%',
                    'height': '100px',
                    'lineHeight': '100px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'backgroundColor': '#F1F1F1',
                    'color': '#333333',
                    'margin-left' : '20%',
                    'margin-right' : '20%'
                },
        multiple=False
    ),
    html.Div(id='output-data-upload1', style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-data2',
        children=html.Div([
            'Target voice sample'
        ]),
        style={
                    'width': '60%',
                    'height': '100px',
                    'lineHeight': '100px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px',
                    'backgroundColor': '#F1F1F1',
                    'color': '#333333',
                    'margin-left' : '20%',
                    'margin-right' : '20%'
                },
        multiple=False
    ),
    html.Div(id='output-data-upload2', style={'textAlign': 'center'}),
    html.Div(html.Button('Generate conversion', id='convert-button',
                         style={
                            'background-color': 'blue',
                            'color': 'white',          
                            'font-size': '18px',       
                            'border': 'none',          
                            'border-radius': '8px',    
                            'padding': '10px 20px',
                            'margin-top' : '10px'
                        }), 
                        style={
                            'display': 'flex',
                            'justify-content': 'center',
                            'align-items': 'center',
                }),
    html.Div(id='conversion-data', style={'textAlign': 'center'}),
])

if __name__ == '__main__':
    app.run_server(debug=True)