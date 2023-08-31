from dash.dependencies import Input, Output
from dash import html

def register_upload_callbacks(app):
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