from dash.dependencies import Input, Output, State
from dash import html
from voice_utils.utils import (
    byte_string_to_array,
    encode_to_byte_string,
    get_spectrograms,
    melspectrogram2wav,
    preprocess_to_torch,
    normalize,
    denormalize,
    convert_from_torch,
    SAMPLE_RATE
)
import torch

def register_conversion_callback(app, model):
        @app.callback(Output('conversion-data', 'children'),
                State('upload-data1', 'contents'),
                State('upload-data2', 'contents'),
                Input('convert-button', 'n_clicks'))
        def convert(voice1, voice2, n_click):
            if voice1 is not None and voice2 is not None:
                source, _ = byte_string_to_array(voice1)
                target, _ = byte_string_to_array(voice2)
                #
                source_mel = get_spectrograms(source)
                target_mel = get_spectrograms(target)

                source_mel, target_mel, mean, std = normalize(source_mel, target_mel)
                source_mel, target_mel = preprocess_to_torch(source_mel, target_mel)
                # Conversion by model
                with torch.no_grad():
                     source_target = model.inference(source_mel, target_mel)
                #
                source_target = convert_from_torch(source_target)
                source_target = denormalize(source_target, mean, std)

                source_target_wav = melspectrogram2wav(source_target)
                source_target_byte = encode_to_byte_string(source_target_wav, SAMPLE_RATE)

                children = [
                    html.Audio(src = source_target_byte, controls=True)
                ]
                return children