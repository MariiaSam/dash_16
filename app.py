import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

cnn_model = tf.keras.models.load_model('model_name.h5')
vgg_model = tf.keras.models.load_model('model_vgg16.h5')

def predict_image(model, image):
    image_resized = cv2.resize(image, (28, 28)) if model.input_shape[1] == 28 else cv2.resize(image, (224, 224))
    image_scaled = image_resized / 255.0
    image_expanded = np.expand_dims(image_scaled, axis=0)
    predictions = model.predict(image_expanded)
    return predictions

def plot_metrics(history):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(history['accuracy'], label='Train Accuracy')
    axs[0].plot(history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Accuracy')
    axs[0].legend()
    axs[1].plot(history['loss'], label='Train Loss')
    axs[1].plot(history['val_loss'], label='Validation Loss')
    axs[1].set_title('Loss')
    axs[1].legend()
    plt.tight_layout()
    return fig

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Класифікація зображень за допомогою нейронних мереж", className="text-center mt-3"),
    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='upload-image',
                children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                style={
                    'width': '100%', 'height': '60px', 'lineHeight': '60px',
                    'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                    'textAlign': 'center', 'margin': '10px'
                },
                multiple=False
            ),
            html.Div(id='output-image', className="mt-3"),
            dcc.RadioItems(
                id='model-selector',
                options=[
                    {'label': 'CNN', 'value': 'cnn'},
                    {'label': 'VGG16', 'value': 'vgg'}
                ],
                value='cnn',
                labelStyle={'display': 'block'}
            ),
            dbc.Button("Classify", id='classify-button', color="primary", className="mt-3")
        ], width=4),
        dbc.Col([
            html.Div(id='prediction-output', className="mt-3"),
            html.Div(id='metrics-output', className="mt-3")
        ], width=8),
    ])
], fluid=True)

@app.callback(
    [Output('output-image', 'children'), Output('prediction-output', 'children')],
    [Input('classify-button', 'n_clicks')],
    [State('upload-image', 'contents'), State('model-selector', 'value')]
)
def classify_image(n_clicks, image_content, model_name):
    if n_clicks is None or image_content is None:
        return None, None

    content_type, content_string = image_content.split(',')
    decoded = base64.b64decode(content_string)
    image = np.array(Image.open(io.BytesIO(decoded)).convert('RGB'))

    model = cnn_model if model_name == 'cnn' else vgg_model

    predictions = predict_image(model, image)
    predicted_class = np.argmax(predictions)
    probabilities = predictions[0]

    image_encoded = base64.b64encode(decoded).decode('ascii')
    image_display = html.Img(src='data:image/png;base64,' + image_encoded, style={'width': '100%'})
    results_display = html.Div([
        html.H4(f"Predicted Class: {predicted_class}"),
        html.P(f"Probabilities: {probabilities}")
    ])

    return image_display, results_display

if __name__ == '__main__':
    app.run_server(debug=True)
