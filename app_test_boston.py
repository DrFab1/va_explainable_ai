

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import shap
import base64
from io import BytesIO
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

# -----------------------------------------------------------------------------------
"""
    Later replace this part with loading data from csv.
    (which has to be fully prepared beforehand.)
"""
# data preperation for boston data set
target = load_boston().target
# re to real
target = np.multiply(target, 1000)
target = target.astype(int)

df = pd.DataFrame(load_boston().data, columns=load_boston().feature_names)

df['new_col'] = target
df = df.rename(columns={"new_col": "Price"})

all_dims = df.columns.tolist()

# -----------------------------------------------------------------------------------

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Label(["Select Features to use for regression", dcc.Dropdown(
        id="dropdown_features",
        options=[{"label": x, "value": x} 
                 for x in all_dims],
        value=all_dims[:-1],
        multi=True
    )]),
    html.Label(["Select Target for regression", dcc.Dropdown(
        id="dropdown_targets",
        options=[{"label": x, "value": x} 
                 for x in all_dims],
        value=all_dims[-1],
        multi=False
    )]),
    dcc.Graph(id="splom"),
    dcc.Graph(id="parcoord"),
    html.Img(id='waterfall_shap')
])


@app.callback(
    Output("splom", "figure"), 
    [Input("dropdown_features", "value"),
     Input("dropdown_targets", "value")])
def update_scatter_chart(dims, label):
    fig = px.scatter_matrix(df, dimensions=dims, color=label,
                            color_continuous_scale=px.colors.sequential.Viridis) # make colordynamic dependent on parcoord
    return fig


@app.callback(
    Output("parcoord", "figure"),
    [Input("dropdown_features", "value"),
     Input("dropdown_targets", "value")])
def update_paar_coord_chart(dims, label):
    fig = px.parallel_coordinates(df, color=label, dimensions=dims,
                                  color_continuous_scale=px.colors.sequential.Viridis) # make colordynamic dependent on scatter
    return fig

@app.callback(
    Output("waterfall_shap", "src"),
    [Input("dropdown_features", "value"),
     Input("dropdown_targets", "value")])
def update_waterfall_shap_chart(dims, label):
    X = df[dims]
    y = df[label]
    # train-test-split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Build and train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # make prediction
    y_pred = model.predict(X_test)

    # compute the SHAP values for the linear model
    explainer = shap.Explainer(model.predict, X_test)
    shap_values = explainer(X_train)

    sample_ind = 18  # what is this lul´´

    # plot results
    #shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    shap.plots.waterfall(shap_values[sample_ind], show=False)
    plt.savefig('grafic.png')

    #from PIL import Image
    #img = Image.open("grafic.png")

    image_path = "grafic.png"

    encoded_image = base64.b64encode(open(image_path, 'rb').read())

    return 'data:image/png;base64,{}'.format(encoded_image.decode())




app.run_server(debug=True)
