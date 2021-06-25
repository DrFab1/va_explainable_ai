import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
from sklearn.datasets import load_boston
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import base64
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import io
# -----------------------------------------------------------------------------------
"""
    Pre-processing
    
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
"""
    Dashboard layout
"""

app = dash.Dash(__name__)

app.layout = html.Div([

    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),

    dash_table.DataTable(
                id = "datatable",
                data=df.to_dict("records"),
                columns=[{'id': c, 'name': c} for c in df.columns],
                page_action='none',
                style_table={'height': '300px', 'overflowY': 'auto'}
                ),
    html.H1(children='XAI for Regression'),
    html.H2(children='Dataset Visualization'),
    # TODO: plot table
    html.Label(["Select Features to use for regression", dcc.Dropdown(
        id="dropdown_features",
        options=[{"label": x, "value": x} 
                 for x in all_dims],
        value=all_dims[:6],
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
    html.H2(children='Model Visualization'),
    html.Img(id='waterfall_shap')
])

# -----------------------------------------------------------------------------------
"""
    Dashboard Components
"""


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
    fig = px.parallel_coordinates(df, color=label, dimensions=dims+[label],
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
    # shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
    shap.plots.waterfall(shap_values[sample_ind], show=False)
    plt.savefig('grafic.png')
    plt.close()

    image_path = "grafic.png"
    encoded_image = base64.b64encode(open(image_path, 'rb').read())

    return 'data:image/png;base64,{}'.format(encoded_image.decode())



def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))

    except Exception as e:
        print(e)
        return None
    
    return df
# -----------------------------------------------------------------------------------
@app.callback(Output('datatable', 'columns'),
              Output('datatable', 'data'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):

        if list_of_names is not None:
            
            df = parse_contents(list_of_contents, list_of_names, list_of_dates)
            data = df.to_dict("records")
            columns = [{'id': c, 'name': c} for c in df.columns]
            

            return columns, data

app.run_server(debug=True)