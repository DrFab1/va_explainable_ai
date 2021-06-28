import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import base64
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io

# TODO: 1 weiteres Beispieldatensets fertig pre-processed als .csv
# TODO: filter über plots setzt auch globalen filter für andere Daten-vis plots
# TODO: mehr Shap plots
# TODO: mehr Erklärungen hinzufügen
# TODO: mehr Daten-vis Plots
# TODO: fix abgeschnittenheit vom shap plot

# TODO: "Information seeking mantra" umsetzen
"""
-    The most important rule for visualization of data is the “information seeking mantra” by Ben Shneiderman:
o    1. overview first (Übersicht zuerst)
o    2. zoom
o    3. filter 
o    4. details on demand (Details auf Abruf)
o    5. relate: show relationships between data items 
o    6. history: allow undo/redo 
o    7. extract: allow extraction of data and query parameters
"""

# -----------------------------------------------------------------------------------
# Default exemplary dataset

df = pd.read_csv('ready_to_use_data/boston.csv')

# -----------------------------------------------------------------------------------
"""
    Dashboard layout
"""

app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1(children='A Dashboard for showing Explainable AI for Regression Tasks'),
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
                id="datatable",
                data=df.to_dict("records"),
                columns=[{'id': c, 'name': c} for c in df.columns],
                page_action='none',
                style_table={'height': '300px', 'overflowY': 'auto'}
                ),
    html.H2(children='Dataset Visualization'),
    html.Label(["Select Features to use for regression", dcc.Dropdown(
        id="dropdown_features",
        options=[{"label": x, "value": x} 
                 for x in df.columns.tolist()],
        value=df.columns.tolist()[:6],
        multi=True
    )]),
    html.Label(["Select Target for regression", dcc.Dropdown(
        id="dropdown_targets",
        options=[{"label": x, "value": x}
                 for x in df.columns.tolist()],
        value=df.columns.tolist()[-1],
        multi=False
    )]),
    dcc.Graph(id="splom"),
    dcc.Graph(id="parcoord"),
    html.H2(children='Model Visualization'),
    html.Img(id='waterfall_shap')
])

# -----------------------------------------------------------------------------------
"""
    Dashboard Components/Update Functions
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
    shap.plots.waterfall(shap_values[sample_ind], show=False, max_display=20)
    fig = plt.gcf()
    #fig.set_figheight(6)
    #fig.set_figwidth(10)
    plt.xlabel('xlabel', fontsize=8)
    plt.ylabel('ylabel', fontsize=8)
    plt.savefig('shap_waterfall.png')
    plt.close()

    image_path = "shap_waterfall.png"
    encoded_image = base64.b64encode(open(image_path, 'rb').read())

    return 'data:image/png;base64,{}'.format(encoded_image.decode())


@app.callback(Output('datatable', 'columns'),
              Output('datatable', 'data'),
              Output('dropdown_features', 'options'),
              Output('dropdown_features', 'value'),
              Output('dropdown_targets', 'options'),
              Output('dropdown_targets', 'value'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    global df

    def parse_contents(contents, filename, date):
        content_type, content_string = contents.split(',')

        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                return df

        except Exception as e:
            print(e)
            return None

    if list_of_names is not None:
        df = parse_contents(list_of_contents, list_of_names, list_of_dates)
        try:
            df = df.sample(1000)
        except ValueError:
            pass
        
    data = df.to_dict("records")
    columns = [{'id': c, 'name': c} for c in df.columns]
    options_f = [{"label": x, "value": x} for x in df.columns.tolist()]
    value_f = df.columns.tolist()[:6]
    options_t = [{"label": x, "value": x} for x in df.columns.tolist()]
    value_t = df.columns.tolist()[-1]

    return columns, data, options_f, value_f, options_t, value_t


app.run_server(debug=True)
