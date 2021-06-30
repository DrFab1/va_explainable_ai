import dash
from dash_bootstrap_components._components.CardBody import CardBody
from dash_bootstrap_components._components.Row import Row
from dash_html_components.Span import Span
import dash_table
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
import base64
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
import io
from sklearn.decomposition import PCA

# TODO: mehr Shap plots
# TODO: mehr Erklärungen hinzufügen
# TODO: fix abgeschnittenheit vom shap plot

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
matplotlib.use('Agg')
df = pd.read_csv('ready_to_use_data/boston.csv')

# -----------------------------------------------------------------------------------
"""
    Dashboard layout
"""

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])

app.layout = html.Div([

    dbc.Card(
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.H1(children='A Dashboard for showing Explainable AI for Regression Tasks', 
                    style={
                        'textAlign': 'center',
                        'border-border-bottom':'5px solid black'})  
                ], width=12)
            ], align='center'),

            html.Br(),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "About this Dashboard",
                        id="collapse-button1",
                        className="mb-3",
                        color="primary",
                        n_clicks=0,
                    ),
                    dbc.Collapse(
                        dbc.Card(dbc.CardBody("Insert Info here")),
                        id="collapse1",
                        is_open=False,
                    ),
                ], width=4,
                )
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
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
                        multiple=False
                    )
                ], width=12)
            ]),
              
            html.Br(),

            dbc.Row([
                dbc.Col([
                     dbc.Card(
                        dbc.CardBody(
                                dash_table.DataTable(
                                id="datatable",
                                data=df.to_dict("records"),
                                columns=[{'id': c, 'name': c} for c in df.columns],
                                page_action='none',
                                fixed_rows={'headers': True},
                                style_table={
                                    'height': '300px', 
                                    'overflowY': 'auto',
                                    'backgroundColor' : 'rgba(0, 0, 0, 0)'
                                    },
                                style_cell={
                                    'backgroundColor' : 'rgba(255,0,0,0.2)',
                                    'color' : 'white'
                                },
                                style_header={
                                    'backgroundColor' : 'rgba(0, 0, 0, 0.5)',
                                    'fontWeight': 'bold',
                                    'textAlign' : 'center'
                                },
                            )
                        )
                    ) 
                ])
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.Span(
                        "?",
                        id="tooltip-target1",
                        style={
                           "textDecoration": "underline", 
                           "cursor": "pointer" 
                        }
                    ),
                    dbc.Tooltip(
                        "Info for plot",
                        target="tooltip-target1",
                    )
                ])
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.H1(children='Dataset Visualization', 
                    style={
                        'textAlign': 'left',
                        'border-border-bottom':'5px solid black'})  
                ], width=12)
            ], align='left'),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "About Viz",
                        id="collapse-button2",
                        className="mb-3",
                        color="primary",
                        n_clicks=0,
                    ),
                    dbc.Collapse(
                        dbc.Card(dbc.CardBody("Insert Info here")),
                        id="collapse2",
                        is_open=False,
                    ),
                ], width=4,
                )
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                     dbc.Card(
                        dbc.CardBody(
                            html.Label(["Select Features to use for regression", dcc.Dropdown(
                                id="dropdown_features",
                                options=[{"label": x, "value": x} 
                                        for x in df.columns.tolist()],
                                value=df.columns.tolist()[:6],
                                multi=True
                            )])
                        )
                    ) 
                ]),
                dbc.Col([
                     dbc.Card(
                        dbc.CardBody(
                            html.Label(["Select Target for regression", dcc.Dropdown(
                                id="dropdown_targets",
                                options=[{"label": x, "value": x}
                                        for x in df.columns.tolist()],
                                value=df.columns.tolist()[-1],
                                multi=False
                            )])
                        )
                    ) 
                ])
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                     dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id="parcoord")
                        )
                    ) 
                ])
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.Span(
                        "?",
                        id="tooltip-target2",
                        style={
                           "textDecoration": "underline", 
                           "cursor": "pointer" 
                        }
                    ),
                    dbc.Tooltip(
                        "Info for ParaCoord",
                        target="tooltip-target2",
                    )
                ])
            ]),            

            html.Br(),

            dbc.Row([
                dbc.Col([
                     dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id="violin")
                        )
                    ) 
                ]),
                dbc.Col([
                     dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id="reduction-graph")
                        )
                    ) 
                ])               
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.Span(
                        "?",
                        id="tooltip-target3",
                        style={
                           "textDecoration": "underline", 
                           "cursor": "pointer" 
                        }
                    ),
                    dbc.Tooltip(
                        "Info for Violin Plot",
                        target="tooltip-target3",
                    )
                ]),
                dbc.Col([
                    html.Span(
                        "?",
                        id="tooltip-target4",
                        style={
                           "textDecoration": "underline", 
                           "cursor": "pointer" 
                        }
                    ),
                    dbc.Tooltip(
                        "Info for Reduction Plot",
                        target="tooltip-target4",
                    )
                ])
            ]),            

            html.Br(),

            dbc.Row([
                dbc.Col([
                     dbc.Card(
                        dbc.CardBody(
                            dcc.Graph(id="splom")
                        )
                    ) 
                ])          
            ]),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.Span(
                        "?",
                        id="tooltip-target5",
                        style={
                           "textDecoration": "underline", 
                           "cursor": "pointer" 
                        }
                    ),
                    dbc.Tooltip(
                        "Info for Scatter Matrix plot",
                        target="tooltip-target5",
                    )
                ])
            ]),             

            html.Br(),

            dbc.Row([
                dbc.Col([
                    html.H1(children='Model Visualization', 
                    style={
                        'textAlign': 'left',
                        'border-border-bottom':'5px solid black'})  
                ], width=12)
            ], align='left'),

            html.Br(),

            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "About Viz",
                        id="collapse-button3",
                        className="mb-3",
                        color="primary",
                        n_clicks=0,
                    ),
                    dbc.Collapse(
                        dbc.Card(dbc.CardBody("Insert Info here")),
                        id="collapse3",
                        is_open=False,
                    ),
                ], width=4,
                )
            ]),            

            html.Br(),

            dbc.Row([
                dbc.Col([
                     dbc.Card(
                        dbc.CardBody(
                            html.Img(id='waterfall_shap')
                        )
                    ) 
                ]),
                dbc.Col([
                     dbc.Card(
                        dbc.CardBody(
                            html.Img(id='beeswarm')
                        )
                    ) 
                ])               
            ]),

        ]), color='dark'
    )     
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
    fig = px.scatter_matrix(df, dimensions=dims+[label], color=label,
                            color_continuous_scale=px.colors.sequential.Bluered, height=800).update_layout(
                                template='plotly_dark',
                                plot_bgcolor= 'rgba(0, 0, 0, 0.5)',
                                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                )#make colordynamic dependent on parcoord
    return fig


@app.callback(
    Output("parcoord", "figure"),
    [Input("dropdown_features", "value"),
     Input("dropdown_targets", "value")])
def update_paar_coord_chart(dims, label):
    fig = px.parallel_coordinates(df, color=label, dimensions=dims+[label],
                                  color_continuous_scale=px.colors.sequential.Bluered).update_layout(
                                template='plotly_dark',
                                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                ) # make colordynamic dependent on scatter
    return fig

@app.callback(
    Output("waterfall_shap", "src"),
    Output("beeswarm","src"),
    [Input("dropdown_features", "value"),
     Input("dropdown_targets", "value")])
def update_shap_charts(dims, label):

    if label in dims:
        dims.remove(label)

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
    fig = plt.gcf()
    fig.set_figheight(3)
    fig.set_figwidth(5)
    #plt.xlabel('xlabel', fontsize=8)
    #plt.ylabel('ylabel', fontsize=8)
    plt.savefig('shap_waterfall.png')
    plt.close()


    shap.plots.beeswarm(shap_values, max_display=14)
    fig = plt.gcf()
    fig.set_figheight(3)
    fig.set_figwidth(5)
    #plt.xlabel('xlabel', fontsize=8)
    #plt.ylabel('ylabel', fontsize=8)
    plt.savefig('shap_beeswarm.png')
    plt.close()

    image_path_wf = "shap_waterfall.png"
    encoded_image_wf = base64.b64encode(open(image_path_wf, 'rb').read())
    image_path_bs = "shap_beeswarm.png"
    encoded_image_bs = base64.b64encode(open(image_path_bs, 'rb').read())

    return 'data:image/png;base64,{}'.format(encoded_image_wf.decode()), 'data:image/png;base64,{}'.format(encoded_image_bs.decode())


@app.callback(Output('datatable', 'columns'),
              Output('datatable', 'data'),
              Output('dropdown_features', 'options'),
              Output('dropdown_features', 'value'),
              Output('dropdown_targets', 'options'),
              Output('dropdown_targets', 'value'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_on_drag_and_drop(list_of_contents, list_of_names, list_of_dates):
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


@app.callback(
    Output("violin", "figure"),
    [Input("dropdown_features", "value"),
     Input("dropdown_targets", "value")])
def update_violin(dims, label):
    fig = px.violin(df, y=label, box=True, points='all').update_layout(
                                template='plotly_dark',
                                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                )
    return fig


@app.callback(
    Output("reduction-graph", "figure"), 
    [Input("dropdown_features", "value"),
     Input("dropdown_targets", "value")])
def update_reduction_chart(feat, label):

    if label in feat:
        feat.remove(label)

    X = df[feat].values
    target = df[label].values

    X = PCA(n_components=3).fit_transform(X)
    fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2],
                        color=target, color_continuous_scale=px.colors.sequential.Bluered).update_layout(
                                template='plotly_dark',
                                plot_bgcolor= 'rgba(0, 0, 0, 0)',
                                paper_bgcolor= 'rgba(0, 0, 0, 0)',
                                )
    return fig

@app.callback(
    Output("collapse1", "is_open"),
    [Input("collapse-button1", "n_clicks")],
    [State("collapse1", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse2", "is_open"),
    [Input("collapse-button2", "n_clicks")],
    [State("collapse2", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse3", "is_open"),
    [Input("collapse-button3", "n_clicks")],
    [State("collapse3", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)