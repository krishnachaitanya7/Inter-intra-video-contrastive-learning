import plotly.express as px
import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
app = dash.Dash()
app.layout = html.Div([
    html.H1(id='live-counter'),
    dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("GradCam for Videos"),
                    html.A("(Source code)",
                           href="https://github.com/krishnachaitanya7/Inter-intra-video-contrastive-learning/blob/devel/gradcam_sc1.py")
                ], width=True)])
            ]),
    dcc.Interval(
        id='1-second-interval',
        interval=1000,
        n_intervals=0
    )
])
@app.callback(Output('live-counter', 'children'),
              [Input('1-second-interval', 'n_intervals')])
def update_layout(n):
    return 'This app has updated itself for {} times, every second.'.format(n)


if __name__ == '__main__':
    app.run_server()