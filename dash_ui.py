import plotly.express as px
import dash
import pandas as pd
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import gradcam_sc1
import base64

# I am not a big fan of global variables, but here we go
# I apologize again
# Changing Globals
original_vid = None
gradcam_vid = None
play_video = True
predicted_indices = None
original_indices = None
# Un changing Globals
vid_sleep_time = 900  # In milliseconds
video_frame_index = 0


# End Global Variables


def id_to_label(x: int):
    class_idx_path = "./data/ucf101/split/classInd.txt"
    label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1][x + 1]
    return label


if __name__ == "__main__":
    app = dash.Dash(external_stylesheets=[dbc.themes.UNITED])
    # encoded_image = base64.b64encode(open("colormap_Jet.png", 'rb').read())
    original_vid, gradcam_vid, predicted_indices, original_indices = gradcam_sc1.main()
    # Now the predicted and original indices are for one video. Each video usually contains of n = 16 frames.
    # during plotting the GUI, as we plot for every frame, it will be useful if we have predicted indices first shapes
    # equal, so we are gonna do that
    frames_per_video = int(original_vid.shape[0] / len(original_indices))
    predicted_indices = [index for index in predicted_indices for _ in range(frames_per_video)]
    original_indices = [index for index in original_indices for _ in range(frames_per_video)]
    dropdown_menu_items = [
        {'label': k, 'value': k} for k in open("data/ucf101/split/all_test.txt", "r").readlines()
    ]
    app.layout = html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H2("GradCam for Videos"),
                    html.A("(Source code)",
                           href="https://github.com/krishnachaitanya7/Inter-intra-video-contrastive-learning/blob/devel/gradcam_sc1.py")
                ], width=True),
            ], align="end"),
            html.Div(id="hidden-div", style={"display": "none"}),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.H2("Select Video to be Analyzed:"),
                    dcc.Dropdown(
                        id='select_video',
                        options=dropdown_menu_items,
                        value=dropdown_menu_items[0]['value']
                    ),
                ], width=True),
            ], align="end"),
            html.Hr(),
            dcc.Interval('interval_component', interval=vid_sleep_time, n_intervals=0),
            html.Div(id="tab-content", className="p-4")

        ],
            fluid=True)
    ])


    @app.callback(
        [Output('tab-content', 'children')],
        [Input('interval_component', 'n_intervals')])
    def update_figure(n):
        global original_vid
        global gradcam_vid
        global original_indices
        global predicted_indices
        global play_video
        global video_frame_index
        if play_video:
            original_vid_frame = original_vid[video_frame_index]
            gradcam_vid_frame = gradcam_vid[video_frame_index]
            predicted_index = id_to_label(predicted_indices[video_frame_index])
            original_index = id_to_label(original_indices[video_frame_index])
            fig1 = px.imshow(original_vid_frame)
            fig2 = px.imshow(gradcam_vid_frame)
            video_frame_index += 1
            print(f"Play Video True with {video_frame_index}")
            if video_frame_index == original_vid.shape[0]:
                # The video is played, stop the video playing
                play_video = False
                print("Play Video False")
                return dash.no_update
            return dbc.Row([
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(
                            id="original_video",
                            figure=fig1
                        ),
                    ], width=True),
                    dbc.Col([
                        dcc.Graph(
                            id="gradcam_video",
                            figure=fig2),
                        dbc.Button(f"Predicted: {predicted_index}",
                                   color="success" if predicted_index == original_index else "danger",
                                   className="mr-1"),
                        dbc.Button(f"Original: {original_index}",
                                   color="success" if predicted_index == original_index else "danger",
                                   className="mr-1")
                    ], width=True),
                    dbc.Col([
                        html.Img(src=app.get_asset_url('colormap_Jet.png'), style={'height': '83%', 'width': '90%'})
                    ])
                ])
            ]),
        else:
            return dash.no_update


    @app.callback(
        [Output('hidden-div', 'children')],
        [Input('select_video', 'value')])
    def get_video(value):
        with open("./data/ucf101/split/testlist04.txt", "w") as f:
            f.write(value)
        global original_vid, gradcam_vid, play_video, predicted_indices, original_indices, video_frame_index
        original_vid, gradcam_vid, predicted_indices, original_indices = gradcam_sc1.main()
        frames_per_video = int(original_vid.shape[0] / len(original_indices))
        predicted_indices = [index for index in predicted_indices for _ in range(frames_per_video)]
        original_indices = [index for index in original_indices for _ in range(frames_per_video)]
        video_frame_index = 0
        play_video = True
        return dash.no_update


    app.run_server(debug=True)
