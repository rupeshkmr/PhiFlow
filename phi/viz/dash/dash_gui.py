# coding=utf-8
import dash_core_components as dcc
import dash_html_components as html

from phi.struct.tensorop import collapsed_gather_nd
from phi.viz.dash.board import build_benchmark, build_tf_profiler, build_tensorboard_launcher
from phi.viz.dash.log import build_log
from phi.viz.dash.model_controls import build_model_controls
from phi.viz.dash.viewsettings import build_view_selection
from .dash_app import DashApp
from .info import build_app_details, build_description, build_phiflow_info, build_app_time
from .viewer import build_viewer
from phi.viz.dash.player_controls import build_status_bar, build_player_controls
from ..display import AppDisplay


class DashGui(AppDisplay):

    def __init__(self, app):
        AppDisplay.__init__(self, app)
        self.dash_app = None

    def setup(self):
        header_layout = html.Div([
                dcc.Link('Home', href='/'),
                ' - ',
                dcc.Link('Side-by-Side', href='/side-by-side'),
                ' - ',
                dcc.Link('Info', href='/info'),
                ' - ',
                dcc.Link('Log', href='/log'),
                ' - ',
                dcc.Link(u'Φ Board', href='/board'),
                # ' - ',
                # dcc.Link('Scripting', href='/scripting'),
            ])
        dash_app = self.dash_app = DashApp(self.app, self.config, header_layout)

        # --- Shared components ---
        player_controls = build_player_controls(dash_app)
        status_bar = build_status_bar(dash_app)
        model_controls = build_model_controls(dash_app)

        # --- Home ---
        layout = html.Div([
            build_description(dash_app),
            build_view_selection(dash_app),
            html.Div(style={'width': 1000, 'height': 800, 'margin-left': 'auto', 'margin-right': 'auto'}, children=[
                build_viewer(dash_app, id='home', initial_field_name=collapsed_gather_nd(self.config.get('display', None), [0])),
            ]),
            status_bar,
            player_controls,
            model_controls,
        ])
        dash_app.add_page('/', layout)

        # --- Player ---
        layout = html.Div([
            build_view_selection(dash_app),
            html.Div(style={'width': '50%', 'display': 'inline-block'}, children=[
                build_viewer(dash_app, id='left'),
            ]),
            html.Div(style={'width': '50%', 'display': 'inline-block'}, children=[
                build_viewer(dash_app, id='right'),
            ]),
            status_bar,
            player_controls,
            model_controls,
        ])
        dash_app.add_page('/side-by-side', layout)

        # --- Log ---
        layout = html.Div([
            dcc.Markdown('# Log'),
            status_bar,
            player_controls,
            build_log(dash_app)
        ])
        dash_app.add_page('/log', layout)

        # --- Info ---
        layout = html.Div([
            build_description(dash_app),
            status_bar,
            player_controls,
            build_phiflow_info(dash_app),
            build_app_details(dash_app),
            build_app_time(dash_app),
        ])
        dash_app.add_page('/info', layout)

        # --- Board ---
        layout = html.Div([
            dcc.Markdown(u'# Φ Board'),
            status_bar,
            player_controls,
        ] + ([] if 'tensorflow' not in dash_app.app.traits else [
            build_tensorboard_launcher(dash_app),
        ]) + [
            build_benchmark(dash_app),
            model_controls,
        ] + ([] if 'tensorflow' not in dash_app.app.traits else [
            build_tf_profiler(dash_app),
        ]) + [
            # ToDo: 'Graphs, Record/Animate, Exit/Restart/ShutdownUI',
        ])
        dash_app.add_page('/board', layout)

        # --- Scripting ---
        layout = html.Div([
            dcc.Markdown(u'# Python Scripting'),
            'Custom Fields, Execute script, Restart'
        ])
        dash_app.add_page('/scripting', layout)

        return self.dash_app.dash

    def show(self, caller_is_main):
        if caller_is_main:
            port = self.config.get('port', 8051)
            print('Starting Dash server on http://localhost:%d/' % port)
            self.dash_app.dash.run_server(debug=True, host='0.0.0.0', port=port, use_reloader=False)
            return self
        else:
            return self.dash_app.server
