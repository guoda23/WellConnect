import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
import plotly.tools as tls
import numpy as np

class Visualizer3DScatterPlot:
    def __init__(self, scatter_data, x_label="Weight Entropy", y_label="Trait Entropy", z_label="Statistical Power"):
        """
        Takes extracted metrics from OutputGenerator.
        scatter_data contains:
        - (x, y, z) values
        - Corresponding NetworkX group networks (precomputed).
        """
        self.scatter_data = scatter_data
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label

        self.group_list = [d["group_network"] for d in scatter_data if d["group_network"] is not None]  

        if not self.group_list:
            print("Warning: No valid group networks found in scatter_data!")

        # Initialize Dash App
        self.app = dash.Dash(__name__)
        self.setup_layout()


    def setup_layout(self):
        """ Sets up the Dash app layout with a scatter plot and network graph. """
        self.app.layout = html.Div([
            html.H3("3D Scatter Plot of Group Metrics"),
            dcc.Graph(id="scatter-plot", figure=self.get_scatter_fig()),

            html.H3("Selected Group Network"),
            dcc.Graph(id="network-graph")
        ])

        @self.app.callback(
            Output("network-graph", "figure"),
            Input("scatter-plot", "clickData")
        )
        def update_network_graph(clickData):
            """ Updates the network graph when a point is clicked. """
            if clickData is None:
                return go.Figure()  # Empty figure

            # Access the index from 'customdata'
            selected_index = clickData.get("points", [{}])[0].get("customdata", None)
            if selected_index is None:
                return go.Figure()  # Return an empty figure if no valid point is selected

            # Retrieve group network using index
            selected_group = self.group_list[selected_index]

            # Ensure it's a valid NetworkX graph
            if not isinstance(selected_group, nx.Graph):
                print(f"⚠️ Error: Selected group is not a NetworkX graph! Received: {type(selected_group)}")
                return go.Figure()  # Return an empty figure

            return self.plot_network(selected_group)


    def run(self, port=8050):
        """ Starts the Dash server. """
        self.app.run_server(debug=True, port=port)

    def get_scatter_fig(self):
        """ Generates a 3D scatter plot of group metrics. """
        x_data = [d["weight_entropy"] for d in self.scatter_data]
        y_data = [d["trait_entropy"] for d in self.scatter_data]
        z_data = [d["stat_power"] for d in self.scatter_data]

        fig = go.Figure(data=[go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode='markers',
            marker=dict(size=6, color=z_data, colorscale="Viridis"),
            text=[f"Point {i}" for i in range(len(self.scatter_data))],  # Just for hover info
            customdata=list(range(len(self.scatter_data))),  # Stores index for lookup
        )])

        fig.update_layout(
            title="3D Scatter of Group Metrics",
            scene=dict(
                xaxis=dict(title=self.x_label),  
                yaxis=dict(title=self.y_label),  
                zaxis=dict(title=self.z_label)       
            )
        )

        return fig


    def plot_network(self, group_network, edge_width=2):
        """ Generates a 2D network visualization using Plotly with edge weights and labels. """
        G = group_network

        if G is None or not isinstance(G, nx.Graph):
            return go.Figure()  # Return empty figure if group is missing or invalid

        # Prepare the network layout using NetworkX's spring layout
        pos = nx.spring_layout(G, seed=42)

        for node in G.nodes():
            if node not in pos: # Assign random 2D position if node not in pos
                pos[node] = np.random.rand(2)  

        # Extract node positions and labels
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [str(node) for node in G.nodes()]

        edge_x = []
        edge_y = []
        edge_weights = []
        edge_labels = []
        #midpoints for edge labels
        edge_mid_x = [] 
        edge_mid_y = []

        for edge in G.edges(data=True):  # True for edge attributes (incl. weights)
            node1, node2, edge_data = edge
            weight = edge_data.get('weight', 0)  # Access edge weight from dict; default to 0 if missing

            # Get positions of the nodes from pos dict
            if node1 in pos and node2 in pos:
                x0, y0 = pos[node1]
                x1, y1 = pos[node2]

                edge_x.append(x0)
                edge_y.append(y0)
                edge_x.append(x1)
                edge_y.append(y1)
                edge_weights.append(weight)

                # Midpoint for placing the label
                mid_x = (x0 + x1) / 2
                mid_y = (y0 + y1) / 2
                edge_mid_x.append(mid_x)
                edge_mid_y.append(mid_y)
                edge_labels.append(f'{weight:.2f}')  # Format weight as string with 2 decimals

        if not edge_weights:
            return go.Figure() # Return empty figure if no edges are processed

        edge_trace = []
        for i in range(len(edge_x) // 2):
            edge_trace.append(go.Scatter(
                x=[edge_x[2 * i], edge_x[2 * i + 1]], y=[edge_y[2 * i], edge_y[2 * i + 1]],
                mode='lines',
                line=dict(width=edge_width, color='gray'),
                hoverinfo='none'
            ))

        # Create Plotly figures for nodes
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            marker=dict(size=10, color='blue', opacity=0.8),
            text=node_text,
            hoverinfo='text'
        )

        # Create Plotly figures for edge labels (weights)
        edge_label_trace = go.Scatter(
            x=edge_mid_x,
            y=edge_mid_y,
            mode='text',
            text=edge_labels,
            textposition='top center',
            showlegend=False
        )

        # Combine the traces into one figure
        fig = go.Figure(data=edge_trace + [node_trace, edge_label_trace])

        # Update layout and return the figure
        fig.update_layout(
            title="Group Network Graph",
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        return fig
