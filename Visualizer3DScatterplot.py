import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash import dash_table
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import pandas as pd


class Visualizer3DScatterPlot:
    def __init__(self, data, x_label="Weight Entropy", y_label="Trait Entropy", z_label="Statistical Power"):
        """
        Takes extracted metrics from OutputGenerator.
        data contains:
        - (x, y, z) values
        - Corresponding NetworkX group networks (precomputed).
        -
        #TODO:finish the doc string
        """
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        self.z_label = z_label

        self.group_list = [d["group"] for d in data if d["group"] is not None]  

        if not self.group_list:
            print("Warning: No valid group networks found in data!")

        # Initialize Dash App
        self.app = dash.Dash(__name__)
        self.setup_layout()


    def setup_layout(self):
        """ Sets up the Dash app layout with a scatter plot, network graph, and tables. """
        self.app.layout = html.Div([
            html.H3("3D Scatter Plot of Group Metrics"),
            dcc.Graph(id="scatter-plot", figure=self.get_scatter_fig()),

            # Selected Group Network Graph
            html.H3("Selected Group Network"),
            dcc.Graph(id="network-graph"),

            # List of Agents in the Group with Trait Values
            html.H3("List of Agents and Their Trait Values"),
            dash_table.DataTable(id="agents-table"),

            # Regression Results Table
            html.H3("Regression: Recovered Weights with reference of True Weights (SEE selected = True!)"),
            dash_table.DataTable(id="regression-table"),


        ])

        @self.app.callback(
            [Output("network-graph", "figure"),
             Output("regression-table", "data"),
             Output("agents-table", "data")],
            [Input("scatter-plot", "clickData")]
        )
        def update_graph_and_tables(clickData):
            """ Updates the network graph and tables when a point is clicked. """
            if clickData is None:
                return go.Figure(), [], []  # Empty figures and data if no click

            # Access the index from 'customdata'
            selected_index = clickData.get("points", [{}])[0].get("customdata", None)
            if selected_index is None:
                return go.Figure(), [], []  # Return empty if no valid point is selected

            # Retrieve group network using index
            selected_group = self.group_list[selected_index]
            selected_group_network = selected_group.network

            # Ensure it's a valid NetworkX graph
            if not isinstance(selected_group_network, nx.Graph):
                return go.Figure(), [], []  # Return empty if not a valid graph

            # Plot the network
            network_fig = self.plot_network(selected_group_network)

            # Prepare the data for the tables
            regression_data = self.get_regression_data(selected_index)
            agents_data = self.get_agents_data(selected_group)

            return network_fig, regression_data, agents_data


    def highlight_row(self, regression_data, selected_index):
        """ Adds a highlight flag to the relevant row for the regression table. """
        for row in regression_data:
            row['highlight'] = (row['Recovered Weight'] == selected_index)  # Use the index to mark the row
        return regression_data


    def run(self, port=8050):
        """ Starts the Dash server. """
        self.app.run_server(debug=True, port=port)


    def get_scatter_fig(self):
        """ Generates a 3D scatter plot of group metrics. """
        x_data = [d["weight_entropy"] for d in self.data]
        y_data = [d["trait_entropy"] for d in self.data]
        z_data = [d["stat_power"] for d in self.data]

        hover_text = [f"{self.x_label}: {x:.4f}<br>{self.y_label}: {y:.4f}<br>{self.z_label}: {z:.4f}" for x, y, z in zip(x_data, y_data, z_data)]
        # text=[f"Point {i}" for i in range(len(self.data))], 

        fig = go.Figure(data=[go.Scatter3d(
            x=x_data, y=y_data, z=z_data,
            mode='markers',
            marker=dict(size=6, color=z_data, colorscale="Viridis"),
            text=hover_text,  # Just for hover info
            customdata=list(range(len(self.data))),  # Stores index for lookup
            hoverinfo = "text",
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


    def plot_network(self, group_network, edge_thickness_range=(1, 5)):
        """ Generates a 2D network visualization using Plotly with edge weights and labels. """
        G = group_network

        if G is None or not isinstance(G, nx.Graph):
            return go.Figure()  # Return empty figure if group is missing or invalid

        # Prepare the network layout using NetworkX's spring layout
        pos = nx.spring_layout(G, seed=42) #TODO: make seed input dynamic?

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

        # Normalize weights to fit within the provided thickness range
        min_weight = min(edge_weights) if edge_weights else 0
        max_weight = max(edge_weights) if edge_weights else 1  # Avoid division by zero

        min_thickness, max_thickness = edge_thickness_range

        edge_trace = []
        for i in range(len(edge_x) // 2):
            edge_weight = edge_weights[i]

            if edge_weight > 0.001:  # Only draw edges if there's a connection
                norm_weight = (edge_weight - min_weight) / (max_weight - min_weight) if max_weight > min_weight else 1
                thickness = min_thickness + norm_weight * (max_thickness - min_thickness)

                edge_trace.append(go.Scatter(
                    x=[edge_x[2 * i], edge_x[2 * i + 1]],
                    y=[edge_y[2 * i], edge_y[2 * i + 1]],
                    mode='lines',
                    line=dict(width=thickness, color='gray'),
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

        edge_label_x = []
        edge_label_y = []
        edge_labels = []

        for i in range(len(edge_mid_x)):  # Iterate over the midpoints
            if edge_weights[i] > 0.001:  # Filter out edges with no homophily
                edge_label_x.append(edge_mid_x[i])
                edge_label_y.append(edge_mid_y[i])
                edge_labels.append(f'{edge_weights[i]:.2f}')  # Keep only meaningful labels

        edge_label_trace = go.Scatter(
            x=edge_label_x,
            y=edge_label_y,
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
    

    def get_regression_data(self, selected_index):
        """ Returns the regression data for the table, including the true weights row and highlighting logic. """
        group_data = self.data[selected_index]
        df_regression = group_data['recovered_weights_df']
        true_weights = group_data['true_weights']

        # Display NaN values
        df_regression = df_regression.fillna("N/A")


        # Add the true weights as a row to the DataFrame
        true_weights_row = {'group_id': 'TRUE'}  # Set 'group_id' to 'TRUE' for the true weights row
        for col in df_regression.columns:
            if col != 'group_id':  # Skip the 'group_id' column
                if col in true_weights:
                    true_weights_row[col] = true_weights[col]
                else:
                    true_weights_row[col] = 'N/A'  # In case a column is missing in true weights

        true_weights_df = pd.DataFrame([true_weights_row])
        df_regression = pd.concat([df_regression, true_weights_df], ignore_index=True)

        # Prepare the data to return with styles
        regression_data = []
        
        for index, row in df_regression.iterrows():
            row_data = row.to_dict()

            # Mark the selected row
            if index == group_data['row_of_interest_in_table']:
                row_data['selected'] = True  # Mark the selected row
            else:
                row_data['selected'] = False  # Other rows are not highlighted

            # Mark the true weights row
            if row_data['group_id'] == 'TRUE':
                row_data['selected'] = True  # Highlight the true weights row
            
            regression_data.append(row_data)

        return regression_data


    def get_agents_data(self, group):
        """ Returns a dummy DataFrame for agent attribute data. """
        # Creating a simple dummy DataFrame with agent data
        attribute_df = group.get_attribute_table()
        return attribute_df.to_dict('records')

