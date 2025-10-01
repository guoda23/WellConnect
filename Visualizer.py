import networkx as nx
import pandas as pd
#TODO: remove??
class Visualizer: #TODO: adjust this to communicate with a data file where the graph graphs/data are
    def __init__(self): 
        """Initialize the Visualizer"""

    def show_network(self, G, layout="circular"):
        """
        Visualizes a given network graph.

        Parameters:
            G (networkx.Graph): The graph to visualize.

        Outputs:
            Displays the graph visualization with labeled nodes and edge weights.
        """

        # Select layout
        if layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            raise ValueError(f"Unsupported layout: {layout}")

        nx.draw(G, pos, with_labels=True, font_weight='bold')
        edge_labels = nx.get_edge_attributes(G, 'weight')

        if edge_labels:
            nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=8, font_color='red')


    def display_graph(self, group_graphs, group_ids = None, layout="spring"):
        """
        Displays the specified group graphs or all graphs if no IDs are provided.

        Parameters:
            group_graphs (dict): Dictionary of group graphs.
            group_ids (list): List of group IDs to display (default is None).
            layout (str): Graph layout method ("spring" or "circular").

        Outputs:
            Displays the network graph visualizations.
        """

        if group_ids is None:
            group_ids = group_graphs.keys()

        for group_id in group_ids:
            if group_id not in self.group_graphs:
                print(f"Group ID {group_id} not found. Skipping.")
                continue

            print(f"Displaying Group {group_id}...")
            G = self.group_graphs[group_id]
            self.show_network(G, layout=layout)


    def display_group_traits(self, grouped_data, group_ids = None):
        """Displays the trait values of inidividuals in the specified group(s).

        Parameters:
            - grouped_data (pd.DataFrame): The DataFrame containing group information.
            - group_ids (list or int): Group ID(s) to display.

        Outputs:
            Prints the rows of the DataFrame corresponding to the specific group(s).
        """

        if isinstance(group_ids, int):
            group_ids = [group_ids]
        
        if "group_id" not in grouped_data.columns:
            raise ValueError("'group_id' column not found in grouped_data.") #TODO: make this more flexible? (as input)
        
        if group_ids is None:
            print("Displaying traits for all groups:")
            print(grouped_data)
            return


        for group_id in group_ids:
            if group_id not in self.group_graphs:
                print(f"Group ID {group_id} not found in group graphs. Skipping.")
                continue

            group_data = grouped_data[grouped_data["group_id"] == group_id]

            # Check if there is data for the group
            if group_data.empty:
                print(f"No data found for Group ID {group_id}.")
            else:
                print(f"\nDisplaying traits for Group ID {group_id}:")
                print(group_data)

                
                
        