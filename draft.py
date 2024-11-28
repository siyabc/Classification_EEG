from mne.io import read_epochs_eeglab
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

epochs = read_epochs_eeglab(r'C:\Users\chenzhijia\Desktop\机器学习\sedation-restingstate\Sedation-RestingState\02-2010-anest- 20100210 16.003.set')
data = epochs.get_data()

avg_data = np.mean(data, axis=0)

corr_matrix = np.corrcoef(avg_data)

G = nx.Graph()

num_nodes = avg_data.shape[0]
G.add_nodes_from(range(num_nodes))

threshold = 0
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):
        weight = corr_matrix[i, j]
        if weight > threshold:
            G.add_edge(i, j, weight=weight)

pos = nx.spring_layout(G)
nx.draw(G, pos=pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=500, font_size=10)

edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

plt.show()
