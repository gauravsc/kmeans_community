import networkx as nx
import numpy
import Pycluster
import igraph
import random
import matplotlib.pyplot as plt
_num_cluster=3
random.seed(9)
def kmean_community_detection():   
    Graph_networkx=nx.Graph(nx.read_dot("twittercrawl.dot"))
    nodes=Graph_networkx.nodes()
    dict_nodes={}
    for i in range(len(nodes)):
        dict_nodes[nodes[i]]=i

    X=[]    
    for node in nodes:
        neighbors_node=Graph_networkx.neighbors(node)
        node_nb_vector=[0]*len(nodes)
        for node_nb in neighbors_node:
            node_nb_vector[dict_nodes[node_nb]]=1
        X.append(node_nb_vector)

    X=numpy.array(X)

    labels, error, nfound = Pycluster.kcluster(X, _num_cluster)
    communities=[[] for i in range(_num_cluster)]
    communities_index=[[] for i in range(_num_cluster)]
    for i in range(len(labels)):
        communities[labels[i]].append(nodes[i])
        communities_index[labels[i]].append(i)

    return Graph_networkx,communities,communities_index




def igraph_community_detection():
    Graph_igraph=igraph.read("twitter.gml",format="gml")
    Graph_igraph=Graph_igraph.as_undirected()
    fg_communities=Graph_igraph.community_fastgreedy()
    fg_communities=fg_communities.as_clustering()
    sizes_comm=fg_communities.sizes()
    community_id=fg_communities.membership
    vertices=igraph.VertexSeq(Graph_igraph)
    nodes_igraph=[vertices[i]['label'] for i in range(len(vertices))]
    communities=[[] for i in range(len(fg_communities.sizes()))]
    for i in range(len(nodes_igraph)):
        communities[community_id[i]].append(nodes_igraph[i])


    return communities  

def get_average_clustering(Graph_nx,cluster):
    cluster_coeff=0
    for i in range(len(cluster)):
        g=Graph_nx.subgraph(cluster[0])
        cluster_coeff+=nx.average_clustering(g)
    cluster_coeff/=len(cluster)    
    return cluster_coeff


def save_all_subgraphs(clusters,G,prefix):
    nodes=G.nodes()
    
    i=0
    for cluster in clusters:
        i+=1
        g=nx.Graph(G.subgraph(cluster))
        nx.draw(g,node_size=1,with_labels=False)
        plt.savefig(prefix+str(i)+".png")
        plt.clf()


def kmean_community_detection_neighbor_information():   
    Graph_networkx=nx.Graph(nx.read_dot("twittercrawl.dot"))
    nodes=Graph_networkx.nodes()
    dict_nodes={}
    for i in range(len(nodes)):
        dict_nodes[nodes[i]]=i
    degrees=Graph_networkx.degree()
    X=[]    
    for node in nodes:
        neighbors_node=Graph_networkx.neighbors(node)
        node_nb_vector=[0]*len(nodes)
        for node_nb in neighbors_node:
            node_nb_vector[dict_nodes[node_nb]]=float(1)/(degrees[node_nb])       
        X.append(node_nb_vector)


    X=numpy.array(X)

    labels, error, nfound = Pycluster.kcluster(X, _num_cluster)
    communities=[[] for i in range(_num_cluster)]
    communities_index=[[] for i in range(_num_cluster)]
    for i in range(len(labels)):
        communities[labels[i]].append(nodes[i])
        communities_index[labels[i]].append(i)

    return Graph_networkx,communities,communities_index






cluster_igraph=igraph_community_detection()
#Graph_nx,cluster_kmeans,communities_index=kmean_community_detection_neighbor_information()
#average_clustering_kmeans=get_average_clustering(Graph_nx,cluster_kmeans)
Graph_nx,cluster_kmeans,communities_index=kmean_community_detection()
average_clustering_kmeans=get_average_clustering(Graph_nx,cluster_kmeans)
average_clustering_greedy=get_average_clustering(Graph_nx,cluster_igraph)

print "k means clustering coeff",average_clustering_kmeans
print "fast greedy clustering coeff", average_clustering_greedy
save_all_subgraphs(cluster_kmeans,Graph_nx,"kmeans_"+str(_num_cluster))
save_all_subgraphs(cluster_igraph,Graph_nx,"igraph_")




'''
c=set(cluster_igraph[0]).intersection(set(cluster_kmeans[0]))
print "igraph cluster length", len(set(cluster_igraph[0]))
print "kmeans cluster length",len(set(cluster_kmeans[0]))
print "intersection cluster length",len(c)
'''









    