# NetInGraph

## 1\_flow

P&R scripts are in scripts/innovus\_pr.tcl
    ## invoking scripts/zhiyao.tcl to save output

Example Output: output/b14.txt
    ## with cell and net information

----------------------------------------------------------------
##2\_preprocess

Input: dataT\_folder/b14.txt

2.1 python save\_hyperGraph.py -> output: edgeT_folder/
    python save_hyperGraph_node.py -> output: edgeT_node_folder/
    ## generate graph connections, used by hmetis for partitioning 

2.2 python hmetis_generate_Diffnode.py 
    python hmetis_hyperGraph_Diffnode.py -> output: clusT_Lnode_more*/
    ## partition with hmetis, and save to clusters (cell as node)

    python hmetis_hyperGraph.py -> output: clusT_folder*/
    ## partition with hmetis, and save to clusters (net as node)

2.3 python pre_generate.py -> output: output/data_Graph.pickle
    ## save a lot of different features on nodes and edges

----------------------------------------------------------------
3_ml

Input: data generated in previous steps

3.1 GNN/sGAT.py
    ## GNN method
    ## torch_geometric library used

3.2 customizedGNN/wireGraph.py
    ## customized GNN method
    ## torch_geometric library used


