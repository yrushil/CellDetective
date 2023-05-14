import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import scanpy as sc
import anndata

def annotate_cluster(adata: anndata.AnnData, cluster_ids: str):
    #Tumor Lists
    Tumor = ["SOX2"]

    #Immune Cell Lists
    Immune_cells = ["IL7R","CD3E","CD3D","CD8A","CD8B", "MS4A1", "BANK1","CD79A","MZB1","TNFRSF17",'LILRA4']
    Macrophages = ["S100A8", "CD14","LYZ",'FCGR3A','CD1C',"CST3","FCER1G"]
    T_cells = ["IL7R","CD3E","CD3D","CD8A","CD8B","NKG7"]
    B_cells = ["MS4A1", "BANK1","CD79A"]  

    #Brain Cells
    Brain_tissue = ["CD163","MBP", "ST18", "MOG","ALDH1L1", "NDRG2", "S100B"]
    Microglia = ["CD163"]
    Oligo = ["MBP", "ST18", "MOG"]
    Pericyte = ["MYO1B", "PDGFRB", "LAMC3"]
    Astrocyte = ["ALDH1L1", "NDRG2", "S100B"]

    #Other
    Endo = ["PECAM1", "VWF", "ANGPT2"]
    cell_types = [Tumor, Immune_cells, Macrophages, T_cells, B_cells, Brain_tissue, Microglia, Oligo, Pericyte, Astrocyte,Endo]
    cell_names = ['Tumor', 'Immune_cells', 'Macrophages', 'T_cells', 'B_cells', 'Brain_tissue', 'Microglia', 'Oligo', 'Pericyte', 'Astrocyte','Endo'] 
    
    from scipy import stats
    #CALCULATE GENE SCORES
    x = 0
    for genes_set in cell_types:
        sc.tl.score_genes(adata, genes_set,  score_name = cell_names[x] + '_score')
        x+=1
    df = adata.obs.groupby([cluster_ids]).mean().reset_index().groupby(cluster_ids).mean()
    
    #IDENTIFY TUMOR CELLS
    threshold = 0.5
    scores = ['Tumor_score']
    for score_label in scores:
        z_scores = stats.zscore(df[score_label])
        outliers = df[z_scores > threshold]
        tumor_clust = outliers.index.tolist()
        df['cluster_labels'] = 'Tumor'
    
    df_filt = df[~df.index.isin(tumor_clust)]

    #IDENTIFY IMMUNE CELLS
    scores = ['Immune_cells_score']
    threshold = 0.1
    for score_label in scores:
        z_scores = stats.zscore(df_filt[score_label])
        outliers = df_filt[z_scores > threshold]
        immune_clust = outliers.index.tolist()

    immune_df = df_filt[df_filt.index.isin(immune_clust)]
    brain_df =  df_filt[~df_filt.index.isin(immune_clust)]
    
    # Classifying Immune Cells
    scores = ['Macrophages_score', 'T_cells_score', 'B_cells_score']
    names = ['Macrophages', 'T cells', 'B cells']

    x = 0
    threshold = 0.1
    for score_label in scores:
        z_scores = stats.zscore(immune_df[score_label])
        outliers = immune_df[z_scores > threshold]
        immune_clust = outliers.index.tolist()
        for cluster in immune_clust:
            df.loc[df.index == cluster, 'cluster_labels'] = names[x]
        x += 1
    
    # CLASSIFYING BRAIN TISSUE
    scores = ['Endo_score','Pericyte_score','Oligo_score',  'Astrocyte_score']
    names = ['Endothelial cells','Pericytes',' Oligodendrocytes','Astrocytes']

    threshold = 0.99
    temp_df = brain_df  

    x = 0
    for score_label in scores:
        z_scores = stats.zscore(temp_df[score_label])
        outliers = temp_df[z_scores > threshold]
        brain_clust = outliers.index.tolist()
        for cluster in brain_clust:
            df.loc[df.index == cluster, 'cluster_labels'] = names[x]
        x += 1

        temp_df =  temp_df[~temp_df.index.isin(brain_clust)]
    
    for cluster in temp_df.index.tolist():
        df.loc[df.index == cluster, 'cluster_labels'] = 'Microglia'
    df = df.iloc[:, -1:]
    new_index = df.index.tolist()
    new_cluster_labels = df['cluster_labels'].tolist()

    output_df = pd.DataFrame(new_cluster_labels, index = new_index, columns =['cluster_labels'])
    return output_df

colors = ['tomato', 'paleturquoise', 'cornflowerblue', 'mediumseagreen',
    'mediumpurple', 'goldenrod', 'lightgreen', 'palevioletred',
    'lightsalmon', 'thistle', 'brown', 'plum', 'lightskyblue',
    'yellowgreen', 'mediumorchid', 'mediumturquoise', '#F08000',
    '#bf812d', 'mediumaquamarine', 'lightslategrey', 'dodgerblue',
    '#4daf4a', '#377eb8']

marker_genes_dict = {"Macrophages": ["S100A8", "CD14","LYZ",'FCGR3A','CD1C',"CST3","FCER1G"],
                     'Mono':["S100A8", "CD14","LYZ",'FCGR3A'],
                     'DCs':['CD1C',"CST3","FCER1G",],
                     'T cells':["IL7R","CD3E","CD3D","CD8A","CD8B",],
                     'B cells':["MS4A1", "BANK1","CD79A"],                     
                     'PCs':["MZB1","TNFRSF17",'LILRA4',],
                     "Tumor": ["SOX2","PTPRZ1","EGFR"],
                     "Microglia":["PTPRC", "CD163", "IL1B"],
                     "Oligo":["MBP", "ST18", "MOG"],
                     "Pericyte":["MYO1B", "PDGFRB", "LAMC3"],
                     "Endoth":["PECAM1", "VWF", "ANGPT2"],
                     "Astrocyte":["ALDH1L1", "NDRG2", "S100B"]
                    }

st.title('Cluster annotation of glioma single-cell data')
st.set_option('deprecation.showPyplotGlobalUse', False)

uploaded_file = st.file_uploader("Upload AnnData Object", type="h5ad")
if uploaded_file is not None:
    file_to_annotate = anndata.read_h5ad(uploaded_file)

    #GATHER INFOMRATION ABOUT OBJECT
    st.header('Information about the AnnData Object')
    st.write("Number of cells:", file_to_annotate.n_obs)
    st.write("Number of genes:", file_to_annotate.n_vars)
    file_to_annotate.obs.to_csv('temp.csv')
    obs_data = pd.read_csv('temp.csv')
    st.subheader("AnnData Observations:")
    st.write(obs_data)

    #CLUSTER THE ANNDATA OBJECT
    st.header('Cluster cells Using Leiden')
    res = st.number_input('Enter cluster resolution value')
    if res != 0:
        st.text("Computing UMAP...")
        #file_to_annotate.uns['neighbors']['params']['metric'] = 'cosine'
        sc.tl.umap(file_to_annotate)
        st.text("Computing Leiden clustering...")
        sc.tl.leiden(file_to_annotate, resolution = res,key_added='clusters')

        #PLOTTING CLUSTERS ON UMAP PLOT
        st.subheader("Clusters on UMAP Plot")
        st.pyplot(sc.pl.umap(file_to_annotate, color=['clusters'],legend_loc ='right margin',frameon=False, title='',legend_fontsize=9,legend_fontoutline=True))

        #ANNOTATE THE ANNDATA OBJECT
        st.header('Annotating Clusters...')
        df = annotate_cluster(file_to_annotate, 'clusters')
        st.write(df)

        st.header('Plotting Annotated Clusters')

        st.subheader("Annotated Clusters on UMAP Plot")
        file_to_annotate.obs['annotated_clusters'] = file_to_annotate.obs['clusters']
        df['cluster_num'] = df.index.astype(str)
        cluster_labels = 'c'+ df['cluster_num'] + '_' + df['cluster_labels']
        cluster_labels = cluster_labels.to_numpy()
        cluster_labels = cluster_labels.tolist()
        st.text("Plotting...")
        file_to_annotate.rename_categories('annotated_clusters', cluster_labels)
        st.pyplot(sc.pl.umap(file_to_annotate, color=['annotated_clusters'],legend_loc ='right margin',frameon=False, title='',legend_fontsize=12,legend_fontoutline=True))

        st.subheader("Dot Plot of Gene Expression Level of Marker genes")
        st.text("Plotting...")
        sc.tl.dendrogram(file_to_annotate, groupby='annotated_clusters')
        st.pyplot(sc.pl.dotplot(file_to_annotate, marker_genes_dict, groupby='annotated_clusters', dendrogram=True, swap_axes=True))


