import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.conversion import localconverter

# load r packages
biocParallel = importr('BiocParallel')
deseq = importr('DESeq2')

setParallel = ro.r('function(x) register(SnowParam(x))')

# keep gene rows here fpm > 1 for more than half the samples 
isExpr = ro.r('''
function(dds) {
    dds <- estimateSizeFactors(dds)
    isExpr <- rowSums(fpm(dds)>1) >= 0.5 * ncol(dds)
    ddsExpr <- dds[isExpr,]
    return(ddsExpr)
    }
    '''
)

isExpr2 = ro.r('''
function(dds) {
    geoMeans <- apply(counts(dds), 1, function(row) if (all(row == 0)) 0 else exp(mean(log(row[row != 0]))))
    dds <- estimateSizeFactors(dds, geoMeans=geoMeans)
    isExpr <- rowSums(fpm(dds)>1) >= 0.5 * ncol(dds)
    ddsExpr <- dds[isExpr,]
    return(ddsExpr)
    }
    '''
)

# calculate geomean of gene row 
geomeans = ro.r('function(dds) { apply(counts(dds), 1, function(row) if (all(row == 0)) 0 else exp(mean(log(row[row != 0])))) }')

# convert array to
to_dataframe = ro.r('function(x) as.data.frame(x)')

sizeFactor = ro.r('function(dds) sizeFactors(dds)')

""" Class to Deseq2 R package
    ...
    Attributes
    ----------
    genes : array
        array of deseq genes
    geneToGeomean : dict 
        maps gene to geomean of gene counts
"""
class DeseqRunner:
    def __init__(self, counts, columnData, designStr):
        self.counts = counts
        self.columnData = columnData 
        self.design = Formula(designStr)
    
    def run(self,withCalcGeoMean=False):
        setParallel(4)
        with localconverter(ro.default_converter + pandas2ri.converter):
            counts = ro.conversion.py2rpy(self.counts)
        with localconverter(ro.default_converter + pandas2ri.converter):
            columnData = ro.conversion.py2rpy(self.columnData)
        dds = deseq.DESeqDataSetFromMatrix(countData=counts, colData=columnData, design=self.design)
        geoMeans = geomeans(dds)
        geneToGeomean = {}
        for i, gene in enumerate(np.array(self.counts.index)):
            geneToGeomean[gene] = geoMeans[i]
    
        dds = isExpr2(dds) if withCalcGeoMean else isExpr(dds)
        
        self.size_factors = sizeFactor(dds)
        
        dds = deseq.DESeq(dds)
        
        deseq_result = deseq.results(dds)
        self.deseq_result_r = to_dataframe(deseq_result)
        
        with localconverter(ro.default_converter + pandas2ri.converter):
            self.deseq_results_df = ro.conversion.rpy2py(self.deseq_result_r)
            
        tmp = self.deseq_results_df[self.deseq_results_df['padj'] < 0.01]
        tmp = tmp[np.abs(tmp['log2FoldChange']) > 1]
        self.deseq_filtered = tmp 
        self.genes = np.array(tmp.index)
        selected_geneToGeomean = {}
        for gene in self.genes:
            selected_geneToGeomean[gene] = geneToGeomean[gene]
        self.geneToGeomean = selected_geneToGeomean
        return self.genes

""" Normalize based on size factor
    For each sample, size factor is the median of all the ratios of gene count to the geomean of gene 
"""
class SizeFactorNormalize:
    def __init__(self, gene_to_geomean):
        self.gene_to_geomean = gene_to_geomean
    
    def calculateSizeFactor(self, columns, counts):
        # get ratio for each gene
        values = counts.astype(float)
        size_factors = []
        for sampleI in range(len(values)):
            ratios = []
            for geneI in range(len(values[0])):
                # calculat ratio of count to geomean for each gene 
                ratios.append(values[sampleI][geneI]/self.gene_to_geomean[columns[geneI]])
            size_factors.append(np.median(ratios))
        return size_factors
    
    def normalize(self, columns, counts):
        """ normalize gene counts using calculated geomean
        Parameters
        ----------
        counts : array
            gene counts data; rows are patients, columns are gene counts
        Returns array of normalized genecount
        """
        sf = self.calculateSizeFactor(columns, counts) # calculate size factor for new samples based on geomeans of gene counts 
        return self.normalize_with_size_factors(counts, sf)
    
    def normalize_with_size_factors(self, counts, size_factors):
        """ normalize gene counts based on provided size factors
        Parameters
        ----------
        counts : array
            gene counts data; rows are patients, columns are gene counts
        size_factors : bool, optional
            size factor for each patient, size_factor[i] is size factor for patient[i]
        Returns array of normalized genecount
        """
        values = counts.astype(float)
        for i in range(len(values)):
            for j in range(len(values[0])):
                values[i][j] /= size_factors[i]
        return counts 