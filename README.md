# COSMAP
COSine MAtrix Predictor is an approach and a neural network for protein-protein interaction prediction. 


Two folders containing PKLs of trained networks and ZIP with should be availible at [Google Drive](https://drive.google.com/drive/folders/1Ie4GQh8ocsVjxe9ssAzMpX9lLP7tsY1D?usp=drive_link).

The code has next dependencies:
- numpy
- torch
- pytorch-lightning <pre>  (better pre 2.0.0, Lightning often regress)</pre>
- pytorch-msssim
- matplotlib <pre>  (for notebooks)</pre>
- biopython <pre>  (for PDB processing)</pre>
- scipy <pre>  (solely for cdist(..) fucntion)</pre>
