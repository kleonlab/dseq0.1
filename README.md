# dseq0.1
follow up to dseq, trying out structure space instead of sequence space


## setup and downloads 
the transcription factor database is downloaded from: "https://humantfs.ccbr.utoronto.ca/download.php" 
Download the human TFs list of the gene names and the ensemble IDs. 
After the csv downloads, run the setup file in dataload to get the clean and usable fasta versions. 

## finetune setup - first download boltzgen from the git repo by 
'''git clone https://github.com/HannesStark/boltzgen.git models/boltzgen''' and then run '''uv pip install -e ./models/boltzgen''' 

## use this to get the data from replog - wget -O "datasets/k562.h5ad" "https://ndownloader.figshare.com/files/35774440" 

then run the train with the file path location. 
