# EPOCH
This is a sequence-based fine-tuned ESM-2 model, named HLPartner, for predicting paired epitope and paratope.  you can clone this repository locally:
```
git clone https://github.com/zzyywww/EPOCH.git 
cd EPOCH
pip install -r requirements.txt --ignore-installed
```
Download the model
```
wget https://i.uestc.edu.cn/EPOCH.zip
unzip EPOCH.zip
```
If you are using a Windows system, you can copy the link and paste it into browser to download the file directly.

**Note:** The unzipped EPOCH folder must be in the same directory as the EPOCH.py script."

Usage:

```
python EPOCH.py [inputfile]
```
inputfile: sequence file, 'id' column for **sequence id**, 'antibody_seq1	' column for **heavy chain** sequence ,  'antibody_seq2' column for  **light chain** sequence. For single-chain antibodies, the antibody_seq2 column should remain blank. 'antigen_seq' for **antigen** sequence
The example of inputfile can be found in **./example/example_seq.txt**

Example:
```
python EPOCH.py ./example/example_seq.txt
```
The results will be saved in **results** folder automatically, or you can modify the source code in **EPOCH.py** to save your results more customistically. The result file provides the predicted probability for each amino acid. The predicted labels are obtained using the default threshold of 0.1191, where ​1 indicates an interacting amino acid and ​0 indicates a non-interacting amino acid.
