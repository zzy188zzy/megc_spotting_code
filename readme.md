## A Multi-scale Feature Learning Network with Optical Flow Correction for Micro-and Macro-expression Spotting
- Use Openface Toolkit to crop and align faces
- Extracting facial features using Videomae-v2-ginat finetuned on the k710 dataset
- Install Python package according to Actionformer
- Run train_cas.py / train_samm.py (Pay attention to adjusting the path)
- Run eval.py to obtains candidate segments
- Run redetector.py to filter the candidate segments
