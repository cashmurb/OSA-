# OSA: Observational Spatiotemporal Anomaly Detector

Official implementation of **OSA**, a modular framework for weakly supervised abnormality detection in paediatric echocardiographic video, developed as part of a Final Year Project at Xi'an Jiaotong-Liverpool University (XJTLU).

OSA combines a frozen UniFormerV2-B/16 backbone with LoRA adapters, a Temporal Aggregation and Anomaly Head (TAH), and Masked Temporal Reconstruction (MTR) pretraining to detect functional cardiac abnormalities from video-level labels only.

## Highlights
- Parameter-efficient adaptation via LoRA (1.18M trainable parameters)
- Self-supervised MTR pretraining without frame-level annotations
- Controlled ablation isolating temporal aggregation and spatial 
  prior contributions
- Evaluated on EchoNet-Pediatric (7,810 paediatric echo videos)
- Best variant achieves AUROC 0.882, PR-AUC 0.698, F1 0.644

## Citation
If you use this work, please cite:

> Cashmere Blanche Ruben Alin. *OSA: Observational Spatiotemporal 
> Anomaly Detector in Weakly Supervised Echocardiographic Video 
> Analysis*. BEng Final Year Project, School of AI and Advanced 
> Computing, Xi'an Jiaotong-Liverpool University, May 2026.

## Acknowledgements
This work was conducted at the **School of AI and Advanced Computing, Xi'an Jiaotong-Liverpool University (XJTLU)**, supervised by Dr. Netzahualcoyotl Hernandez-Cruz, and supported by the Research Development Fund Grant No. RDF-25-01-076.
