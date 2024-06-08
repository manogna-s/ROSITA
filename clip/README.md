# ROSITA

Official code for the paper 
**"Effectiveness of Vision-Language Models for Open-World Single Image Test Time Adaptation"**

[[Project-page]](https://manogna-s.github.io/rosita/) [[arXiV]](https://arxiv.org/abs/2406.00481)
 
[//]: # ([Manogna Sreenivas]&#40;https://manogna-s.github.io&#41; and Soma Biswas.)

[//]: # (![image]&#40;docs/ROSITA.png&#41;)
![image](../docs/ROSITA_gif.gif)
> <p align="justify"> <b> <span style="color: green;">ROSITA framework</span></b>:
> The test samples with weak and strong OOD data arrive one at a time. The image features are matched with the text based classifier, the confidence scores of which are used to distinguish between weak and strong OOD samples through a simple LDA based OOD classifier. Based on this classification and if a sample is identified to be reliable, the feature banks are updated and test-time objective is optimized to update the LayerNorm parameters of the Vision Encoder.

Viz credits: [Rohit](https://in.linkedin.com/in/rohitronie)


[//]: # (> **<p align="justify"> Abstract:** *We propose a novel framework to address the real-world challenging task of Single Image Test-Time Adaptation in an open and dynamic environment. We leverage large-scale Vision Language Models like CLIP to enable real-time adaptation on a per-image basis without access to source data or ground truth labels. Since the deployed model can also encounter unseen classes in an open world, we first employ a simple and effective Out of Distribution &#40;OOD&#41; detection module to distinguish between weak and strong OOD samples. )

[//]: # (We propose a novel contrastive learning based objective to enhance discriminability between weak and strong OOD samples by utilizing  small, dynamically updated feature banks. )

[//]: # (By pushing weak and strong OOD samples apart in the latent space using reliable test samples, our method continuously improves OOD detection performance. )

[//]: # (Finally, we also employ a classification objective for adapting the model using the reliable weak OOD samples. )

[//]: # (The final objective combines these components, enabling continuous online adaptation of Vision Language Models on a single image basis. )

[//]: # (Extensive experimentation on diverse domain adaptation benchmarks validates the effectiveness of the proposed framework.* </p>)

----------
### Installation

Please follow the instructions at [INSTALL.md](../docs/INSTALL.md) to setup the environment.


### Dataset preparation

Please follow the instructions at [DATASETS.md](../docs/DATASETS.md) to prepare the datasets.

### Experiments 

Please follow the instructions at [RUN.md](../docs/RUN.md) to run the experiments.

-------------
### Citation

If you find our code useful or our work relevant, please cite our paper:

```angular2html
@misc{sreenivas2024effectiveness,
      title={Effectiveness of Vision Language Models for Open-world Single Image Test Time Adaptation}, 
      author={Manogna Sreenivas and Soma Biswas},
      year={2024},
      eprint={2406.00481},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


-------------
### Acknowledgements

We thank the authors of [TPT](https://github.com/azshue/TPT), [PromptAlign](https://github.com/jameelhassan/PromptAlign), [OWTTT](https://github.com/Yushu-Li/OWTTT) for their open-source implementation, which helped us establish the baselines for this work.




