# Overcoming Generic Knowledge Loss with Selective Parameter Update (CVPR'24)

<a target="_blank" href="https://arxiv.org/pdf/2308.12462">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv">
</a><a target="_blank" href="https://github.com/wx-zhang/spu">
<img style="height:22pt" src="https://img.shields.io/badge/-Code-black?style=flat&logo=github"></a>
<a target="_blank" href="https://wx-zhang.github.io/spu/html">
<img style="height:22pt" src="https://img.shields.io/badge/-Project Page-white?style=flat&color=%236f91b6"></a>
<br>


<span style="color:black; font-size: 14pt; font-family: Roboto, Helvetica, Arial, Heveltica Neue, sans-serif">
     <b>Authors:</b> <a class="name" target="_blank" href="https://wx-zhang.github.io/">Wenxuan Zhang</a>, 
     <a class="name" target="_blank" href="https://pauljanson002.github.io/">Paul Janson</a>,
     <a class="name" target="_blank" href="https://scholar.google.fr/citations?user=YLh7yrwAAAAJ&hl=en">Rahaf Aljundi</a>,
     <a class="name" target="_blank" href="https://www.mohamed-elhoseiny.com/">Mohamed Elhoseiny</a>&nbsp; @ 
     <a class="btna" target="_blank" href="https://cemse.kaust.edu.sa/vision-cair/vision-cair">KAUST Vision-CAIR</a>, 
          <a class="btna" target="_blank" href="https://www.toyota-europe.com/about-us/toyota-in-europe/toyota-motor-europe">TME</a> &nbsp &nbsp; 
     </span>
     <br>



Use this repo to reproduce the results of our methods as well as the baselines.




## Installation
```
conda env create -f environment.yml
conda activate clip
``` 
Use the following to install the learning rate scheduler 
```
pip install 'git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup'
```
## Reproduce Our Results
### Dataset
Prepare the datasets by following the instructions in the `data` folder.
- CIFAR 100, FGVC-Aircraft, GTSRB: Automatically downloaded by torchvision.
- CUB-200-2011, Stanford Cars: Automatically downloaded by Huggingface. 
- Birdsnap: Follow the download instructions from [here](https://thomasberg.org/).
- CC12M: Follow the download instructions from [here](https://github.com/google-research-datasets/conceptual-12m).
- ImageNet 1k: Follow the download instructions from [the official website](http://image-net.org/).

### Reproduce our method
```
python main.py dataset=[cifar100 | cub | cars | aircraft | gtsrb | birdsnap ]
```

## Run baselines
Supported baselines: 
- `flyp`: Finetune like you pretrain [[paper](https://arxiv.org/abs/2212.00638)]
- `er`: Experimence replay [[paper](https://arxiv.org/abs/1902.10486)]
- `lwf`: Learning without forgetting [[paper](https://arxiv.org/abs/1606.09282)]
- `mas`: Memory aware synapses [[paper](https://arxiv.org/abs/1711.09601)]
- `prd`: Prototype-sample relation distillation [[paper](https://arxiv.org/abs/2303.14771)]
- `loraewc`: LoRA finetune with EWC regularization[[paper](https://arxiv.org/abs/2305.10626)]
- `slca`: Slow learner with classifier alignment [[paper](https://arxiv.org/abs/2303.05118)]
- `sparsecl`: Sparse Continual Learning [[paper](https://arxiv.org/abs/2209.09476)]
- `spg`: Soft-masks parameter updating  [[paper](https://arxiv.org/pdf/2306.14775)]
- `zscl`: Zero-shot Continual Learning [[paper](https://arxiv.org/abs/2303.06628)]

```
python main.py \
    dataset=[cifar100 | cub | cars | aircraft | gtsrb | birdsnap ] \
    baseline@_global_=[flyp | er | lwf | mas | prd | loraewc | slca | sparsecl | spg | zscl]
```

### Supported features
For replay based method, use `balanced_buffer=False` to apply uniform sampling (uniformly from buffer and the current task)
```
python main.py dataset=your_dataset baseline@_global_=your_baseline balanced_buffer=False
```
Use `joint=True` for joint training
```
python main.py dataset=your_dataset baseline@_global_=your_baseline joint=True
```
Adjust `buffer_size` to scaling down or up the buffer size 
```
python main.py dataset=your_dataset baseline@_global_=your_baseline buffer_size=0.5
```
Adjust `num_tasks` to adjust the number of split of dataset 
```
python main.py dataset=your_dataset baseline@_global_=your_baseline num_tasks=20
```
## Acknowledgement
- Learning rate scheduler: [Cosine Annealing with Warmup for PyTorch](https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup)
- CLIP backbone: [OpenCLIP](https://github.com/mlfoundations/open_clip)
- Classic continual learning methods: [Avalanche](https://github.com/ContinualAI/avalanche)

## Citation
```
@inproceedings{zhang2024overcoming,
          title={Overcoming Generic Knowledge Loss with Selective Parameter Update},
          author={Zhang, Wenxuan and Janson, Paul and Aljundi, Rahaf and Elhoseiny, Mohamed},
          booktitle={CVPR},
          year={2024}
        }
```
