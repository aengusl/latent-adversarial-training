# Targeted Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs

Abhay Sheshadri,* [asheshadri31@gatech.edu](asheshadri31@gatech.edu); 
Aidan Ewart,* [aidanprattewart@gmail.com](aidanprattewart@gmail.com); 
Phillip Guo,* [phguo@umd.edu](phguo@umd.edu); 
Aengus Lynch,* [aenguslynch@gmail.com](aenguslynch@gmail.com);
Cindy Wu,* [wu.cindyx@gmail.com](wu.cindyx@gmail.com);
Vivek Hebbar*;
Henry Sleight;
Asa Cooper Stickland
Ethan Peres
Dylan Hadfield-Menell
Stephen Casper, [scasper@mit.edu](scasper@mit.edu)

arXiv and BibTeX coming soon!

See our models on Hugging Face Hub at [https://huggingface.co/LLM-LAT](https://huggingface.co/LLM-LAT).

## This repository

This repository contains code for implementing latent adversarial attacks 
and latent adversarial training (LAT) in LLMs. 

<figure>
  <img src="figs/fig1.png" alt="Targeted Latent Adversarial Training">
  <figcaption>Targeted Latent Adversarial Training (LAT) in LLMs: We perturb the latent activations
in an LLM’s residual stream to elicit specific failure modes from the model. Then, we fine-tune
LLMs on the target task under these perturbations. We use this approach to improve robustness to
jailbreaks (Section 4.1), remove backdoors without access to the trigger (Section 4.2), and unlearn
undesirable knowledge (Section 4.3).</figcaption>
</figure>


## Setup

After you clone and navigate to the repository:

```angular2html
pip install -r requirements.txt
bash install_tasks_from_github.sh
```


## Ready to go with the notebooks

Find notebooks for latent space attacks, jaiblreak robustness, 
backdoor removal, harry potter unlearning, and wmdp unlearning 
in the ```/notebooks``` folder.
