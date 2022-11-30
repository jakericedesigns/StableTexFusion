# StableTexFusion
A WIP method for generating seamless texture mapping guided Stable Diffusion

---
## Usage Info
to install this, just like use conda.
you'll need to install pytorch3d and diffusers:
```
conda create -n pytorch3d python=3.9
conda activate pytorch3d
conda install -c pytorch pytorch=1.9.1 torchvision cudatoolkit=10.2
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c conda-forge diffusers

```
Then just run the stabletexfusion.py script, and use the flag `--help` if you need help :) 

I could probably streamline this or make a google colab, which I might. but if you ask for it, I'll take even longer because I have severe anxiety :) 
I'm probably missing steps in the install as well, but I'm sure y'all can figure it out!

Requires having like an input obj, with an associated MTL file and texture map, though tbh in the current cut i throw out all input textures. Your object needs to have vertex UV's as well. I believe in you!

Currently it takes a while on my 3090, like 30 minutes, and the outputs are hit or miss. But it's still cool. 
