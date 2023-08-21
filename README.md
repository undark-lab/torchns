# torchns
Nested sampling in torch

----
## What is it?

`torchns` is a nested sampler with a slice sampling exploration scheme based on [torch](https://github.com/pytorch/pytorch) library. It is designed to be integrated with the [swyft](https://github.com/undark-lab/swyft) sequential simulation-based inference code.

Its main features are:
- Vectorized evaluations of slice sampling chains to draw new live points.
- Vectorized evaluations of the log-likelihood.
- Functionality to define constrained prior regions, useful for sequential simulation-based inference applications. 
  
----
## Installation

- Change directory to wherever you would like to store the library, then run:
```
git clone https://github.com/undark-lab/torchns.git # for https client
[or git clone git@github.com:undark-lab/torchns.git # for ssh client]
```
- Making sure that the desired python environment is active, run the following installation code:
```
cd torchns/
pip install .
```
- This will install `torchns` in the current python environment that is active on your system and will be available via `import torchns`

----
## Further information

- **Source code:** [https://github.com/undark-lab/torchns](https://github.com/undark-lab/torchns)
- **Example usage:** [https://github.com/undark-lab/torchns/examples](https://github.com/undark-lab/torchns/examples)
- **Support & discussion:** [https://github.com/undark-lab/torchns/discussions](https://github.com/undark-lab/torchns/discussions)
- **Bug reports:** [https://github.com/undark-lab/torchns/issues](https://github.com/undark-lab/torchns/issues)
- **Related paper:** `torchns` was first introduced in [arxiv:2308.08597](https://arxiv.org/abs/2308.08597).

----
## Release Details

- **v0.0.1** | *August 2023* | Initial release based on [arxiv:2308.08597](https://arxiv.org/abs/2308.08597).

----
## Relevant other nested sampling packages
- [jaxns](https://github.com/Joshuaalbert/jaxns) is a JAX based nested sampler.
- [proxnest](https://github.com/astro-informatics/proxnest) implements the proximal nested sampling algorithm.
- [Polychord](https://github.com/PolyChord) is a nested sampler that uses slice sampling.
- [MultiNest](https://github.com/JohannesBuchner/MultiNest) is a nested sampler that uses ellipsoidal sampling.
- [dynesty](https://github.com/joshspeagle/dynesty) is a dynamic nested sampler.
