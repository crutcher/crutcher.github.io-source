---
title: 'Tapestry: The Vision'
tags:
  - tapestry
date: 2023-01-02 15:24:58
---


This post develops part of this document:
* [Tapestry: Shardable Tensor Expression Environments](/Tapestry)

# The Tapestry Vision

To explain the larger vision of Tapestry, we need to explore the uses cases
of a large system which does not yet exist, which we'll also call A Tapestry.

> Note: The motivation for the synecdoche here is taken from SQL, where SQL is both
> the language and the environment, as their semantics are formal and shared.

Grid-scale datacenters filled with GPU nodes are becoming commonplace;
datacenters with 1000s of server-grade GPUs, commonly hosted on densely networked
machines with 2-8 GPUs per machine. These machines have tremendous theoretical
compute throughput, but existing programming environments for them require
multiple layers of systems engineers and operations engineers to successfully
exploit the theoretical potential.

Common use involves compiling specialized application images for a given task
to be run on these systems, allocating some subset of the available machines
as machines for that task, pushing machine images to each machine allocated
for that subset, and running a controller and worker instances built
from the compiled application image. The entire workflow is inefficient
and fragile, and encourages sizing datacenter usage to the max storage
or compute needs that a task will need for its entire lifecycle.

Suppose we instead had a uniform collection of interconnected nodes with unified management,
holding both storage and compute resources. Interconnected service grids
where point-to-point communication is abstracted into semantic routing
are frequently called "meshes" and "fabrics"; to be specific here
we'll call this:
* a tensor fabric, or
* a tapestry environment

<img src="/Tapestry/tapestry.pastel.svg"/>

The individual nodes in this environment would be largely opaque to us; we 
would not send data or jobs to them individually, or push virtual machine images to them;
they act in concert as a unified environment, and we work with them
in terms of the total environment.

One way of thinking of this is as a *very* large `numpy` or `torch` environment.

Suppose we can perform a few operations in this environment:
* Allocate and manage named tensors in the environment.
* Copy portions of tensors into other tensors.
* Load and export external data into and from named tensors in the environment;
  for example from and to databases and network file sources.

<img src="/Tapestry/tapestry.io.svg"/>

This is a very basic environment; and for now, we've omitted a number of details.
* How is the data split between nodes?
* How is node recovery managed (do we have duplicate copies of the data)?


Given only the ability to create storage, move data around, and move data
into and out of the tapestry; we've defined an environment with scratch-space semantics.
We could find a use for this environment; run data jobs which unpack their data
and construct large tensor objects, followed by data jobs which re-pack that
data differently.

Now, suppose in addition to injecting data into our environment, we'd like to 
be able to inject functions which manipulate the data. The environment
has many compute nodes, but they're distributed; it stores a lot of data,
but not in places or shards we know about.

To be able to inject functions, we desire a way to describe semantic actions
we wish to take on the data ("apply this function to that tensor, yielding a new tensor"):
* without specifying the details of the layout or scheduling,
* while getting close to theoretically resource optimal results.

A high-level desired workflow would permit us to:
1. Load tensor data into tapestry: <img src="/Tapestry/tapestry.basic.load.svg" width="300"/>
2. Inject and apply an operation, transforming existing tensors into new tensors:
   <img src="/Tapestry/tapestry.basic.apply.svg" width="300"/>
3. Export the result tensor data: <img src="/Tapestry/tapestry.basic.save.svg" width="300"/>

The details of the operation injection step are the crucial issue here;
finding a family of operations which can be injected into such an environment
and yield optimal resource (time and space) efficient results with minimal external
knowledge about the tapestry layout.

Many environments exist for flat-mapped collections, for data that is structured
in lists, arrays, or key/value mapped dictionaries:
* [Apache Spark](https://spark.apache.org/)
* [Google Dataflow](https://cloud.google.com/dataflow)

These environments do not model tensor-indexed values, or have effective
mechanisms for distributing dataflow and functional dependencies across
polyhedrally typed operations; a new formalism is needed to effectively
describe operator injection into distributed tensor environments.

Tapestry is an effort to describe such a formalism, focusing on shardable by construction
operation graph semantics.

## Applications

A brief review of some applications of a tapestry environment.

### Artificial Intelligence / Machine Learning

Deep learning AI/ML applications describe models as stacks of tensor-valued weights,
connected by a computation graph describing how tensor-valued data injected into
the model evolves through a series of compute steps to produce the predictions
of the model, or to compute changes to the models weights for learning.

Existing systems are built on stacks of `numpy`-inspired APIs, with minimal
optimization passes; an API which was not designed to restrict itself to shardable
operations. As a result, existing systems struggle with operational engineering
to achieve effective distribution and sharding, and leave a great deal of
theoretical throughput unexploited.

An optimizing tapestry environment could greatly speed AI/ML research, by removing
the requirements for much of the task-specific operational scaling engineering;
while simultaneously reducing the research costs, by optimizing the resulting
operations graphs to more effectively utilize available resources.

### Finite Element Simulations

Finite element simulations decompose a problem (weather, heat flow, stress) onto
tensors describing a space to be simulated, valued in terms of the material
properties at each point in the space (density, energy, pressure, material, heat, etc).

Finite element simulations evolve by describing a kernel transition function,
which predicts the next time value for a given point in space by applying a
kernel transition function which examines only the tensor values of the
local neighborhood of that point at the previous time value.

This is equivalent to iteratively applying a kernel convolution to
produce successive time step world tensors.

An effective tapestry environment sufficient to host finite element simulations
would permit accelerated research into anything built upon finite element simulations;
which includes a great deal of modern engineering and physical sciences applications.
