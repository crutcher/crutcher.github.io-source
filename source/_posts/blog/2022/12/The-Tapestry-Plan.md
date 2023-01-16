---
title: The Tapestry Plan
tags:
  - tensor expressions
  - tapestry
mathjax: true
date: 2022-12-19 15:02:02
---


This post develops part of this document:
* [Tapestry: Shardable Tensor Expression Environments](/Tapestry)

# The Tapestry Plan

## Overview

I'm developing out a project in defining the bottom-up sharding and scheduling of grid-scale
tensor expression languages; its name is "Tapestry", for the way expression
value flow graphs weave between execution contexts.

I am of the opinion that this is a project which requires no *new* computer science;
just the careful and methodical application of pieces from a number of sub-fields.

As there are many projects exploring how to take existing evaluation environments
and back-fit sharding machinery too them, and as those projects are continuing to
make reasonable progress, I feel that there's no short-term urgency to solve this;
so I'm taking the pure-language design route.

  * We don't have users, and won't have them till the whole stack works. We won't have
    to worry about maintaining semantics or operational decisions when problems are
    encountered with them.

  * We will have some trouble acquiring people to help; everything is going to
    appear *very* abstract until the functional machinery is in-place.


## Stages

*Tapestry* will be built out in the following stages of work, which correspond to a series
of technical embeddings going deeper into the stack, and will remain as rewrite layers.

  * Tensor Valued Block Operation Graph Language
  * Block Operation Index Projection Sharding Graph Language 
  * Block Operation Substitution Rewrite Graph Language 
  * Block Operation Fusion Rewrite Graph Language 
  * Block Operation Rewrite Sharding Optimizer
    * Configurable Execution Cost Model
    * Pareto-Front Stochastic Search Optimizer
  * Block Shard Operational Embedding
  * Block Shard Grid Host
    * Shard Scheduling
    * Shard Metric Instrumentation

When this stack is *semantically* complete, even in a preview form; we can begin to
preview applications written in the block operation graph language.

From this stage onward, development will bifurcate:
  * Applications and extensions written *above* the block operation language; and
  * Optimization research and operational implementation work done *below* the block operation language.

The goal is:

> Provide an existence proof that a provably shardable formal language is possible
> (we can prove that it *can* be made fast); then make it easy to program for to
> get more help; then make it fast.