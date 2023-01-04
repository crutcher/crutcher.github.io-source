---
title: 'Tapestry: Parallel Stochastic Optimization'
tags:
  - tapestry
mathjax: true
date: 2023-01-04 13:48:03
---

This post develops part of this document:
* [Tapestry: Shardable Tensor Expression Languages](/Tapestry)


## Parallel Stochastic Optimization

The tapestry work thus far has focused on establishing rewrite rules
to find equivalent evaluation graphs to an initial high-level abstract program.
Given an initial graph $G$ in a system of formal semantics, we have established
rules which permit us to mechanically derive a large family of alternative
graphs ($G_1$, $G_2$, $G_3$, ...) which evaluate to the same results
under that system of formal semantics.

Tapestry is designed to be amenable to parallel stochastic multi-objective
optimization; the choices made thus far have focused on enabling effective
use of parallel optimizer search.

An optimizer can be described, at a very high level, as a process to take
an initial graph $G$, a cost model $C: G \rightarrow cost$, and a family of rewrite rules
$\\{ R_i: G \rightarrow G' \\}$, and select the lowest-cost $G'$ graph it can find.

In some optimization problems, the cost model returns not a single value,
but a collection of values we might be simultaneously interested in improving. For example:

* the total memory usage
* the total compute usage
* the expected wall-clock time
* the peak node memory usage
* the peak node compute usage
* the node memory utilization waste
* the node compute utilization waste

### Stochastic Pareto Frontier Optimization

Enter the field of [multi-objective optimization](https://en.wikipedia.org/wiki/Multi-objective_optimization);
which is the research field into optimization when we have multiple dimensions to care about.
This section is a brief overview of multi-objective optimization, as it applies to tapestry
plans.

Given an existing population of instance trials $\\{ G_i \\}$, we can run our cost
model on each trial $C_i = C(G_i)$, and produce a multi-dimensional cost value.
Placing those costs in space, we can establish a surface known as the "Pareto frontier",
made up of all instances which are better than any other instance on at least one dimension:

<img src="/Tapestry/optimization/pareto.basic.svg"/>

The **Pareto frontier** represents the instances (found so far) making the best trade-offs
between resources we care about from our cost model.

When searching for improvements, we can select one (or more, in the case of graph or genetic cross-over)
instance(s) of the instances $G_i$ from the **pareto frontier** (or, in the case of some models,
sampled proportionally relative to their distance from the frontier); apply one or more
of the mutation rules, producing a new instance $G'$, and run the cost model to establish
the new cost $C': C(G')$, placing the new instance somewhere in the optimization space:

<img src="/Tapestry/optimization/pareto.trial.svg"/>

With the addition of a new cost-annotated instance, we recompute the **pareto frontier**;
if the new instance represents an improvement, we move the frontier:

<img src="/Tapestry/optimization/pareto.update.svg"/>

There are many ways to improve this. It's common to sample parent instances from points *near*
the frontier, even if they no longer lie upon it; and the generalization of that is to say that
there's distribution of parent selection probability which is willing to sample any instance
within some distance of the frontier with some probability relative to that distance.

<img src="/Tapestry/optimization/pareto.neighborhood.svg"/>

A large motivation for the sampling approach is that many mutations may not produce better
children, but might enable further mutations which do, and we don't want to close ourselves
to exploring those options.

Further, it may be the case that there are regions of the optimization space which constitute
external constraints; but we'd still like to include instances outside that region to permit appropriate
exploration.

For example, our initial graph $G$ likely has peak node memory and compute utilization greater
than any of our existing compute resources; we can't schedule it at all, but it's the basis
for our initial optimization search.

<img src="/Tapestry/optimization/pareto.constraint.svg"/>

### Graph Mutator Selection Optimization

There's also a host of research about how to balance selecting which mutation rules
from our collection of mutation rules $\\{ R_i: G \rightarrow G' \\}$ to apply.

In practice, not every mutator rule can apply to every graph; so we can extend our description
of mutations to include applicability rules $\\{ A_i: G \rightarrow bool \\}$; such that
for a given instance, we only consider rules $\\{ R_i | A_i(G) \\}$ where the
applicability test for the rule says it applies.

We could select from these rules uniformly, or hand-code their firing probabilities.
In practice, it is common to tune the triggering rate for optimization rules against
metrics collected over a series of benchmarks.

As long as *every* rule has a chance to apply, and *every* instance has a chance to be
selected, then the entire space of legal graphs constructed by some series of mutations
is reachable; though it may be intractable to fully search, so we'd like to bias
our exploration to things likely to yield improvements.

One approach is to track the mutation histories of each instance (the series of mutation
rules which lead to each instance), and compute the rates at which each mutation rule
contributed to improvements in the pareto frontier.

This can be done offline, by running benchmarks and hard-coding the resulting values;
or it can be done dynamically while running a given optimization.

In practice, a combination approach is powerful: offline benchmarking to establish a prior distribution,
combined with dynamic statistics based in the observed optimization histories
attached to a given problem.

One additional value of dynamic mutator rule balancing is that it eases use of
third-party and application-specific mutations rules.

### Parallel Optimization

Given that our instances are very small relative to the data they operate on
(they describe execution graphs), and our cost models are relatively abstract
(they compute expected compute and memory and timing costs for a given graph);
we expect that examining any given instance will be very fast and small, compared
to actually running the described execution.

If optimization evaluation is sufficiently fast and small, and if mutators
have a high enough chance of producing improvements, a local optimization
loop running on one machine, in one thread, has a good chance of producing
a good execution graph for our needs.

But if the graph is complicated, or the rules do not immediately produce
improvements, or if the graph optimization surface has lots of local minima;
we may need to examine parallel optimization.

Parallel optimization is running trials in multiple threads, or even potentially
on multiple machines, in parallel. Stochastic pareto front optimization is
well suited for parallel optimization; at the limit, machines only need
to communicate when they've found improvements to the pareto frontier.
Optimizing this form of search is a large field of research.

One interesting approach to take is to run the search as long as continuing
to do so is expected to reduce the total value of some function of the
cost model.

Say, we're currently seeing a 5% improvement every 1000 trials of the optimization
search? When should we stop looking for a better answer? An optimal choice
depends on:
* how expensive are the optimization trials?
* how valuable is that 5% reduction in the optimized graph schedule?

When targeting jobs meant to run for weeks to months on 1000s of GPUs;
we may reasonably aim to run the optimizer on 100 machines for a few hours,
if doing so reliably reduces the long term utilization.

However, when targeting jobs which should take 5 machines 20 minutes;
the target optimization time should probably be a great deal shorter.
