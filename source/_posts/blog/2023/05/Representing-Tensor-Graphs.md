---
title: Representing Tensor Graphs
date: 2023-05-30 13:46:33
tags:
 - tapestry
---

This post is about notes towards an implementable representation of the
"Abstract Expression Graph" / "Sharded Expression Graph" relationship in
[Tapestry](/Tapestry).

Expression languages differ from process languages in that define values in terms of
transformations on previous values. The simplest outcome of this is that it's quite
easy to use a given value more than once; but by adding an observer, we can define
directly which values are ever observed by the outside world.

Values which are never observed are free to be inlined (when they contribute to other
values which transitively are observed), or even eliminated entirely (when they don't
contribute to any observed values).

### Simple Expressions

```graphviz
digraph G {
  rankdir=RL;
  
  A [label="Tensor: A", shape=box3d, fillcolor="#d0d0ff", style=filled];
  
  obs [shape=doublecircle];
  obs -> A [arrowhead=dot];
}
```

What does it mean for us to be able to observe a tensor value?

* After the expression is evaluated, we can read the value of the tensor.

### Chained Expressions

We're generally interested in more complex expressions, where transformations are applied to
tensor values, and then to the results of those transformations, and so on.

```graphviz
digraph G {
  rankdir=RL;
  
  A [label="Tensor: A", shape=box3d, fillcolor="#d0d0ff", style=filled];
  B [label="Tensor: B", shape=box3d, fillcolor="#d0d0ff", style=filled];
  C [label="Tensor: C", shape=box3d, fillcolor="#d0d0ff", style=filled];
  D [label="Tensor: D", shape=box3d, fillcolor="#d0d0ff", style=filled];
  E [label="Tensor: E", shape=box3d, fillcolor="#d0d0ff", style=filled];
  
  X -> A;
  X -> B;
  X [label="BlockExpr: X", shape=Msquare, fillcolor="#d0ffd0", style=filled];
  C -> X;
  D -> X;
  
  Y -> D;
  Y [label="BlockExpr: Y", shape=Msquare, fillcolor="#d0ffd0", style=filled];
  
  E -> Y;

  obs [shape=doublecircle];
  obs -> E [arrowhead=dot];
}
```

In this example, the *Tensor: C* value is never observed, and so it can dropped entirely
from our schedule, or generated and *written* to a null-store by the block expr.

We are operating with a contract that if we provide the data in `A` and `B` to `X`;
that it will correctly produce `C` and `D` for us; and that this operation is idempotent.

Additionally, at this level it's quite possible that the tensors are abstractions
which could not fit on a single machine.

### Sharded Expressions

We are interested in the ability to:

* shard these operations and values;
* execute a given sharded schedule;
* to compare the costs (in time and space) of different sharding choices;
* and prune expression trees which are not transitivity observed.
 
```graphviz
digraph G {
  compound=true;
  rankdir=RL;
  
  subgraph cluster_aeg {
    label="Abstract Expression Graph";
    style=dotted;
    
    B [label="Tensor: B", shape=box3d, fillcolor="#d0d0ff", style=filled];
    A [label="Tensor: A", shape=box3d, fillcolor="#d0d0ff", style=filled];
    D [label="Tensor: D", shape=box3d, fillcolor="#d0d0ff", style=filled];
    
    subgraph cluster_C {
      label="Pruned";
      style=filled;
      color=lightgray;
      
      C [label="Tensor: C", shape=box3d, fillcolor="#d0d0ff", style=filled];
    }
    
    E [label="Tensor: E", shape=box3d, fillcolor="#d0d0ff", style=filled];
    
    X -> B;
    X -> A;
    X [label="BlockExpr: X", shape=Msquare, fillcolor="#d0ffd0", style=filled];
    C -> X;
    D -> X;
    
    Y -> D;
    Y [label="BlockExpr: Y", shape=Msquare, fillcolor="#d0ffd0", style=filled];
    
    E -> Y;
  }
  
  subgraph cluster_seg {
    label="Sharded Expression Graph";
    style=dotted;
    
    subgraph cluster_B {
      label="Tensor: B";
      style=solid;
      B1 [label="Tensor: B.1", shape=box3d, fillcolor="#d0d0ff", style=filled];
    }
    subgraph cluster_A {
      label="Tensor: A";
      style=solid;
      A1 [label="Tensor: A.1", shape=box3d, fillcolor="#d0d0ff", style=filled];
      A2 [label="Tensor: A.2", shape=box3d, fillcolor="#d0d0ff", style=filled];
    }
    
    subgraph cluster_X {
      label="BlockExpr: X";
      style=solid;
      X1 [label="Shard: X.1", shape=Msquare, fillcolor="#d0ffd0", style=filled];
      X2 [label="Shard: X.2", shape=Msquare, fillcolor="#d0ffd0", style=filled];
    }
    
    X1 -> A1;
    X1 -> B1;
    D1 -> X1;
    
    X2 -> A2;
    X2 -> B1;
    D2 -> X2;
    
    subgraph cluster_D {
      label="Tensor: D";
      style=solid;
      D1 [label="Tensor: D.1", shape=box3d, fillcolor="#d0d0ff", style=filled];
      D2 [label="Tensor: D.2", shape=box3d, fillcolor="#d0d0ff", style=filled];
    }
    
    Y1 -> D1;
    E1 -> Y1;
    
    Y2 -> D2;
    E2 -> Y2;
    
    subgraph cluster_Y {
      label="BlockExpr: Y";
      style=solid;
      Y1 [label="Shard: Y.1", shape=Msquare, fillcolor="#d0ffd0", style=filled];
      Y2 [label="Shard: Y.2", shape=Msquare, fillcolor="#d0ffd0", style=filled];
    }
    
    subgraph cluster_E {
      label="Tensor: E";
      style=solid;
      E1 [label="Tensor: E.1", shape=box3d, fillcolor="#d0d0ff", style=filled];
      E2 [label="Tensor: E.2", shape=box3d, fillcolor="#d0d0ff", style=filled];
    }
  }
  
  outputorder="edgesfirst";
  
  X2 -> X [constraint=false, style=dashed];
  X1 -> X [constraint=false, style=dashed];
  
  Y2 -> Y [constraint=false, style=dashed];
  Y1 -> Y [constraint=false, style=dashed];

  obs [shape=doublecircle];
  obs -> E2 [arrowhead=dot, lhead=cluster_E];
  obs -> E [arrowhead=dot];
}
```

This continues the assertion that this is an equivalent and correct sharding;
that each of the operations, if performed in dependency order, will produce the same
result as the original expression.

### Polyhedral Type Information

Being able to say:

* Expression `X'` is a sharded version of expression `X`

Is independent of our ability to:

* Verify that `X'` is a sharded version of `X`; or
* Given `X`, generate shareded versions `X'` and `X''`

If we have an execution environment for `X'`; having the sharded version is sufficient
for execution.

* Being able to describe the relative components in a tractable manner is the main project.

The additional information, needed to verify and generate sharded versions, is
the polyhedral type signature information attached to the expressions.

This is discussed in great detail in [Tapestry](/Tapestry);
the core ideas center around a characteristic shardable index space associated
with each expression, and affine projection functions (with resulting fixed
marginal steps) from that index space to the spaces of the inputs and outputs.

Finding a concrete representation to describe the relationships between the
abstract expression graphs, the polyhedral type information, and the sharded
expression graphs is the next major block on this project,
in a way which enables us to:

* Verify that the sharded graphs are correct;
* Generate sharded graphs from the abstract graphs;
* Generate abstract graphs from the sharded graphs;
* Apply a cost model to the sharded graphs;
* Write a stochastic optimizer to find good sharding choices.
 
#### The cost information

As a consequence of the choice of index spaces and index projection functions for the
*Tapestry* expression representations; we can show that the marginal data sharing
costs for input and output have constant marginal costs along each dimension of
the index space; e.g. the marginal cost change of including one additional step along a
batch dimension is constant, though different, than taking one additional step along
a channel dimension.

As the block compute model assumes shardable blocks which are location agnostic in
slice space; *Assuming* that the marginal compute/memory costs of blocks is linearly
related to their inputs along the above dimensions; we can take as an abstrac cost model the
notion of marginal resource cost per step along each dimension of the index space.

Additionally, at this layer we don't know what to *do* with those costs, that is a function
of the cost model / scheduling simulator (how are parallel costs managed? are transmission/bandwidth
costs elided when a tensor is being moved to the same machine it's already on; etc.);
so we can model costs as fixed marginal costs per step along each dimension of the index space;
for each of an arbitrary number of inputs.

Given an index space `I` with dimensions `batch, x, y, k`;

|       | gpu | ram |
|-------|-----|-----|
| batch | 1   | 1   |
| x     | 4   | 8   |
| y     | 4   | 8   |
| k     | 128 | 64  |

We also assume that the transmission of tensors is well modeled,
and that the marginal costs associated with the tensors is borne
entirely by the marginal data overlap and the transmissions costs.

Additionally, multiple sharded expressions can share the same shape
and cost information (as well as information about the operation being modeled).

In this diagram, we've added the marginal costs, the index projection functions
(<code>P<sub>a</sub>(idx)</code>), the abstract and concrete index, and tensor
selection slices to the information present in the block expression:

```graphviz
digraph G {
  compound=true;
  rankdir=RL;
  
  B [label="Tensor: B", shape=box3d, fillcolor="#d0d0ff", style=filled];
  A [label="Tensor: A", shape=box3d, fillcolor="#d0d0ff", style=filled];
  C [label="Tensor: C", shape=box3d, fillcolor="#d0d0ff", style=filled];
  
  
  subgraph cluster_expr {
    label="BlockExpr";
    style=filled;
    shape=Msquare;
    fillcolor="#d0ffd0";
    
    Index [shape=box3d, fillcolor="#d0d0ff", style=filled];
    Costs [shape=box3d, fillcolor="#d0d0dd", style=filled];
    
    ASel [label="Slice: A", shape=box3d, fillcolor="#d0d0dd", style=filled];
    BSel [label="Slice: B", shape=box3d, fillcolor="#d0d0dd", style=filled];
    CSel [label="Slice: C", shape=box3d, fillcolor="#d0d0dd", style=filled];
      
    subgraph cluster_signature {
      label="Signature";
      fillcolor="#ffffd0";
      
      AbstractIndex [shape=box3d];
      
      Pa [label=<P<sub>a</sub>(idx)>, shape=parallelogram];
      Pb [label=<P<sub>b</sub>(idx)>, shape=parallelogram];
      Pc [label=<P<sub>c</sub>(idx)>, shape=parallelogram];
      
      AbstractIndex -> Pa;
      AbstractIndex -> Pb;
      Pc -> AbstractIndex [dir=back];
      
      MarginalCosts [shape=box3d];
      
      {rank=same; MarginalCosts; AbstractIndex}
    }
    
    MarginalCosts -> Costs [constraint=false];
    AbstractIndex -> MarginalCosts;
    
    Pa -> MarginalCosts [constraint=false];
    Pb -> MarginalCosts [constraint=false];
    MarginalCosts -> Pc [dir=back, constraint=false];
    
    Index -> AbstractIndex [constraint=false];
  }
  
  Pa -> ASel;
  ASel -> A;
  
  Pb -> BSel;
  BSel -> B;
  
  CSel -> Pc [dir=back];
  C -> CSel [dir=back];
}
```

This information is necessary to make any changes to the sharding of the expressions; though it
is not necessary to schedule or execute a correct sharding as-is.

Additionally, there's annotation information we could include or derive, such as:

* the expected size of the input / output tensors
    * when married with a concrete execution schedule, this permits transmission bandwith/delay modeling.
* the expected compute costs of the block
    * CPU/GPU delay
    * Block memory usage

This information about blocks, describing the cost models, is needed in most places where the
polyhedral type information is needed.

### Encapsulation

When picking graph ownership mechanics, we're selecting between different encapsulation
options to represent the relationship between abstract and sharded expression graphs,
and the signatures which describe legal sharding and marginal costs.

Choosing a concrete representation of the above relationships determines the traversal API for
the compiler's cost models, mutation proposers, and debuggers. This in turn affects the
development and communication costs of the entire effort.

### Previous Work

I speculate that many of the previous efforts in this space have struggled under the requirement
that they start with a concrete expression sharding, and work backwards attempting to
derive an abstract graph and operator signatures for the associated expressions,
and then to produce transformations which maintain the semantics of the original
expressions.

And this has been difficult, because many of the languages in question lack
particularly strong shape signature information; most of the development
effort seems to get soaked up in this code analysis phase.
