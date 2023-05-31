---
title: Working in ZSpace
tags:
  - tapestry
mathjax: true
date: 2023-05-26 14:55:55
---

$$\begin{eqnarray\*}
ZSpace := \\{ \mathbb{Z}^n | n \in \mathbb{Z}^+ \\}
\end{eqnarray\*}$$

"ZSpace" is a common shorthand, typographically simple name for the infinite
family of $n$-dimensional discrete Euclidean spaces.

The $n$-dimensional coordinates of discrete-celled tensors (the kind of tensors we work with on computers)
are *ZSpace* objects, as are bounding regions selecting those coordinates, and morphisms or maps from one region to
another.

Though we could, in principle, simply call a coordinate an array of integers;
performing any non-trivial index math on discrete location $n$-dimensional tensors requires
libraries for representing and manipulating these tensors.

As I've been working on pieces of [Tapestry: Shardable Tensor Expression Environments](/Tapestry);
most of the work has be focused on libraries for manipulating objects in ZSpace without spending
all of my time debugging math errors.

Most tensor libraries I've been able to examine, for Python, C++, Java, and Rust, focus primarily on abstracting
the details of using hardware accelerated vectorized floating point operations. They carry big dependency costs,
and have lots of runtime call patterns, driven by this.

So I've been building my own ZSpace libs, which cannot represent anything other than integer values;
because my focus isn't on the performance of the calculations of the data in the values; but on correctly manipulating
(with type checking and runtime assertions) index regions describing shards of expressions.

This is, for instance, the ZTensor and tests:

* [ZTensor.java](https://github.com/crutcher/loom/blob/main/java/src/main/java/loom/zspace/ZTensor.java)
* [ZTensorTest.java](https://github.com/crutcher/loom/blob/main/java/src/test/java/loom/zspace/ZTensorTest.java)

This is a situation where the existing libraries were just not built for manipulating polyhedral types and ranges
in ZSpace; where we frequently wish to perform transforms which result in coordinates.

There's a tremendous amount of clever little tricks wrapped up in how tensor libs get built; and how
things like `transpopse`, `permute`, `reverse`, `select`, `squeeze`, `unsqueeze`, and `broadcastTo`
can be implemented with zero-copy views which read or write back to their parent; and I may do a series on "How to write
a Tensor";
but for now a fair number of those tricks are wrapped up in that code.

## Side Note: Size, Z^0, and Scalars

The size of a contiguous slice of ZSpace (the number of elements contained in it), and thus of a contiguous slice of a
tensor; is the product of the size of
the inclusive bounds of that slice; aka, the *shape* of the tensor.

* In $\mathbb{Z}^1$, simple arrays, the size is trivially the length of the array;
* In $\mathbb{Z}^2$, simple matrices, the size is $rows * cols$, the product of the dimensions;
* and so on for $n >= 1$

However, consider the $0$-dimensional space $\mathbb{Z}^0$. The product of an empty collection is defined as $1$;
as this is the most consistent answer for a "zero" for multiplication; so we have this argument for
the existence of $0$-dimensional tensors which still have one element in them; purely from
the math of the product of shapes.

And it turns out, that's how all tensor libraries model scalar values; as $0$-dimensional tensors.


