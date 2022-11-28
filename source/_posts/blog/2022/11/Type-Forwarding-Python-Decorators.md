---
title: Type-Forwarding Python Decorators
date: 2022-11-27 20:28:26
tags:
---

Suppose you wish to define a proper type-forwarding decorator in python, which
supports both the common default call pattern; and the argument override call
pattern:

``` python
@foo
def some_method(x: int) -> float:
    ...

@foo(a="xyz")
def some_other_method(x: int) -> float:
    ...
```

The mechanics of this at call time are relatively straightforward in python:

``` python
def foo(
    fn = None,
    *,
    a = "xyz",
):
    def decorator(fn):
        setattr(fn, "__foo__", a)
        return fn

    if fn is None:
        return decorator

    else:
        return decorator(fn)

@foo
def some_method(x: int) -> float:
    ...

@foo
def some_other_method(x: int) -> float:
    ...
```

We can ask `mypy` the type of the resultant decorated method:

``` sh
$ mypy -c 'import simple; reveal_type(simple.some_method)'
<string>:1: note: Revealed type is "Any"
```

But establishing appropriate types, such that the types of the decorated method
are well-formed, is a challenge which requires use of `TypeVar` and the
`@overload` mechanic, and a fair amount of boilerplate:

``` python
from typing import (
    Callable,
    Optional,
    overload,
    TypeVar,
    Union,
)

C = TypeVar("C", bound=Callable)


@overload
def foo(fn: C) -> C:
    ...


@overload
def foo(
    *,
    a: Optional[str] = "xyz",
) -> Callable[[C], C]:
    ...


def foo(
    fn: Optional[C] = None,
    *,
    a: Optional[str] = "xyz",
) -> Union[Callable[[C], C], C]:
    def decorator(fn: C) -> C:
        setattr(fn, "__foo__", a)
        return fn

    if fn is None:
        return decorator

    else:
        return decorator(fn)


@foo
def foo_example(x: int, *, y: int) -> float:
    return float(x * y)
```

We can ask `mypy` the type of the decorated value:

``` sh
$ mypy -c 'import example; reveal_type(example.foo_example)'
<string>:1: note: Revealed type is "def (x: builtins.int, *, y: builtins.int) -> builtins.float"
```

Of note is that most of the actual core of this is very simple, suppose we
could say the following:

``` python
@typed_decorator
def foo(fn: C, *, a: str = "xyz") -> C:
    setattr(fn, '__foo__', a)
    return fn
```

What remains an open question to me, and I've tried many approaches, is if is
possible to define `@typed_decorator` in `mypy`s current semantics.

