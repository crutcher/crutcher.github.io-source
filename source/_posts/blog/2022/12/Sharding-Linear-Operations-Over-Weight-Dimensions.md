---
title: Sharding Linear Operations Over Weight Dimensions
date: 2022-12-17 09:34:34
tags: ["tensor expressions", "tapestry"]
mathjax: true
---

```graphviz
digraph G {
    rankdir=LR;

    x [
        shape="plain",
        label=<
        <table bgcolor="#D5F5E3" cellpadding="8">
            <tr>
                <td>x<sub>i,m</sub></td>
                <td>…</td>
                </tr>
            <tr>
                <td>…</td>
                <td>⋱</td>
                </tr>
            </table>
        >,
    ];

    w [
        shape="plain",
        label=<
        <table bgcolor="#D5F5E3" cellpadding="8">
            <tr>
                <td bgcolor="#D6EAF8">w<sub>m,n</sub></td>
                <td bgcolor="#EBDEF0">…</td>
                </tr>
            <tr>
                <td bgcolor="#D6EAF8">…</td>
                <td bgcolor="#EBDEF0">⋱</td>
                </tr>
            </table>
        >,
    ];

    b [
        shape="plain",
        label=<
        <table bgcolor="#D5F5E3" cellpadding="8">
            <tr>
                <td bgcolor="#D6EAF8">b<sub>n</sub></td>
                <td bgcolor="#EBDEF0">…</td>
                </tr>
            </table>
        >,
    ];


    subgraph cluster_0 {

    w0 [
        shape="plain",
        label=<
        <table bgcolor="#D6EAF8" cellpadding="8">
            <tr>
                <td>w<sub>0,0</sub></td>
                <td>…</td>
                <td>w<sub>0,k</sub></td>
                </tr>
            <tr>
                <td>…</td>
                <td>…</td>
                <td>…</td>
                </tr>
            <tr>
                <td>w<sub>i,0</sub></td>
                <td>…</td>
                <td>w<sub>i,k</sub></td>
                </tr>
            </table>
        >,
    ];

    b0 [
        shape="plain",
        label=<
        <table bgcolor="#D6EAF8" cellpadding="8">
            <tr>
                <td>b<sub>0</sub></td>
                <td>…</td>
                <td>b<sub>k</sub></td>
                </tr>
            </table>
        >,
    ];

    op0 [
        label=<Linear<sub>0</sub>>,
        shape=rarrow,
        style=filled,
        fillcolor="#E5E8E8",
        margin=0.3
    ];

    y0 [
        shape="plain",
        label=<
        <table bgcolor="#D6EAF8" cellpadding="8">
            <tr>
                <td>y<sub>0,0</sub></td>
                <td>…</td>
                <td>y<sub>0,k</sub></td>
                </tr>
            <tr>
                <td>…</td>
                <td>…</td>
                <td>…</td>
                </tr>
            <tr>
                <td>y<sub>i,0</sub></td>
                <td>…</td>
                <td>y<sub>i,k</sub></td>
                </tr>
            </table>
        >,
    ];

    b0 -> op0;
    op0 -> y0;

    }

    subgraph cluster_1 {

    wk [
        shape="plain",
        label=<
        <table bgcolor="#EBDEF0" cellpadding="8">
            <tr>
                <td>w<sub>0,k+1</sub></td>
                <td>…</td>
                <td>w<sub>0,n</sub></td>
                </tr>
            <tr>
                <td>…</td>
                <td>…</td>
                <td>…</td>
                </tr>
            <tr>
                <td>w<sub>i,k+1</sub></td>
                <td>…</td>
                <td>w<sub>i,n</sub></td>
                </tr>
            </table>
        >,
    ];

    bk [
        shape="plain",
        label=<
        <table bgcolor="#EBDEF0" cellpadding="8">
            <tr>
                <td>b<sub>k+1</sub></td>
                <td>…</td>
                <td>b<sub>n</sub></td>
                </tr>
            </table>
        >,
    ];

    yk [
        shape="plain",
        label=<
        <table bgcolor="#EBDEF0" cellpadding="8">
            <tr>
                <td>y<sub>0,k+1</sub></td>
                <td>…</td>
                <td>y<sub>0,n</sub></td>
                </tr>
            <tr>
                <td>…</td>
                <td>…</td>
                <td>…</td>
                </tr>
            <tr>
                <td>y<sub>i,k+1</sub></td>
                <td>…</td>
                <td>y<sub>i,n</sub></td>
                </tr>
            </table>
        >,
    ];

    opk [
        label=<Linear<sub>1</sub>>,
        shape=rarrow,
        style=filled,
        fillcolor="#E5E8E8",
        margin=0.3
    ];

    bk -> opk;
    opk -> yk;

    }

    y [
        shape="plain",
        label=<
        <table cellpadding="8">
            <tr>
                <td bgcolor="#D6EAF8">y<sub>i,n</sub></td>
                <td bgcolor="#EBDEF0">…</td>
                </tr>
            <tr>
                <td bgcolor="#D6EAF8">…</td>
                <td bgcolor="#EBDEF0">⋱</td>
                </tr>
            </table>
        >,
    ];


    x -> op0 [weight=0];
    x -> opk [weight=0];

    w -> w0;
    w -> wk;

    b -> b0;
    b -> bk;

    w0 -> op0;
    wk -> opk;

    y0 -> y;
    yk -> y;
}
```

```graphviz
digraph G {
  rankdir=LR;

  x [
      shape="plain",
      label=<
      <table cellpadding="8">
          <tr>
              <td border="3" bgcolor="#D6EAF8">x<sub>i,m</sub></td>
              <td border="3" bgcolor="#D6EAF8">…</td>
              <td>⋰</td>
              </tr>
          <tr>
              <td>…</td>
              <td>…</td>
              <td>…</td>
              </tr>
          <tr>
          	<td>⋰</td>
              <td>…</td>
          	<td>⋱</td>
              </tr>
          </table>
      >,
  ];

  w [
      shape="plain",
      label=<
      <table cellpadding="8">
          <tr>
              <td border="3" bgcolor="#D6EAF8">w<sub>m,n</sub></td>
              <td>…</td>
              <td>⋰</td>
              </tr>
          <tr>
              <td border="3" bgcolor="#D6EAF8">…</td>
              <td>…</td>
              <td>…</td>
              </tr>
          <tr>
          	<td>⋰</td>
              <td>…</td>
          	<td>⋱</td>
              </tr>
          </table>
      >,
  ];

  subgraph cluster_0 {
    idx0 [
      shape="plain",
      label=<
          <table border="0">
      <tr><td>
        <table cellpadding="8">
          <tr><td>
        
        <table cellpadding="8">
          <tr>
            <td bgcolor="#D6EAF8">i,n,k</td>
            <td>⋰</td>
            </tr>
          <tr>
            <td>⋰</td>
            <td>⋱</td>
            </tr>
          </table>
            </td></tr>
          </table>
            </td></tr>
      <tr><td><i>matmul index</i></td></tr>
            </table>
      >,
    ];

    x0 [
        shape="plain",
        label=<
        <table cellpadding="8">
            <tr>
                <td border="3" bgcolor="#D6EAF8">x<sub>i,m</sub></td>
                <td border="3" bgcolor="#D6EAF8">…</td>
                <td>⋰</td>
                </tr>
            <tr>
                <td>…</td>
                <td>…</td>
                <td>…</td>
                </tr>
            <tr>
            	<td>⋰</td>
                <td>…</td>
            	<td>⋱</td>
                </tr>
            </table>
        >,
    ];

    w0 [
        shape="plain",
        label=<
        <table cellpadding="8">
            <tr>
                <td border="3" bgcolor="#D6EAF8">w<sub>m,n</sub></td>
                <td>…</td>
                <td>⋰</td>
                </tr>
            <tr>
                <td border="3" bgcolor="#D6EAF8">…</td>
                <td>…</td>
                <td>…</td>
                </tr>
            <tr>
            	<td>⋰</td>
                <td>…</td>
            	<td>⋱</td>
                </tr>
            </table>
        >,
    ];

    op0 [
        label=matmul,
        shape=rarrow,
        style=filled,
        fillcolor="#E5E8E8",
        margin=0.3
    ];

    z0 [
      shape="plain",
      label=<
      <table cellpadding="8">
        <tr><td>

      <table cellpadding="8">
        <tr>
          <td bgcolor="#D6EAF8">z<sub>i,k,n</sub></td>
          <td>…</td>
          </tr>
        <tr>
          <td>⋰</td>
          <td>⋱</td>
          </tr>
        </table>

	  </td></tr>
        </table>
      >,
    ];

    x0 -> op0;
    w0 -> op0;
    op0 -> z0;

    { rank=same; idx0; op0; }
  }

  x -> x0;
  z0 -> z;

  z [
    shape="plain",
    label=<
      <table cellpadding="8">
        <tr><td>
      
      <table cellpadding="8">
        <tr>
          <td>z<sub>i,k,n</sub></td>
          <td>⋰</td>
          </tr>
        <tr>
          <td>⋰</td>
          <td>⋱</td>
          </tr>
        </table>

	  </td></tr>
        </table>
    >,
  ];

  w -> w0;

  subgraph cluster_1 {
    idx1 [
      shape="plain",
      label=<
          <table border="0">
      <tr><td>
        <table cellpadding="8">
          <tr>
            <td bgcolor="#EBDEF0">i,n</td>
            <td>⋰</td>
            </tr>
          <tr>
            <td>⋰</td>
            <td>⋱</td>
            </tr>
          </table>
            </td></tr>
      <tr><td><i>sum index</i></td></tr>
            </table>
      >,
    ];

    z1 [
      shape="plain",
      label=<
      <table cellpadding="8">
        <tr><td>

      <table cellpadding="8">
        <tr>
          <td bgcolor="#EBDEF0">z<sub>i,k,n</sub></td>
          <td>…</td>
          </tr>
        <tr>
          <td>⋰</td>
          <td>⋱</td>
          </tr>
        </table>

	  </td></tr>
        </table>
      >,
    ];

    op1 [
        label=sum,
        shape=rarrow,
        style=filled,
        fillcolor="#E5E8E8",
        margin=0.3
    ];

    y1 [
      shape="plain",
      label=<
      <table cellpadding="8">
        <tr>
          <td bgcolor="#EBDEF0">y<sub>i,n</sub></td>
          <td>…</td>
          </tr>
        <tr>
          <td>⋱</td>
          <td>…</td>
          </tr>
        </table>
      >,
    ];

    b1 [
      shape="plain",
      label=<
      <table cellpadding="8">
        <tr>
          <td bgcolor="#EBDEF0">b<sub>n</sub></td>
          <td>…</td>
          </tr>
        </table>
      >,
    ];

    z1 -> op1;
    b1 -> op1;
    op1 -> y1;

    { rank=same; idx1; op1; }
  }

  z -> z1;

  b [
    shape="plain",
    label=<
    <table cellpadding="8">
      <tr>
        <td bgcolor="#EBDEF0">b<sub>n</sub></td>
        <td>…</td>
        </tr>
      </table>
    >,
  ];

  b -> b1;

  y [
    shape="plain",
    label=<
    <table cellpadding="8">
      <tr>
        <td bgcolor="#EBDEF0">y<sub>i,n</sub></td>
        <td>…</td>
        </tr>
      <tr>
        <td>⋱</td>
        <td>…</td>
        </tr>
      </table>
    >,
  ];

  y1 -> y;

  { rank=same; x; w; b; }

}
```