---
title: What Are AI Art Models?
date: 2022-12-25 11:26:25
tags: ["explainer"]
mathjax: true
---

There's a great deal of conversation at the moment about AI Art Models;
and there's a fair amount of confusion and complex feelings about them.
In online discussions, some friends have asked me to take a stab at
describing what they are, for people outside the ML community.

This post aims to explain some of what's going on from a very
high view; outside the math, to give people more grounding
in the ideas as to what's happening right now.

# De-Noising Images

All modern AI Art Models are built on a concept called "de-noising",
literally, "removing noise". Denoising models have existed in various
forms for a long time, but deep learning has made them powerful enough
to be useful for generating new data.

Let's define some basic ideas; this is a picture of my cat, Eliza:

<img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png">

If I add some random noise to the picture, I might get this:

<img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png">

Seen side-by-side, it's easy to see the noise:

<table>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png" width="200"></td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png" width="200"></td>
    </tr>
  </table>

The noise I added to mess up the image was random; but if I
already *had* the messed up image, and wanted to ask what the
difference between the noisy image and the *true* image was,
I'd see something like this, which is clearly not random:

<table>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png" width="200"></td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png" width="200"></td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.diff.0-1.png" width="200"></td>
    </tr>
  </table>

The core idea of a denoising algorithm is, given some data,
remove the data that looks like it's noise, leaving the data
that looks consistent with what we expect things *like this*
to look like.

So, given a picture like this:

<img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png">

Can I apply some magic *Enhance* button to *clean up* the image,
and get a picture like this:

<img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png">

Or side-by-side, can I go from the left, to the right:

<table>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png" width="200"></td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.diff.0-1.png" width="200"></td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png" width="200"></td>
    </tr>
  </table>

Simple, right? Just fix all the bits that *don't look like a cat*.

# Magic Enhancement Buttons

A magic [Enhance Button](https://allthetropes.org/wiki/Enhance_Button) has long
been a trope in sci-fi police procedural dramas. It's common to see a low resolution
image, taken in bad lighting, reflected
off of a surface, "cleaned up" by the data techs or AI to recover details
("the killer's face!", "the licences plate!") that were difficult or impossible
to see in the original image.

And while some information is always present which isn't easy to distinguish
by the human eye; and some amount of information recovery is possible;
there are limits, we can only recover information which is there.

This is such a common and silly trope, that it has become a point of pride
in some media and tech-savy fans to point it out when it occurs in fiction.

But ... what if we didn't care if the recovered information was *accurate*?

What if we only cared if the recovered information were *plausible*?

# Enhance My Cat

If I wished to build software which could take a noisy image of my cat,
Eliza, and clean it up, I could now do that.

I'd start by taking a *lot* of pictures of my cat, *on the same carpet*,
and messing them up to various degrees:

<table>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png" width="200"></td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png" width="200"></td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.2.png" width="200"></td>
    </tr>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.3.png" width="200"></td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.4.png" width="200"></td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.5.png" width="200"></td>
    </tr>
  </table>

And then I'd take the steps in this degeneration, say *Step K* and *Step K+1*,
and I'd train a model to go from *Step K+1* to *Step K*.

Teaching it to remove one step of noise from pictures of my cat:

<table>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png" width="200"></td>
    <td> $ \Rightarrow $ </td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png" width="200"></td>
    </tr>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.2.png" width="200"></td>
    <td> $ \Rightarrow $ </td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png" width="200"></td>
    </tr>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.3.png" width="200"></td>
    <td> $ \Rightarrow $ </td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.2.png" width="200"></td>
    </tr>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.4.png" width="200"></td>
    <td> $ \Rightarrow $ </td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.3.png" width="200"></td>
    </tr>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.5.png" width="200"></td>
    <td> $ \Rightarrow $ </td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.4.png" width="200"></td>
    </tr>
  </table>

If I trained this with just one picture, I could build a model
that would be very good at starting with random noise, and
when applied iteratively, would make it look more and more
like a picture of my cat:

<table>
  <tr>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.5.png" width="200"></td>
    <td> $ \Rightarrow $ </td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png" width="200"></td>
    </tr>
  </table>

And if I trained a model on *many* pictures of my cat
against the same background, I could start with random
noise and iteratively apply that algorithm to get
an image which was *plausibly* like a picture of my cat
against the same background.

I'm leaving a lot out, like *How*?

This is a big field of research, there's a lot of math
and statistics involved in working out how to imagine
pictures of my cat.

But what we've described so far is only good if
every picture has the same kind of content, it's
limited because we've trained one statistical model
to remove one kind of noise:
   * Noise that doesn't look like Eliza on a Carpet.

# Denoising Lots of Things

If I want to be able to *enhance* / *denoise* pictures
of lots of things other than pictures of my cat, Eliza;
the simplest way would be to just *train a lot of different models*.

Let's name the current model:
   * "Maine coon Cat laying on a white and black salt and pepper rug."

Now, my model of my cat is certainly bigger than one picture, but it's probably
a lot smaller than all the pictures I used to train it; probably a *lot*
smaller if I've used good math to do the job.

If I want to be able to denoise other things, I could train other models, say:
   * "Maine coon Cat sitting on a white and black salt and pepper rug."
   * "Maine coon Cat laying on wooden floor."
   * "Maine coon Cat sitting on a wooden floor."
   * "Planes on a runway."
   * Etc.

For a good few years, this is what people did. They built very specialized
denoising models targeting exactly what they were working with.
The models did a good job at enhancing those things with plausible
hallucinated details; but failed spectacularly when unexpected details
were mixed in.

A number of commercial image editing and enhancement packages had, internally,
large libraries of models they could use to fix up images; and they'd
first run other models to tell them what they were looking at ("that's a cat")
before selecting a model from the library to apply.

But the size of these models, and the cost of training them, while individually
small, is very large in aggregate. We can't effectively create a separate
model for every kind of image we ever might want to denoise, enhance, or just make up.

So we need to find a way to compress them together.

It would be nice if we could take a description of the kind of image
we're looking at, and turn it into some kind of key so that models can
share information.

It would be nice if these two models could share their *cat-related* information:
   * "Maine coon Cat laying on a white and black salt and pepper rug."
   * "Maine coon Cat laying on wooden floor."

But how do we translate text into a key that models can use?

# Description to Key

In 2013, along came something called [word2vec](https://en.wikipedia.org/wiki/Word2vec).

*word2vec* is a form of text-vectorization. It turned words (like "cat") into
a vector of numbers, that represented a point in a semantic embedding space.

What was *interesting* about this was that the embedding space was learned
in such a way that you could do rough "semantic math" on the results:
   * "Queen" - "Female" + "Male" =~ "King"

And that rough equivalency started to enable text manipulation (Natural Language Processing)
models to get pretty powerful.

Later, mechanisms to fuse series of words into vectors, which maintained
some notion of their original ordering, were invented.

So, today, we can perform embedding math similar to:
  * "Maine coon Cat laying on a white and black salt and pepper rug."
  * \- (offset:7)"white and black salt and pepper rug"
  * \+ (offset:7)"wooden floor"
  * =~ "Maine coon Cat laying on wooden floor."

# Overlapping Diffusion Models

Now we can train a bigger model, much bigger than we would have trained for
just one kind of thing:
   * "Maine coon Cat laying on a white and black salt and pepper rug."

But trained on many, many kinds of things. But in addition to training
the model on the noisy image, and the slightly-less noisy image; we'll
also show it the embedded description context, as a key.

<table style="text-align: center;">
  <tr>
    <td>
      <img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png" width="200">
        <br/>
      "Maine coon Cat laying on a white and black salt and pepper rug."
      </td>
    <td> $ \Rightarrow $ </td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png" width="200"></td>
    </tr>
  <tr>
    <td>
      <img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.2.png" width="200">
        <br/>
      "Maine coon Cat laying on a white and black salt and pepper rug."
      </td>
    <td> $ \Rightarrow $ </td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.noise.1.png" width="200"></td>
    </tr>
  <tr>
    <td>
      <img src="/2022/12/25/What-Are-AI-Art-Models/eliza.wood.noise.1.png" width="200">
        <br/>
      "Maine coon Cat laying on a wooden floor."
      </td>
    <td> $ \Rightarrow $ </td>
    <td><img src="/2022/12/25/What-Are-AI-Art-Models/eliza.wood.png" width="200"></td>
    </tr>
  </table>

The embedded key lets the model decide which *parts* of the model
to use to generate the image; and so parts which are shared *according to the key*
between different images will be trained together and shared in the model.

# Making More Data

Our models do better when we train them on more data; but we only have
so much data, so ... we make some up.

There are a lot of entirely mechanical things we can do which (probably)
don't change the original description.

A really common form of data-augmentation is flipping the original image:

<table style="text-align: center;">
  <tr>
    <td>
      <img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.png" width="200">
      <br/>
      "Maine coon Cat laying on a white and black salt and pepper rug."
      </td>
    <td>
      <img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.flip.png" width="200">
      <br/>
      "Maine coon Cat laying on a white and black salt and pepper rug."
      </td>
    </tr>
  <tr>
    <td>
      <img src="/2022/12/25/What-Are-AI-Art-Models/eliza.wood.png" width="200">
      <br/>
      "Maine coon Cat laying on a wooden floor."
      </td>
    <td>
      <img src="/2022/12/25/What-Are-AI-Art-Models/eliza.wood.flip.png" width="200">
      <br/>
      "Maine coon Cat laying on a wooden floor."
      </td>
    </tr>
  </table>

But if we've got access to image processing tools, we can augment
the data *and* the description:

<table style="text-align: center;">
  <tr>
    <td>
      <img src="/2022/12/25/What-Are-AI-Art-Models/eliza.source.flip.cube.png" width="200">
      <br/>
      "Maine coon Cat laying on a white and black salt and pepper rug. <b>Cubism.</b>"
      </td>
    <td>
      <img src="/2022/12/25/What-Are-AI-Art-Models/eliza.wood.oil.png" width="200">
      <br/>
      "Maine coon Cat laying on a white and black salt and pepper rug. <b>Oil painting.</b>"
      </td>
    </tr>
  </table>

# Novel Combinations

Once we've got this big overlapping model, keyed on embedded descriptions;
we can start asking it for things we *don't* have pictures of:
   * "Main coon cat laying on runway on Mars."

The question of "Does this Work?" is tied up in the math of how good
a job we did on the math for the text embeddings; and how good a
job we did on the math of how the model uses those embeddings to
decide which parts of itself to use for a given problem.

It's all about sharing the *appropriate* things; and is a very
active field of research. They year-over-year improvements in the
basic math are, at this point, astounding; and we have no reason
to currently believe that the surface has really been scratched
on the math.
   * Everyone expects this math to get *MUCH* better, quickly.

# How does this give us Art?

All AI Art Models are using variations of this technology;
and they're trained upon art, and generated and augmented
images.

Many of them are also crowd-sourcing the up and down votes on the new images
they produce, and using them to further augment and train the dataset for the
models they train.

The more images a model has been trained with, and the better the descriptions
of those images, the better the results will tend to be.

There are lots of mathy details, and lots of active research; but that's the
gist.

# Is this a copy of the training data?

The very, very short answer is: No.

Examining the data for open source AI Art Model (so we can look at the sizes of things)
[Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion)

Some numbers there:
   * trained on 2.3 billion 256x256 3-color 24 bit images
   * trained on ~= 450 terrabytes
   * model size: 4.2 gigabytes
   * 100000 / 1 ratio of training data to model size
   * ~= <b>2 bytes</b> of model data for every training image

And this is ignoring the size of the text prompts in the training;
and their representation in the model's embedding spaces.

The training data isn't in the model; the models have learned what high level
concepts imply; concepts like:
  * Cat
  * Oil Painting
  * Picasso

Now, some of those concepts are tightly coupled with the style and work
of current artists and companies with a stake in IP law and financial
survival, which brings us to the Big Question.

# Is it legal to train on copyrighted art?

This seems like a simple question, but we can probably break it down into a whole family:
   * Should it be legal to train on copyrighted art?
   * Is it legal to train on copyrighted art in jurisdiction X?
      * If no, is it legal to *look* at copyrighted art, as an artist?
      * What is the difference?
   * Should artist be a protected financial job?

But we can kick this back further:
   * When AI is taught to do a job, what do we owe to the people who are losing work to that AI?

I, and many other people, have strong opinions about these issues,
but this isn't a post about the law or the ethics of applying AI to existing industries.

Expect a *lot* of lawsuits and new legislation in this space for the rest
of ever; this question isn't going away.
