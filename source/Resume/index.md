---
title: Crutcher Dunnavant's Resume
date: 2022-12-19 17:47:57
---

# Abstract

I believe the defining technical problem of the current age is our struggles to schedule tensor expressions effectively
on large clusters. Not only AI, but a great deal of the sciences can be expressed succinctly in terms of tensor
evaluation; but are difficult to distribute. Hardware has made tremendous strides in throughput, to the point that
distribution programming models are now the dominant bottleneck.

I’ve been working to align my career around this problem; scale unlocks not only AI, but synthetic biology and material
science design. Big Tensors are the next phase of technological development for humanity.

And I want to work on meaningful problems in this space.

I’ve lately been poking at the shard scheduling problem:

* [Tapestry](/Tapestry/)

# Profile

My career has focused on pernicious distributed infrastructure problems with deep technical debt. I’ve worked a 10 year
career at Google, as a Linux OS developer for Red Hat, as the infrastructure tech lead for 3Scan, and as the
Infrastructure Director for Apprente.

I have built multiple petabyte-scale storage systems with complex and novel semantics; I have trained multiple teams up
to full-scale distributed development; I’ve run multiple overlapping development cycles with 10+ simultaneous focus
areas.

I know how to pay down technical debt and establish core system capability while incrementally migrating legacy systems
towards sanity and building out future product requirements.

I have experience teaching most things to direct reports or collaborators; believe strongly in unit testing, code
review, and pair programming; and have experience migrating complex tech stacks to these systems.

# Notable Skills and Systems Experience

## High Throughput AWS Clients

Experience performance tuning AWS client libraries for 10s of millions of robustified/retried calls. The default AWS
client libraries work reasonably well up to 10s of calls per operation; but begin to have reliability and throughput
problems at 1000s of calls per operation; and fall completely over without active work at 1000000s of calls per
operation; high-performance / high reliability systems require layers of robustification.

## High Throughput CompletableFuture Async Java Services

Experience developing and tuning async-native java services, built on hundreds of threads processing thousands of async
events for driving petabyte scale services at high cross sectional throughput.

## Log-Oriented / Append-Only Databases

Log oriented databases store mutations as log additions. The full history of all previous states of records in the
database are accessible via log scanning. I designed and developed a large log-oriented database as the backend for the
CitC Distributed Filesystem at Google, which permitted high-throughput update notification, and perfect snapshots of all
previous filesystem states.

## Distributed Query Design, Caching, and De-Normalization

Distributed systems require query interfaces, and the design of those interfaces has an outsize effect on the
correctness and speed of the whole system. I’ve have experience with the design and evolution of query interfaces for
distributed systems, including cursor design, and server and client caching mechanisms.

## Schema and Scan Design for Distributed Transactional Databases

I’ve been working with large scale distributed transactional databases in all my Google roles for the past 10 years. I
have experience in the design of schemas and scan structures for correctness and performance; and the evolution of
schema to support new features the system wasn’t previously designed for.

## Message Queue / Work Queue Systems Oriented Design

Message queue systems are a mechanism to orient processing around the asynchronous distributed delivery of information
packets; used in many distributed systems on the asynch side of their live/batch divide. I’ve built and modified them in
numerous systems.

## Distributed Load Balancing

Structuring load balancing to maximize both performance and machine utilization is a challenge, which requires analysis
of both traffic patterns, cache structure, and the performance curve of the current application (which will change with
every release). I have experience tuning and structuring load balancing systems to maximize machine utility with minimal
downtime.

## Distributed Notification / Invalidation Broadcast Systems

Mechanisms for informing distributed clients that their state is out of date represent a particular challenge in
distributed systems design. It is difficult to send each client state updates, because that requires modeling the
current state of each client (which is problematic in space as well as correctness races). The usual solution for these
mechanisms is invalidation contracts, which permit the high water mark serial of a given channel to be published to
clients, under the contract that the clients must then do a head scan from their last known state.

## API Development and Evolution

My experience with library development, my academic background in semiotics and formal language design, and my
experience with query and cache design and implementation has given me strong API design and evolution experience.

## Incremental Massive Code Refactoring / API Migration

While working at Red Hat, and particularly at Google, I have experience migrating and changing APIs via finding and
updating 1000s of code call sites; and working with testing and build frameworks to validate those changes and shepherd
them through submission. I’m comfortable with large scale refactoring systems, and incremental API migration.

## Distributed Filesystem Design and Development

I designed the CitC Distributed Filesystem (listed below in Google work experience) is a very large distributed
filesystem covering millions of workspaces and billions of files. It was built using a combination of most of the above
skills, most notably log oriented systems, load balancing, queue systems, invalidation systems, schema design, and cache
optimization.

## Sum Product Networks

Sum Product Networks are a deep learning mechanism for distribution modeling. I worked on implementing and optimizing
them for 2 years while working on distribution modeling for Google Knowledge Graph.

## Distributed Pareto Front Optimization

I built a large scale (1000s of machines) pareto front optimization infrastructure for structural optimization of Sum
Product Networks. Of particular focus was hermeticity and repeatability of the search timeline, which allowed us to
remove statistical noise when focusing on performance optimizations.

## Large MapReduce / Dataflow Algorithm Design and Optimization

Virtually every large system needs batch processing to aggregate and analyze the large quantities of information
contained within. I’ve been writing, debugging, and updating large scale MapReduce and Dataflow algorithms for 8 years.

## Parser Combinators / Domain Specific Language and Custom Parsers

My Master’s work was focused on the design and implementation of DSL parsers and system transformers; source to source
translators and transpilers; and the infrastructure (such as parser combinators and partial rewrite systems) to reduce
the difficulty and time required to devolop such systems. This tech was also applied while working on Continuity of Care
Records as part of Google Health; and really any time I encounter the need to read a dataformat I don’t have a parser
for yet. This has also given me a lot of appreciation for minimizing the need for custom formats, and focusing on using
existing structural formats with room for application growth for new applications.

## Ontology Transcoding Systems

My experience with formal languages, semiotics, and parsers gave me the background to approach ontology transcoding;
wherein we rewrite data encoded using one ontology into best-fit data in another ontology. Google Health worked some in
this space.

## Unit Testing and Testing Frameworks for Distributed Systems

I have a lot of experience writing unit tests for distributed systems, particular transaction systems and database
scanners. If you don’t have automated testing, your code is broken, you just don’t know about it yet. Unit tests
dramatically lower the cost of refactoring for future API migration. I have experience developing the unit testing
frameworks for these systems as well.

## Software Packaging / Image Packaging

Working at Red Hat, I was responsible for building and maintaining many automated installable software packages (RPMs :
Red Hat Package Manager packages). Working at Google, all software which ran on the cluster was installed from software
images. I’ve got a lot of broad experience in the development of packages and package systems.

# Engineering Leadership Life-Cycle

(Included for those considering me for leadership roles.)

Predictable engineering lifecycles are built upon a continuous dialog integrating the internal development cycles of
cross-cutting capability concerns (the development environment, the logging system, the flags system, the telemetry
systems, codec management, deployment management, core library support, common company sub-systems utilities, continuous
integration utilities, etc), which, by LOC make up the bulk of all work; with the external product development cycles of
the features which define the product or service.

I’ve seen great success driving engineering with bug-list-per-capability-group; and another list floating over
everything as the product list; this permits feature clarity of at least several months, even when running 10 focus
areas with 5-6 developers.

My engineering priorities are: change velocity, variance reduction, predictable interfaces, and core capability.

* change velocity prioritizes the ability to make changes in the future; changes which make future work easier, faster,
  or more correct are accepted; changes which make future work harder, slower, or more likely to cause errors are
  rejected.
* variance reduction prioritizes the ability to predict what the system does; unpredictable systems are expensive to
  debug and develop; predictable systems are cheap to debug and develop.
* predictable interfaces prioritizes having libraries and systems have a common predictable behaviours, free of weird
  special cases; this also reduces costs.
* core capability prioritizes placing the sharable components of a project into common libraries and services; reducing
  the cost of building other projects like this one.

My focus, as an engineering target, is upon establishing and improving the core capability of systems to solve “problems
like this one”, to view the current feature request list as an estimator for the future feature request list, and make
infrastructure choices which minimize total cost. Most research into the cost questions of software come to attribute
roughly ¾ of the total cost of any given line of code to the maintenance period of the lifecycle; but generally eng
organizations focus on the ¼, and find themselves drowning in rising technical debt. I am not a believer in Big
ReWrites, but rather in Ratchet Functions; quality boundaries which are not permitted to roll backwards.

# Work Experience

## 2021-2022: Facebook/Meta AI RAISE Program

I joined Meta to put focused IC time into working with pytorch. I wanted focused time in the trenches working on tensor
apis. Work I’ve done in this role:

* Migrating the internal bad-post model (WPIE) from one generation of AI stack to another.
* Work on the scene highlight detection model for the RayBan smart glasses.
* Research into whole-flow optimizing compilers for distributed tensors

## 2019-2020: Apprente => McDonalds, Mountain View, CA

###Infrastructure Director

I joined Apprente in Spring 2019 to upgrade their AI research prototype into a launchable product. I left when that
product was in 5 McDonald’s stores. In the time I was there, I drove the following changes:

* Implemented an automated test stack
* Drove mypy python types through the majority of the codebase
* Refactored and Cleaned the majority of the codebase
* Transitioned to hermetic Bazel builds and tests
* Restructured the App into Dockerized MicroServices communicating over MQTT
* Lead a team to build a live updated database over all orders
* Injected telemetry logging into the stack
* Debugged dozens of system crashes
* Supervised 8 reports.

## 2016-2019: 3Scan, San Francisco, CA

# Infrastructure Tech Lead

I Joined 3Scan to upgrade their development prototype to a modern distributed system. I created and trained up the
infrastructure group, developing a high throughput petabyte scale storage and analysis system.

> http://www.3scan.com/technology/
> “3Scan images entire sample volumes of over 100 cm3 at cellular level resolution, combining elements of non-invasive
> radiology and light microscopy. Our KESM can process up to 3,600 slices per hour (e.g. a whole mouse brain) of any
> tissue sample at submicron resolution (a voxel size of 0.6 um x 0.7 um x 1.0 um).
>
> To handle data outputs of up to a terabyte per cm3, 3Scan has built customized processing and analysis software. 3Scan
> offers data processing software to model 3D tissue reconstructions, provide interactive image views, and apply
> quantitative analytics.”


Major Focus Areas:

* Petabyte-Scale Storage
* Terabyte-Scale Analytics
* Java Image Processing
* BufferedImage/OpenCv
* Common Language Tooling
* Robust AWS Performance

## 6 Month Sabbatical

Left Google to decompress after 10 years.

# 2005 - 2015: Google, Mountain View, CA

## 2013-2015: Knowledge Graph Modeling, Anomaly Detection, And DeDuping

*Keywords: big data, deep learning, distributed optimization, C++*

Worked on a small team of researchers under Moshe Looks (who was under Ray Kurzweil), building a Sum Product Network
distribution modeling deep learning system. Our efforts targeted turn-key feature extraction and modeling over the
Google Knowledge Graph.

Key sub-projects I worked on:

* Distributed Model Optimization - structural alternatives were examined on 1000+ computers, and ranked generation by
  generation on a Pareto optimization front.
* Structural equivalence compression - by forcing models into equivalence classes, the search space can be dramatically
  reduced.
* Partial initialization of derived models via weight jiggling.
* Feature extraction and processing DSL
* C++ logspace quasinumber library - looks like a number, performs all fundamental ops in logspace, with support for
  Kahan summation, and negative numbers. C++ op overloads. Packed into a uint64.

## 2009-2013: Created CitC - Clients in the Cloud Global Distributed Development Filesystem

*Keywords: distributed filesystems, FUSE, append-only database, Java, C++*

Proposed, designed, and tech lead CitC (Clients in the Cloud) at Google. This is described in Rachel Potvin’s public
techtalk on Google source infrastructure, starting here:
https://youtu.be/W71BTkUbdqE?t=10m45s

Extensive article describing the system here:
http://cacm.acm.org/magazines/2016/7/204032-why-google-stores-billions-of-lines-of-code-in-a-single-repository/fulltext

CitC is a globally consistent distributed overlay filesystem for source development, built over a log transaction
database with a stable transaction history. Changes “local” to a workspace are overlaid on content already committed to
the repository. The entire history of all previous versions of all workspaces are stable. Millions of workspaces,
billions of files.

This permits extremely lightweight workspaces, which is leveraged for cloud editing, build farms, code review systems,
and all other code workflow management infrastructure at Google.

## 2007-2009: Google Health Engineer

*Keywords: Java, GWT, CCR (Continuity of Care Record), XML, XSLT*

Google Health was an early experiment (now canceled) in Personal Health Records at Google. While working on Google
Health, I developed a lot of Java backend server code. wrote a lot of front end browser code in GWT (Google Web Toolkit,
a Java-to-Javascript system); including developing a number of utility libraries still at use at Google today.

I also worked heavily on CCR (Continuity of Care Record) processing, parsing, and generation. I attended a number of
conferences and working groups for CCR, and found a few errata in the standard.

## 2005-2007: Ads Production

*Keywords: Production Testing, SQL Monitoring, Load Balancing, Global Services*

While in Ads production, I developed a production monitoring framework (Prodtest) which was used for data center
validation across Google, as well as many configuration utility libraries, to ease development of complex configuration
descriptions in Google’s custom config DSLs.

I also did a lot of work on performance balancing for compute intensive backends, and appropriate load balancing
mechanisms to prevent rolling service collapse in metastable failure states.

## Hella Tools, Hella Refactoring

I have an abiding interest in cross-cutting infrastructure concerns. As a result, I built and maintained many libraries
and infrastructure tools throughout my time at Google. Callback libraries, bug database python network bindings, math
libraries for our cluster configuration language, time series alignment and graphing utilities, several internal code
labs, GWT history management libraries. I also dove into some big API changes, refactoring 10000s of call sites.

# 2001-2005: Finish College, Grad School

Ran the AIX and Linux network for the College of Engineering.

# 2000 - 2001: Red Hat, Research Triangle, NC

## Linux OS Developer

*Keywords: Python, C, Linux Kernel, Postscript, RPM*
Worked on Red Hat’s Linux OS, as an OS Developer. Was responsible for the printing system, binding many third party
drivers into Ghostscript, printing in Asian languages. Also worked on distributed configuration management.

Also, re-wrote the Linux Kernel Magic SysRQ system while at Red Hat (not that this is a big deal):
https://www.kernel.org/doc/Documentation/sysrq.txt

# Academic Interests

Semiotics, Formal Language Systems, distributed queuing systems, distributed batch systems.

# Non-Work Experience

## Installation art

I build physical and AI installation art experiences. I’ve purpose built buildings and AI platforms for art events at
various scales.

## 2014-2018: Running a Burning Man Camp

I ran a 50 person camp at That Thing in The Desert; which is shockingly logistics heavy.

## 1995-2009: DragonCon Tech Theatre

Many years setting up, running, tearing down stages and tech for a 40k person Sci Fi convention in Atlanta, GA.

## Random

MIG welding, sewing, carpentry, basic plumbing, basic electrical, rigging

## Education

* B.S. Computer Science, University of Alabama, 2002
* M.S. Computer Science, University of Alabama, 2004
* Work towards PhD, University of Alabama, Aug 2004 - June 2005

# Technical Skills

Buzzwords of things I’ve spent real time using at various times in my career.

Java, AWS, C, C++, Python, Perl, PHP, XML, XSLT, CSS, GWT (Google Web Toolkit), Javascript, Bash, Shell, RPM, MapReduce,
Paxos, Linux Kernel, Bigtable, Message Queues, CCR (Continuity of Care Records), Distributed Systems, Threads, Async
Programming, Haskell, M4 (shudder), Deep Statistical Learning, Pareto Optimization, Sum Product Networks, Ensemble
Classifiers, Formal Language Design, DSL (domain specific language) design, RPM (Redhat Package Manager), API Design,
API Evolution, MapReduce

