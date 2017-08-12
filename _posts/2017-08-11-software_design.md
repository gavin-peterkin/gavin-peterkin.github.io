---
layout: post
title:  "Python Design?"
date:   2017-08-10 16:38:14 -0700
categories: Programming
---

## Intro

I decided to make this post to go along with [this repository](https://github.com/gavin-peterkin/splter)
that I've been working on for the past week.
I see it as an example of how I would develop a simple python program today, so
I expect to look back at it in a year's time with some disappointment!

## Matlab

I started learning python about 2 or 3 years ago. Before that I'd learned Matlab/octave
at school in a required introductory programming course. As far as I know, Matlab
is used almost exclusively in academia and maybe some engineering fields, which have
also failed to adopt better technologies.

What I hate(d) about Matlab are the following:
1. It's expensive. A starving student license costs $99 and only gives you access
to the license for the duration of your academic career.
2. It has a very small community of users that are also in academia and only using
it because they have to. A small community means that when you google how to do
something, you'll have a _much_ harder time finding a good answer.
3. Because it's so expensive, it makes your code less shareable. If you're doing
science or engineering work, you should aim to have reproducible
results. Using expensive closed-off software makes it harder for people to reproduce
or check your work.

Towards the end of the course we started learning about object oriented programming
(OOP) _using Matlab_...

I should also state that my college is no longer using Matlab
in the introductory programming course. They now use Python!

## Developing in Python

I have no formal training in software development, but I've learned a lot
in the past year and a half from colleagues and lots of online research.
I've had the pleasure (pain) of working on legacy python projects that had files
with thousands of lines of function definitions and zero classes.
I've worked on spaghetti code, and when I first started, I probably
wrote my fair share of spaghetti code as well.

## Really basic principles I try to follow

1. **Modularity and Organization**:
Organizing code into different components that _make physical sense_. By that I mean
things that are similar should be close together.
For "splter",
I used classes that represent two separate structures: a User and a Ledger, which
are able to interact via their own methods. I also tried to write the code
hierarchically. The uppermost level, "Main", is incredibly easy to follow. Unfortunately,
I could have done a better job naming and organizing some of the lower-level functions within the Ledger
and User classes, but that is a genuinely hard problem.

2. **Maintainability and Extensibility**:
I've done
a lot of debugging using `pdb`. If it's easy to figure out where things are going wrong,
`pdb` makes it relatively easy to fix bugs. If the traceback is
multiple layers deep, it starts to get harder to understand. In other words, functional,
lazy code, is more often than not easier to understand (at least to me!). Modularity
also contributes a lot to maintainability and extensibility.

3. **Usability and Simplicity**:
If other people can't follow your code, they won't use it. Even worse,
_you_ will forget what it was doing when you come back to it in a few months, which
means it's not really extensible.

## Speed

In the real world, I try to stick to these ideas, but sometimes it just isn't that important.
If I'm just working in a small jupyter notebook that I know nobody else will see,
there's no point in wasting time worrying about the finer details of software design. A great
thing about python is that it's fast and easy to make new stuff.

I took some shortcuts when working on [splter](https://github.com/gavin-peterkin/splter),
but my main hope is that when or if I decide to work on it again in a year's time,
I'll still be able to figure out what the heck I was thinking.
