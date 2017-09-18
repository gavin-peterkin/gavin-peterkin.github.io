---
layout: post
title:  "Python"
date:   2017-08-10 16:38:14 -0700
categories: Programming
---

## Intro

I decided to make this post to go along with [this repository](https://github.com/gavin-peterkin/splter)
that I've been working on for the past week.
I see it as an example of how I would develop a simple python program today.

# `splt`: a CLI application for tracking transactions with friends

This is a _very_ simple little program for keeping things equitable amongst a
group of friends. Users create accounts associated with a group. User accounts
also have a default "percentage" associated with it that determines that particular
user's ability to pay. It allows groups of friends to keep things equitable without
anyone ever having to go through the painful embarrassment of reminding someone
to pay them back. Just use the `--calc` option, to see if it's probably your turn to pay.

It's also transparent and open. All of the data is stored in a simple, human-readable
JSON file, and before any transactions are committed or deleted, the interface always
asks for confirmation.

## Demo

<video width="750" height="320" align="center" controls autoplay loop>
  <source type="video/mp4" src="/images/splter/splt_demo.mov"/>
  Your browser does not support the video tag.
</video>

## Developing in Python

I have no formal training in software development, but I've learned a lot
in the past year and a half from colleagues and lots of online research.
I've had the pleasure (pain) of working on legacy python projects that had files
with thousands of lines of function definitions and zero classes.
I've worked on spaghetti code, and when I first started with python, I probably
wrote my fair share of spaghetti code as well. Now, I'm also getting into learning C,
which is opening up a whole new world of lower-level computing!

## Really basic things I try to do

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

I took some shortcuts when working on [splter](https://github.com/gavin-peterkin/splter),
but my main hope is that when or if I decide to work on it again in a year's time,
I'll still be able to figure out what the heck I was thinking.

## Another thing
This ended up reminding me a lot of [this youtube video](https://www.youtube.com/watch?v=bBC-nXj3Ng4)
in which cryptocurrencies are explained in depth. My "Ledger" is actually very similar
to the simplified model of a "currency" that the video starts out with where everybody
trusts everyone else.
