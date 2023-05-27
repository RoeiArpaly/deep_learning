# CS236781: Deep Learning on Computational Accelerators
# Homework Assignment 2

Faculty of Computer Science, Technion.

## Introduction

In this assignment we'll create a from-scratch implementation of two fundemental deep learning concepts: the backpropagation algorithm and stochastic gradient descent-based optimizers.
Following that we will focus on convolutional networks with residual blocks.
We'll use PyTorch to create our own network architectures and train them using GPUs, and we'll conduct architecture experiments to determine the the effects of different architectural decisions on the performance of deep networks.


## Important notes:
**1.** when you create the networks, please use pytorch blocks and not your layers and optimizers implementations!
that way you can work on diffrent parts that do not depand on each other, and if you have a bug it will be much faster to find.
If there is dependencies that we've missed around the notebooks, please notify us via the piazza so we can provide a workaround.

**2.** Due to previus homework, some of the tasks here became a bonus, try to get to them last
if there is a dependenciy of a bonus section, please use the pytorch library implementation for the bonus, so you can continue without it

## General Guidelines

- The text and code cells in these notebooks are intended to guide you through the
  assignment and help you verify your solutions.
  The notebooks **do not need to be edited** at all (unless you wish to play around).
  The only exception is to fill your name(s) in the above cell before submission.
  Please do not remove sections or change the order of any cells.
- All your code (and even answers to questions) should be written in the files
  within the python package corresponding the assignment number (`hw1`, `hw2`, etc).
  You can of course use any editor or IDE to work on these files.

## Contents
- [Part1: Backpropagation](#part1)
    - [Comparison with PyTorch](#part1_1)
    - [Layer Implementations](#part1_2)
    - [Building Models](#part1_3)
- [Part 2: Optimization and Training](#part2):
    - [Implementing Optimization Algorithms](#part2_1)
    - [Vanilla SGD with Regularization](#part2_2)
    - [Training](#part2_3)
    - [Momentum](#part2_4)
    - [RMSProp](#part2_5)
    - [Dropout Regularization](#part2_6)
    - [Questions](#part2_7)
- [Part 3: Convolutional Architectures](#part3)
    - [Convolutional layers and networks](#part3_1)
    - [Building convolutional networks with PyTorch](#part3_2)
    - [Experimenting with model architectures](#part3_3)
    - [Questions](#part3_4)
