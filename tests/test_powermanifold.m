clear all; close all; clc;

S = spherefactory(5, 2);
M = powermanifold(S, 7);
checkmanifold(M);
checkretraction(M);
