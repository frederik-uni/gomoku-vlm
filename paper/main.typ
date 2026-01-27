#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [MM],
  paper-size: "a4",
  abstract: [TRAINING],
  authors: (
    (
      name: "Frederik Schwarz",
      department: [Informatik],
      organization: [Hof-University],
    ),
  ),
  bibliography: bibliography("refs.bib"),
)
#include "struc.typ"
#include "train.typ"
