"""The deploy gate: block a serving change that degrades measured quality.

Stdlib only, deliberately. A gate that needs a package index to decide whether
a deploy is safe fails open on a bad network day, which is the one time you
most want it to work.
"""
