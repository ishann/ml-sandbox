# The Missing Semester of Your CS Education.

[webpage](https://missing.csail.mit.edu)

Brushing up on basics.

## Course Overview + The Shell.

* Spaces are important. They help the shell identify separate arguments. `foo=bar` works but, `foo = bar` does not.
* Environment variables, accessible by all processes, allow us to access built-in functions. Eg. `$HOME` for the location of the home directory, and `echo` is the same as `/bin/echo`
* `cd -` takes us to the previous directory.
* Rewiring input and output streams: `command < input.txt > output.txt`; `>>` appends to the file. Programs on opposite sides of `<`/ `>` will not be aware of each other.
* `|` pipes output of one command to another. `ls -l / | tail -n1` prints the last line of the output of `ls -l /`. `|` are powerful when chained.

