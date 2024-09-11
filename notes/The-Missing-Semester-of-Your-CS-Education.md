# The Missing Semester of Your CS Education.

[webpage](https://missing.csail.mit.edu)

Brushing up on basics.

## Course Overview + The Shell.

* Spaces are important. They help the shell identify separate arguments. `foo=bar` works but, `foo = bar` does not.
* Environment variables, accessible by all processes, allow us to access built-in functions. Eg. `$HOME` for the location of the home directory, and `echo` is the same as `/bin/echo`
* `cd -` takes us to the previous directory.
* Rewiring input and output streams: `command < input.txt > output.txt`; `>>` appends to the file. Programs on opposite sides of `<`/ `>` will not be aware of each other.
* `|` pipes output of one command to another. `ls -l / | tail -n1` prints the last line of the output of `ls -l /`. `|` are powerful when chained.

## Shell Tools and Scripting.

(This is not exhaustive. The goal is not to become an expert in bash/zsh scripting, but to be able to do routine tasks efficiently through the CLI.)

* `$0` is the name of the script; `$1` ... `$9` are positional arguments; `$?` is the error code returned by the previous command; `$$` is the PID of the current shell.
* `!!` substitutes the last command.
* `cd ~` moves to home dir; `cd -` moves to previous dir.
* RegEx basics
  - `*` expands to 0 or more characters.
  - `?` expands to exactly 1 character.
  - foo{a,b,c} => fooa, foob, fooc
  - foo{a..d} => fooa, foob, fooc, food
* `find` is useful to search for files.
  - `find [path] [expression] -exec [command] {} \;`
    - `[path]` is the directory to start the search from.
    - `[expression]` is the search criteria.
    - `[command]` is the command to execute on the found files.
    - `{}` is a placeholder for the found file.
    - `\;` is the end of the command.
  - `find . -name src -type d` finds all directories named `src` in the current directory.
  - `find . -path '*/test/*.py' -type f` finds all Python files in all possible `test` directories.
* `grep` searches for patterns in files. `grep -i` ignores case, `grep -r` searches recursively, `grep -v` inverts the search.
* Non-builtins (need `brew install <package>`)
  - `tree` helps us visualize directory structures.
  - `tldr` provides examples for built-ins v/s navigating terse `man` pages.

